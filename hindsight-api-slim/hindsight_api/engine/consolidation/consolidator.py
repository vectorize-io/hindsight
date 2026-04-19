"""Consolidation engine for automatic observation creation from memories.

The consolidation engine runs as a background job after retain operations complete.
It processes new memories and either:
- Creates new observations from novel facts
- Updates existing observations when new evidence supports/contradicts/refines them

Observations are stored in memory_units with fact_type='observation' and include:
- proof_count: Number of supporting memories
- source_memory_ids: Array of memory UUIDs that contribute to this observation
- history: JSONB tracking changes over time

NOTE: Observations are distinct from mental models (pinned reflections).
- Observations: auto-generated bottom-up by this engine from raw facts (memory_units table, fact_type='observation')
- Mental models: user-defined queries stored in the mental_models table, refreshed on demand via reflect
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, field_validator

from ...config import get_config
from ..llm_wrapper import sanitize_llm_output
from ..memory_engine import fq_table
from ..retain import embedding_utils
from .prompts import build_batch_consolidation_prompt

if TYPE_CHECKING:
    from asyncpg import Connection

    from ...api.http import RequestContext
    from ..memory_engine import MemoryEngine
    from ..response_models import MemoryFact, RecallResult

logger = logging.getLogger(__name__)


async def _filter_live_source_memories(
    conn: "Connection",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
) -> list[uuid.UUID]:
    """Return only the source memory ids that still exist in the bank.

    Uses FOR SHARE to block concurrent deletes from removing a row between the
    check and the subsequent insert/update. Combined with the delete path running
    its stale-observation sweep *after* deleting the source row, this closes the
    race window where consolidation would otherwise produce an orphan observation.
    """
    if not source_memory_ids:
        return []
    rows = await conn.fetch(
        f"""
        SELECT id
        FROM {fq_table("memory_units")}
        WHERE id = ANY($1::uuid[]) AND bank_id = $2
        FOR SHARE
        """,
        source_memory_ids,
        bank_id,
    )
    live = {row["id"] for row in rows}
    return [mid for mid in source_memory_ids if mid in live]


class _CreateAction(BaseModel):
    text: str
    source_fact_ids: list[str]  # memory UUIDs from the NEW FACTS list

    @field_validator("text", mode="before")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        return sanitize_llm_output(v) or ""


class _UpdateAction(BaseModel):
    text: str
    observation_id: str  # UUID of the existing observation to update
    source_fact_ids: list[str]  # memory UUIDs from the NEW FACTS list

    @field_validator("text", mode="before")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        return sanitize_llm_output(v) or ""


class _DeleteAction(BaseModel):
    observation_id: str  # UUID of the observation to remove


class _ConsolidationBatchResponse(BaseModel):
    creates: list[_CreateAction] = []
    updates: list[_UpdateAction] = []
    deletes: list[_DeleteAction] = []


@dataclass
class _BatchLLMResult:
    creates: list[_CreateAction] = field(default_factory=list)
    updates: list[_UpdateAction] = field(default_factory=list)
    deletes: list[_DeleteAction] = field(default_factory=list)
    obs_count: int = 0
    prompt_chars: int = 0
    failed: bool = False


@dataclass
class _SourceAggregation:
    """Fields inherited by an observation from its source memories."""

    event_date: datetime | None
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime | None
    tags: list[str]


def _aggregate_source_fields(source_mems: list[dict[str, Any]], tags: list[str] | None = None) -> _SourceAggregation:
    """Compute the observation fields inherited from a set of source memories.

    Temporal aggregation rules:
    - ``event_date``    — earliest across sources (min)
    - ``occurred_start`` — earliest across sources (min)
    - ``occurred_end``   — latest across sources (max)
    - ``mentioned_at``   — latest across sources (max)

    Fields remain ``None`` when no source memory carries that information, so
    observations are never stamped with an artificial timestamp.

    ``tags`` defaults to those of the first source memory when not explicitly
    provided (all memories in a consolidation batch share the same tag set).
    """
    effective_tags = tags if tags is not None else (source_mems[0].get("tags") or [] if source_mems else [])
    return _SourceAggregation(
        event_date=_min_date(m.get("event_date") for m in source_mems),
        occurred_start=_min_date(m.get("occurred_start") for m in source_mems),
        occurred_end=_max_date(m.get("occurred_end") for m in source_mems),
        mentioned_at=_max_date(m.get("mentioned_at") for m in source_mems),
        tags=effective_tags,
    )


async def _count_observations_for_scope(
    conn: "Connection",
    bank_id: str,
    tags: list[str],
) -> int:
    """Count existing observations matching the given tag scope.

    Returns the count of observations whose tags contain all specified tags.
    Observations with no tags are not counted (the limit does not apply to them).
    """
    return await conn.fetchval(
        f"SELECT COUNT(*) FROM {fq_table('memory_units')} "
        f"WHERE bank_id = $1 AND fact_type = 'observation' AND tags @> $2::varchar[]",
        bank_id,
        tags,
    )


def _build_response_model(max_creates: int | None = None) -> type[_ConsolidationBatchResponse]:
    """Build a response model, optionally constraining max creates via JSON schema."""
    if max_creates is None or max_creates < 0:
        return _ConsolidationBatchResponse

    from pydantic import Field as PydanticField

    clamped = max(max_creates, 0)

    class _ConstrainedConsolidationBatchResponse(_ConsolidationBatchResponse):
        creates: list[_CreateAction] = PydanticField(default=[], max_length=clamped)

    return _ConstrainedConsolidationBatchResponse


class ConsolidationPerfLog:
    """Performance logging for consolidation operations."""

    def __init__(self, bank_id: str):
        self.bank_id = bank_id
        self.start_time = time.time()
        self.lines: list[str] = []
        self.timings: dict[str, float] = {}
        self.llm_calls: int = 0
        self.total_obs_in_context: int = 0
        self.total_prompt_chars: int = 0

    def log(self, message: str) -> None:
        """Add a log line."""
        self.lines.append(message)

    def record_timing(self, key: str, duration: float) -> None:
        """Record a timing measurement."""
        if key in self.timings:
            self.timings[key] += duration
        else:
            self.timings[key] = duration

    def record_llm_call(self, obs_count: int, prompt_chars: int) -> None:
        """Record stats for a single LLM call."""
        self.llm_calls += 1
        self.total_obs_in_context += obs_count
        self.total_prompt_chars += prompt_chars

    def flush(self) -> None:
        """Flush all log lines to the logger."""
        total_time = time.time() - self.start_time
        header = f"\n{'=' * 60}\nCONSOLIDATION for bank {self.bank_id}"
        footer = f"{'=' * 60}\nCONSOLIDATION COMPLETE: {total_time:.3f}s total\n{'=' * 60}"

        log_output = header + "\n" + "\n".join(self.lines) + "\n" + footer
        logger.info(log_output)


async def run_consolidation_job(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    operation_id: str | None = None,
) -> dict[str, Any]:
    """
    Run consolidation job for a bank.

    This is called after retain operations to consolidate new memories into mental models.

    Args:
        memory_engine: MemoryEngine instance
        bank_id: Bank identifier
        request_context: Request context for authentication

    Returns:
        Dict with consolidation results
    """
    # Resolve bank-specific config with hierarchical overrides
    config = await memory_engine._config_resolver.resolve_full_config(bank_id, request_context)

    # Build a configured LLM wrapper that applies per-bank settings (e.g. safety settings)
    # to every call without leaking across operations.
    llm_config = memory_engine._consolidation_llm_config.with_config(config)

    perf = ConsolidationPerfLog(bank_id)
    max_memories_per_batch = config.consolidation_batch_size
    max_memories_per_round = config.consolidation_max_memories_per_round
    llm_batch_size = max(1, config.consolidation_llm_batch_size)

    # Check if consolidation is enabled
    if not config.enable_observations:
        logger.debug(f"Consolidation disabled for bank {bank_id}")
        return {"status": "disabled", "bank_id": bank_id}

    pool = memory_engine._pool

    # Get bank profile
    async with pool.acquire() as conn:
        t0 = time.time()
        bank_row = await conn.fetchrow(
            f"""
            SELECT bank_id, name
            FROM {fq_table("banks")}
            WHERE bank_id = $1
            """,
            bank_id,
        )

        if not bank_row:
            logger.warning(f"Bank {bank_id} not found for consolidation")
            return {"status": "bank_not_found", "bank_id": bank_id}

        perf.record_timing("fetch_bank", time.time() - t0)

        # Count total unconsolidated memories for progress logging
        total_count = await conn.fetchval(
            f"""
            SELECT COUNT(*)
            FROM {fq_table("memory_units")}
            WHERE bank_id = $1
              AND consolidated_at IS NULL
              AND consolidation_failed_at IS NULL
              AND fact_type IN ('experience', 'world')
            """,
            bank_id,
        )

    if total_count == 0:
        logger.debug(f"No new memories to consolidate for bank {bank_id}")
        return {"status": "no_new_memories", "bank_id": bank_id, "memories_processed": 0}

    logger.info(f"[CONSOLIDATION] bank={bank_id} total_unconsolidated={total_count}")
    perf.log(f"[1] Found {total_count} pending memories to consolidate")

    # Process each memory with individual commits for crash recovery
    stats: dict[str, int] = {
        "memories_processed": 0,
        "observations_created": 0,
        "observations_updated": 0,
        "observations_merged": 0,
        "observations_deleted": 0,
        "actions_executed": 0,
        "skipped": 0,
        "memories_failed": 0,
    }

    # Track all unique tags from consolidated memories for mental model refresh filtering
    consolidated_tags: set[str] = set()

    round_limit_enabled = max_memories_per_round > 0
    round_remaining = max_memories_per_round if round_limit_enabled else float("inf")
    hit_round_limit = False

    llm_batch_num = 0
    while True:
        # Cap fetch size by remaining round budget
        fetch_limit = (
            min(max_memories_per_batch, int(round_remaining)) if round_limit_enabled else max_memories_per_batch
        )

        # Fetch next batch of unconsolidated memories
        async with pool.acquire() as conn:
            t0 = time.time()
            memories = await conn.fetch(
                f"""
                SELECT id, text, fact_type, occurred_start, occurred_end, event_date, tags, mentioned_at,
                       observation_scopes
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1
                  AND consolidated_at IS NULL
                  AND consolidation_failed_at IS NULL
                  AND fact_type IN ('experience', 'world')
                ORDER BY created_at ASC
                LIMIT $2
                """,
                bank_id,
                fetch_limit,
            )
            perf.record_timing("fetch_memories", time.time() - t0)

        if not memories:
            break  # No more unconsolidated memories

        # Group memories by exact tag set before batching — security requirement:
        # memories with different tags must never share an LLM call.
        tag_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for m in memories:
            tag_key = tuple(sorted(m.get("tags") or []))
            tag_groups.setdefault(tag_key, []).append(dict(m))

        # Flatten into LLM batches respecting both tag groups and llm_batch_size
        llm_batches: list[list[dict[str, Any]]] = []
        for group in tag_groups.values():
            for i in range(0, len(group), llm_batch_size):
                llm_batches.append(group[i : i + llm_batch_size])

        for llm_batch in llm_batches:
            llm_batch_num += 1
            llm_batch_start = time.time()

            # Snapshot perf and stats before this LLM batch
            snap_timings = perf.timings.copy()
            snap_llm_calls = perf.llm_calls
            snap_total_chars = perf.total_prompt_chars
            snap_stats = stats.copy()

            # Track tags for mental model refresh filtering
            for memory in llm_batch:
                memory_tags = memory.get("tags") or []
                if memory_tags:
                    consolidated_tags.update(memory_tags)

            # Process llm_batch with adaptive splitting: on LLM failure, halve the sub-batch
            # and retry, down to batch_size=1. Only if a single-memory batch still fails is
            # the memory marked with consolidation_failed_at and excluded from future runs
            # until explicitly retried via the API.
            all_results: list[dict[str, Any]] = []
            all_deleted = 0
            succeeded_ids: list[Any] = []
            failed_ids: list[Any] = []

            pending: list[list[dict[str, Any]]] = [llm_batch]
            while pending:
                sub_batch = pending.pop(0)

                async with pool.acquire() as conn:
                    # Determine observation_scopes for this sub-batch. All memories share
                    # the same tags (enforced by tag_groups), so we only check the first memory.
                    # asyncpg returns JSONB columns as raw JSON strings, so parse if needed.
                    _obs_raw = sub_batch[0].get("observation_scopes") if sub_batch else None
                    _obs_parsed = json.loads(_obs_raw) if isinstance(_obs_raw, str) else _obs_raw

                    # Resolve the scope spec into a concrete list[list[str]] (or None for combined).
                    if _obs_parsed == "per_tag":
                        _memory_tags = sub_batch[0].get("tags") or []
                        obs_tags_list = [[tag] for tag in _memory_tags] if _memory_tags else None
                    elif _obs_parsed == "all_combinations":
                        _memory_tags = sub_batch[0].get("tags") or []
                        obs_tags_list = (
                            [
                                list(combo)
                                for r in range(1, len(_memory_tags) + 1)
                                for combo in combinations(_memory_tags, r)
                            ]
                            if _memory_tags
                            else None
                        )
                    elif _obs_parsed == "combined" or _obs_parsed is None:
                        obs_tags_list = None  # single combined pass (default behaviour)
                    else:
                        # explicit list[list[str]]
                        obs_tags_list = _obs_parsed

                    sub_deleted: int = 0
                    sub_llm_failed = False
                    if obs_tags_list:
                        # Multi-pass: run one observation consolidation pass per tag set
                        sub_results: list[dict[str, Any]] = []
                        for obs_tags in obs_tags_list:
                            pass_results, pass_deleted, pass_failed = await _process_memory_batch(
                                conn=conn,
                                memory_engine=memory_engine,
                                llm_config=llm_config,
                                bank_id=bank_id,
                                memories=sub_batch,
                                request_context=request_context,
                                perf=perf,
                                config=config,
                                obs_tags_override=obs_tags,
                            )
                            sub_deleted += pass_deleted
                            sub_llm_failed = sub_llm_failed or pass_failed
                            # Merge results: prefer non-skipped actions
                            if not sub_results:
                                sub_results = pass_results
                            else:
                                for i, (existing, new) in enumerate(zip(sub_results, pass_results)):
                                    if existing.get("action") == "skipped" and new.get("action") != "skipped":
                                        sub_results[i] = new
                                    elif existing.get("action") != "skipped" and new.get("action") != "skipped":
                                        # Both did something — combine into "multiple"
                                        existing_created = existing.get(
                                            "created", 1 if existing.get("action") == "created" else 0
                                        )
                                        existing_updated = existing.get(
                                            "updated", 1 if existing.get("action") == "updated" else 0
                                        )
                                        new_created = new.get("created", 1 if new.get("action") == "created" else 0)
                                        new_updated = new.get("updated", 1 if new.get("action") == "updated" else 0)
                                        total = existing_created + existing_updated + new_created + new_updated
                                        sub_results[i] = {
                                            "action": "multiple",
                                            "created": existing_created + new_created,
                                            "updated": existing_updated + new_updated,
                                            "merged": 0,
                                            "total_actions": total,
                                        }
                    else:
                        # Normal single pass using the memory's own tags
                        sub_results, sub_deleted, sub_llm_failed = await _process_memory_batch(
                            conn=conn,
                            memory_engine=memory_engine,
                            llm_config=llm_config,
                            bank_id=bank_id,
                            memories=sub_batch,
                            request_context=request_context,
                            perf=perf,
                            config=config,
                        )

                all_deleted += sub_deleted

                if sub_llm_failed and len(sub_batch) > 1:
                    # Split and retry with smaller batches
                    mid = len(sub_batch) // 2
                    logger.warning(
                        f"[CONSOLIDATION] bank={bank_id} LLM failed for sub-batch of {len(sub_batch)},"
                        f" splitting into {mid}/{len(sub_batch) - mid}"
                    )
                    pending[0:0] = [sub_batch[:mid], sub_batch[mid:]]
                elif sub_llm_failed:
                    # batch_size=1 and still failing — mark as permanently failed for now
                    failed_ids.append(sub_batch[0]["id"])
                    all_results.append({"action": "failed"})
                    logger.warning(
                        f"[CONSOLIDATION] bank={bank_id} LLM failed for single memory"
                        f" {sub_batch[0]['id']}, marking consolidation_failed_at"
                    )
                else:
                    succeeded_ids.extend(m["id"] for m in sub_batch)
                    all_results.extend(sub_results)

            # Commit consolidated_at / consolidation_failed_at in a single DB round-trip
            async with pool.acquire() as conn:
                if succeeded_ids:
                    await conn.executemany(
                        f"UPDATE {fq_table('memory_units')} SET consolidated_at = NOW() WHERE id = $1",
                        [(mem_id,) for mem_id in succeeded_ids],
                    )
                if failed_ids:
                    await conn.executemany(
                        f"UPDATE {fq_table('memory_units')} SET consolidation_failed_at = NOW() WHERE id = $1",
                        [(mem_id,) for mem_id in failed_ids],
                    )

            stats["observations_deleted"] += all_deleted
            results = all_results

            # Checkpoint: abort if the operation (and thus the bank) was deleted mid-run.
            if operation_id and not await memory_engine._check_op_alive(operation_id):
                logger.info(
                    f"[CONSOLIDATION] bank={bank_id} operation {operation_id} cancelled (bank deleted), stopping early"
                )
                return {"status": "cancelled", "bank_id": bank_id, **stats}

            for result in results:
                stats["memories_processed"] += 1
                action = result.get("action")
                if action == "created":
                    stats["observations_created"] += 1
                    stats["actions_executed"] += 1
                elif action == "updated":
                    stats["observations_updated"] += 1
                    stats["actions_executed"] += 1
                elif action == "merged":
                    stats["observations_merged"] += 1
                    stats["actions_executed"] += 1
                elif action == "multiple":
                    stats["observations_created"] += result.get("created", 0)
                    stats["observations_updated"] += result.get("updated", 0)
                    stats["observations_merged"] += result.get("merged", 0)
                    stats["actions_executed"] += result.get("total_actions", 0)
                elif action == "skipped":
                    stats["skipped"] += 1
                elif action == "failed":
                    stats["memories_failed"] += 1

            # Per-LLM-batch log
            llm_batch_time = time.time() - llm_batch_start
            timing_parts = []
            for key in ["recall", "llm", "embedding", "db_write"]:
                if key in perf.timings:
                    delta = perf.timings[key] - snap_timings.get(key, 0)
                    timing_parts.append(f"{key}={delta:.3f}s")
            input_tokens = int((perf.total_prompt_chars - snap_total_chars) / 4)
            batch_created = stats["observations_created"] - snap_stats["observations_created"]
            batch_updated = stats["observations_updated"] - snap_stats["observations_updated"]
            batch_skipped = stats["skipped"] - snap_stats["skipped"]
            batch_failed = stats["memories_failed"] - snap_stats["memories_failed"]
            llm_calls_made = perf.llm_calls - snap_llm_calls
            logger.info(
                f"[CONSOLIDATION] bank={bank_id} llm_batch #{llm_batch_num}"
                f" ({len(llm_batch)} memories, {llm_calls_made} llm calls)"
                f" | {stats['memories_processed']}/{total_count} processed"
                f" | {', '.join(timing_parts)}"
                f" | created={batch_created} updated={batch_updated} skipped={batch_skipped}"
                + (f" failed={batch_failed}" if batch_failed else "")
                + f" | input_tokens=~{input_tokens}"
                f" | avg={llm_batch_time / len(llm_batch):.3f}s/memory"
            )

        # Update round budget after processing this DB fetch batch
        if round_limit_enabled:
            round_remaining -= len(memories)
            if round_remaining <= 0:
                hit_round_limit = True
                break

    # Re-submit consolidation if we hit the round limit and there's likely more work
    if hit_round_limit:
        remaining = total_count - stats["memories_processed"]
        logger.info(
            f"[CONSOLIDATION] bank={bank_id} hit round limit of {max_memories_per_round} memories,"
            f" ~{remaining} remaining. Re-queuing consolidation."
        )
        try:
            await memory_engine.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
        except Exception as e:
            logger.warning(f"[CONSOLIDATION] bank={bank_id} failed to re-queue consolidation: {e}")

    # Build summary
    perf.log(
        f"[3] Results: {stats['memories_processed']} memories -> "
        f"{stats['actions_executed']} actions "
        f"({stats['observations_created']} created, "
        f"{stats['observations_updated']} updated, "
        f"{stats['observations_merged']} merged, "
        f"{stats['skipped']} skipped)"
    )

    # Add timing breakdown
    timing_parts = []
    if "recall" in perf.timings:
        timing_parts.append(f"recall={perf.timings['recall']:.3f}s")
    if "llm" in perf.timings:
        timing_parts.append(f"llm={perf.timings['llm']:.3f}s")
    if "embedding" in perf.timings:
        timing_parts.append(f"embedding={perf.timings['embedding']:.3f}s")
    if "db_write" in perf.timings:
        timing_parts.append(f"db_write={perf.timings['db_write']:.3f}s")

    if perf.llm_calls > 0:
        timing_parts.append(f"avg_obs={perf.total_obs_in_context / perf.llm_calls:.1f}")
        timing_parts.append(f"avg_prompt_tokens=~{perf.total_prompt_chars / perf.llm_calls / 4:.0f}")

    if timing_parts:
        perf.log(f"[4] Timing breakdown: {', '.join(timing_parts)}")

    # Trigger mental model refreshes only on the final round (when all memories are processed).
    # If we hit the round limit and re-queued, skip MM refresh — the next round will handle it.
    if hit_round_limit:
        stats["mental_models_refreshed"] = 0
        logger.info(f"[CONSOLIDATION] bank={bank_id} skipping mental model refresh (round limit hit, re-queued)")
    else:
        # SECURITY: Only refresh mental models with matching tags (or all if no tags were consolidated)
        mental_models_refreshed = await _trigger_mental_model_refreshes(
            memory_engine=memory_engine,
            bank_id=bank_id,
            request_context=request_context,
            consolidated_tags=list(consolidated_tags) if consolidated_tags else None,
            perf=perf,
        )
        stats["mental_models_refreshed"] = mental_models_refreshed

    perf.flush()

    return {"status": "completed", "bank_id": bank_id, **stats}


async def _trigger_mental_model_refreshes(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    consolidated_tags: list[str] | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> int:
    """
    Trigger refreshes for mental models with refresh_after_consolidation=true.

    SECURITY: Only triggers refresh for mental models whose tags overlap with the
    consolidated memory tags, preventing unnecessary refreshes across security boundaries.

    Args:
        memory_engine: MemoryEngine instance
        bank_id: Bank identifier
        request_context: Request context for authentication
        consolidated_tags: Tags from memories that were consolidated (None = refresh all)
        perf: Performance logging

    Returns:
        Number of mental models scheduled for refresh
    """
    pool = memory_engine._pool

    # Find mental models with refresh_after_consolidation=true that are actually stale.
    # The tag filter on the SELECT enforces the security boundary (never look outside the
    # relevant tag scope); compute_mental_model_is_stale then verifies that new memories
    # in the MM's scope really were ingested since its last refresh.
    async with pool.acquire() as conn:
        if consolidated_tags:
            candidates = await conn.fetch(
                f"""
                SELECT id, name, tags, last_refreshed_at, trigger
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1
                  AND (trigger->>'refresh_after_consolidation')::boolean = true
                  AND (
                    (tags IS NOT NULL AND tags != '{{}}' AND tags && $2::varchar[])
                    OR (tags IS NULL OR tags = '{{}}')
                  )
                """,
                bank_id,
                consolidated_tags,
            )
        else:
            candidates = await conn.fetch(
                f"""
                SELECT id, name, tags, last_refreshed_at, trigger
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1
                  AND (trigger->>'refresh_after_consolidation')::boolean = true
                  AND (tags IS NULL OR tags = '{{}}')
                """,
                bank_id,
            )

        rows = []
        for candidate in candidates:
            if await memory_engine.compute_mental_model_is_stale(conn, bank_id, candidate):
                rows.append(candidate)

    if not rows:
        return 0

    if perf:
        if consolidated_tags:
            perf.log(
                f"[5] Triggering refresh for {len(rows)} mental models with refresh_after_consolidation=true "
                f"(filtered by tags: {consolidated_tags})"
            )
        else:
            perf.log(f"[5] Triggering refresh for {len(rows)} mental models with refresh_after_consolidation=true")

    # Submit refresh tasks for each mental model
    refreshed_count = 0
    for row in rows:
        mental_model_id = row["id"]
        try:
            await memory_engine.submit_async_refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
            )
            refreshed_count += 1
            logger.info(
                f"[CONSOLIDATION] Triggered refresh for mental model {mental_model_id} "
                f"(name: {row['name']}) in bank {bank_id}"
            )
        except Exception as e:
            logger.warning(f"[CONSOLIDATION] Failed to trigger refresh for mental model {mental_model_id}: {e}")

    return refreshed_count


async def _process_memory_batch(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    llm_config: Any,
    bank_id: str,
    memories: list[dict[str, Any]],
    request_context: "RequestContext",
    perf: ConsolidationPerfLog | None = None,
    config: Any = None,
    obs_tags_override: list[str] | None = None,
) -> tuple[list[dict[str, Any]], int, bool]:
    """
    Process a batch of memories in a single LLM call.

    Steps:
    1. Parallel recalls — one per fact (read-only; safe to parallelise)
    2. Union of retrieved observations across the batch (deduped by id)
    3. Single LLM call with all N facts + unioned observations
    4. Sequential action execution (writes remain serial for consistency)
    5. Returns one result dict per memory, in the same order as `memories`

    Per-fact security: action execution validates each learning_id against the
    observations that were recalled specifically for that fact, so cross-tag
    updates cannot occur.

    Args:
        obs_tags_override: When set, use these tags for observation recall and
            create/update instead of the memory's own tags. This enables multi-pass
            consolidation where a single memory can contribute to observations
            scoped at different tag levels (e.g., user-level vs session-level).
    """
    import asyncio

    # 1. Parallel recalls — one per fact
    # When obs_tags_override is set, use it as the observation scope for all facts.
    t0 = time.time()
    observation_scope_tags = obs_tags_override if obs_tags_override is not None else None
    recall_tasks = [
        _find_related_observations(
            memory_engine=memory_engine,
            bank_id=bank_id,
            query=m["text"],
            request_context=request_context,
            tags=observation_scope_tags if observation_scope_tags is not None else (m.get("tags") or []),
        )
        for m in memories
    ]
    per_fact_recalls = await asyncio.gather(*recall_tasks)
    if perf:
        perf.record_timing("recall", time.time() - t0)

    # 2. Build per-fact observation sets (keyed by memory ID string) for secure action validation
    per_fact_obs_ids: dict[str, set[str]] = {
        str(memories[i]["id"]): {str(obs.id) for obs in r.results} for i, r in enumerate(per_fact_recalls)
    }

    # Union all observations (deduped by id)
    seen_ids: set[str] = set()
    union_observations: list["MemoryFact"] = []
    union_source_facts: dict[str, "MemoryFact"] = {}
    for recall_result in per_fact_recalls:
        for obs in recall_result.results:
            obs_id = str(obs.id)
            if obs_id not in seen_ids:
                seen_ids.add(obs_id)
                union_observations.append(obs)
        if recall_result.source_facts:
            union_source_facts.update(recall_result.source_facts)

    # Determine effective tag scope for observations.
    # When obs_tags_override is set, use it; otherwise use the memory's own tags.
    if obs_tags_override is not None:
        fact_tags = obs_tags_override
    else:
        # All memories in the batch share the same tag set (enforced by batching)
        fact_tags = memories[0].get("tags") or [] if memories else []

    # 2b. Compute remaining observation slots for this scope (if limit configured)
    max_obs = config.max_observations_per_scope if config is not None else -1
    remaining_observation_slots: int | None = None
    if max_obs > 0 and fact_tags:
        current_count = await _count_observations_for_scope(conn, bank_id, fact_tags)
        remaining_observation_slots = max(max_obs - current_count, 0)
        if remaining_observation_slots == 0:
            logger.info(
                f"[CONSOLIDATION] bank={bank_id} scope={fact_tags} at observation limit "
                f"({current_count}/{max_obs}), only updates/deletes allowed"
            )

    # 3. Single LLM call
    t0 = time.time()
    llm_result = await _consolidate_batch_with_llm(
        llm_config=llm_config,
        memories=memories,
        union_observations=union_observations,
        union_source_facts=union_source_facts,
        config=config,
        remaining_observation_slots=remaining_observation_slots,
        max_observations_per_scope=max_obs,
    )
    if perf:
        perf.record_timing("llm", time.time() - t0)
        perf.record_llm_call(llm_result.obs_count, llm_result.prompt_chars)

    # 4. Sequential execution of deletes / updates / creates
    # Deletes run first to free observation slots before creates consume them.
    # Track which memory indices participated so we can build per-memory results for stats
    per_memory_created: set[str] = set()
    per_memory_updated: set[str] = set()

    mem_by_id = {str(m["id"]): m for m in memories}

    # Execute deletes first to free observation slots before creates consume them
    deleted_count = 0
    for delete in llm_result.deletes:
        # Security: the observation must be present in the unioned recall
        if not any(str(obs.id) == delete.observation_id for obs in union_observations):
            logger.debug(
                f"Batch consolidation: rejected delete — observation {delete.observation_id} not in unioned recall"
            )
            continue
        await _execute_delete_action(conn=conn, bank_id=bank_id, observation_id=delete.observation_id)
        deleted_count += 1

    for update in llm_result.updates:
        source_mems = [mem_by_id[fid] for fid in update.source_fact_ids if fid in mem_by_id]
        if not source_mems:
            continue
        # Security: the observation must have been recalled for at least one of the source facts
        if not any(update.observation_id in per_fact_obs_ids.get(str(m["id"]), set()) for m in source_mems):
            logger.debug(
                f"Batch consolidation: rejected update — observation {update.observation_id} "
                f"not in any source fact's recall"
            )
            continue
        agg = _aggregate_source_fields(source_mems, tags=fact_tags)
        await _execute_update_action(
            conn=conn,
            memory_engine=memory_engine,
            bank_id=bank_id,
            source_memory_ids=[m["id"] for m in source_mems],
            observation_id=update.observation_id,
            new_text=update.text,
            observations=union_observations,
            source_fact_tags=agg.tags,
            source_occurred_start=agg.occurred_start,
            source_occurred_end=agg.occurred_end,
            source_mentioned_at=agg.mentioned_at,
            perf=perf,
        )
        for m in source_mems:
            per_memory_updated.add(str(m["id"]))

    for create in llm_result.creates:
        source_mems = [mem_by_id[fid] for fid in create.source_fact_ids if fid in mem_by_id]
        if not source_mems:
            continue
        agg = _aggregate_source_fields(source_mems, tags=fact_tags)
        await _execute_create_action(
            conn=conn,
            memory_engine=memory_engine,
            bank_id=bank_id,
            source_memory_ids=[m["id"] for m in source_mems],
            text=create.text,
            source_fact_tags=agg.tags,
            event_date=agg.event_date,
            occurred_start=agg.occurred_start,
            occurred_end=agg.occurred_end,
            mentioned_at=agg.mentioned_at,
            perf=perf,
        )
        for m in source_mems:
            per_memory_created.add(str(m["id"]))

    # Build per-memory result dicts for the stats tracker in the outer loop
    results: list[dict[str, Any]] = []
    for m in memories:
        mid = str(m["id"])
        created = mid in per_memory_created
        updated = mid in per_memory_updated
        if created and updated:
            results.append({"action": "multiple", "created": 1, "updated": 1, "merged": 0, "total_actions": 2})
        elif created:
            results.append({"action": "created"})
        elif updated:
            results.append({"action": "updated"})
        else:
            results.append({"action": "skipped", "reason": "no_durable_knowledge"})

    return results, deleted_count, llm_result.failed


def _min_date(dates: "Any") -> "datetime | None":
    """Return the minimum non-None datetime from an iterable."""
    return min((d for d in dates if d is not None), default=None)


def _max_date(dates: "Any") -> "datetime | None":
    """Return the maximum non-None datetime from an iterable."""
    return max((d for d in dates if d is not None), default=None)


async def _execute_update_action(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    observation_id: str,
    new_text: str,
    observations: list["MemoryFact"],
    source_fact_tags: list[str] | None = None,
    source_occurred_start: datetime | None = None,
    source_occurred_end: datetime | None = None,
    source_mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> None:
    """
    Update an existing observation.

    Extends source_memory_ids with all contributing memories, updates temporal fields
    (LEAST for occurred_start, GREATEST for occurred_end / mentioned_at), and merges tags.
    """
    model = next((m for m in observations if str(m.id) == observation_id), None)
    if not model:
        logger.debug(f"Update skipped: observation {observation_id} not found in recall results")
        return

    live_source_memory_ids = await _filter_live_source_memories(conn, bank_id, source_memory_ids)
    if not live_source_memory_ids:
        logger.debug(
            f"Update skipped: all {len(source_memory_ids)} source memories for observation "
            f"{observation_id} were deleted concurrently"
        )
        return
    source_memory_ids = live_source_memory_ids

    from ...config import get_config

    history_entry = {
        "previous_text": model.text,
        "previous_tags": list(model.tags or []),
        "previous_occurred_start": model.occurred_start,
        "previous_occurred_end": model.occurred_end,
        "previous_mentioned_at": model.mentioned_at,
        "changed_at": datetime.now(timezone.utc).isoformat(),
        "new_source_memory_ids": [str(mid) for mid in source_memory_ids],
    }

    source_ids = list(model.source_fact_ids or []) + source_memory_ids

    # SECURITY: Merge source fact's tags into existing observation tags so all contributors can see it
    existing_tags = set(model.tags or [])
    source_tags = set(source_fact_tags or [])
    merged_tags = list(existing_tags | source_tags)

    t0 = time.time()
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [new_text])
    embedding_str = str(embeddings[0]) if embeddings else None
    if perf:
        perf.record_timing("embedding", time.time() - t0)

    config = get_config()
    history_clause = (
        "history = COALESCE(history, '[]'::jsonb) || $3::jsonb," if config.enable_observation_history else ""
    )

    t0 = time.time()
    await conn.execute(
        f"""
        UPDATE {fq_table("memory_units")}
        SET text = $1,
            embedding = $2::vector,
            {history_clause}
            source_memory_ids = $4,
            proof_count = $5,
            tags = $10,
            updated_at = now(),
            occurred_start = LEAST(occurred_start, COALESCE($7, occurred_start)),
            occurred_end = GREATEST(occurred_end, COALESCE($8, occurred_end)),
            mentioned_at = GREATEST(mentioned_at, COALESCE($9, mentioned_at))
        WHERE id = $6
        """,
        new_text,
        embedding_str,
        json.dumps([history_entry]),
        source_ids,
        len(source_ids),
        uuid.UUID(observation_id),
        source_occurred_start,
        source_occurred_end,
        source_mentioned_at,
        merged_tags,
    )
    if perf:
        perf.record_timing("db_write", time.time() - t0)

    logger.debug(f"Updated observation {observation_id} from {len(source_memory_ids)} source memories")


async def _execute_create_action(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    text: str,
    source_fact_tags: list[str] | None = None,
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> None:
    """
    Create a new observation from one or more source memories.

    Tags are inherited from the source facts (determined algorithmically, not by LLM)
    to maintain visibility scope.
    """
    await _create_observation_directly(
        conn=conn,
        memory_engine=memory_engine,
        bank_id=bank_id,
        source_memory_ids=source_memory_ids,
        observation_text=text,
        tags=source_fact_tags or [],
        event_date=event_date,
        occurred_start=occurred_start,
        occurred_end=occurred_end,
        mentioned_at=mentioned_at,
        perf=perf,
    )
    logger.debug(f"Created observation from {len(source_memory_ids)} source memories")


async def _execute_delete_action(
    conn: "Connection",
    bank_id: str,
    observation_id: str,
) -> None:
    """Delete a superseded or contradicted observation."""
    await conn.execute(
        f"DELETE FROM {fq_table('memory_units')} WHERE id = $1 AND bank_id = $2 AND fact_type = 'observation'",
        uuid.UUID(observation_id),
        bank_id,
    )
    logger.debug(f"Deleted observation {observation_id}")


async def _create_memory_links(
    conn: "Connection",
    memory_id: uuid.UUID,
    observation_id: uuid.UUID,
) -> None:
    """
    Placeholder for observation link creation.

    Observations do NOT get any memory_links copied from their source facts.
    Instead, retrieval uses source_memory_ids to traverse:
    - Entity connections: observation → source_memory_ids → unit_entities
    - Semantic similarity: observations have their own embeddings
    - Temporal proximity: observations have their own temporal fields

    This avoids data duplication and ensures observations are always
    connected via their source facts' relationships.

    The memory_id and observation_id parameters are kept for interface
    compatibility but no links are created.
    """
    # No links are created - observations rely on source_memory_ids for traversal
    pass


async def _find_related_observations(
    memory_engine: "MemoryEngine",
    bank_id: str,
    query: str,
    request_context: "RequestContext",
    tags: list[str] | None = None,
) -> "RecallResult":
    """
    Find observations related to the given query using optimized recall.

    SECURITY: Filters by tags using all_strict matching to prevent cross-tenant/cross-user
    information leakage. Observations are only consolidated within the same tag scope.

    Uses max_tokens to naturally limit observations (no artificial count limit).
    Includes source memories with dates for LLM context.

    Args:
        tags: Optional tags to filter observations (uses all_strict matching for security)

    Returns:
        List of related observations with their tags, source memories, and dates
    """
    # Use recall to find related observations with token budget
    # max_tokens naturally limits how many observations are returned
    from ...tracing import get_tracer, is_tracing_enabled

    config = await memory_engine._config_resolver.resolve_full_config(bank_id, request_context)

    # SECURITY: Use all_strict matching if tags provided to prevent cross-scope consolidation
    tags_match = "all_strict" if tags else "any"

    # Create span for recall operation within consolidation
    tracer = get_tracer()
    if is_tracing_enabled():
        recall_span = tracer.start_span("hindsight.consolidation_recall")
        recall_span.set_attribute("hindsight.bank_id", bank_id)
        recall_span.set_attribute("hindsight.query", query[:100])  # Truncate for brevity
        recall_span.set_attribute("hindsight.fact_type", "observation")
    else:
        recall_span = None

    try:
        recall_result = await memory_engine.recall_async(
            bank_id=bank_id,
            query=query,
            max_tokens=config.consolidation_max_tokens,  # Token budget for observations (configurable)
            fact_type=["observation"],  # Only retrieve observations
            request_context=request_context,
            tags=tags,  # Filter by source memory's tags
            tags_match=tags_match,  # Use strict matching for security
            include_source_facts=True,  # Embed source facts so we avoid a separate DB fetch
            max_source_facts_tokens=config.consolidation_source_facts_max_tokens,
            max_source_facts_tokens_per_observation=config.consolidation_source_facts_max_tokens_per_observation,
            _quiet=True,  # Suppress logging
        )
    finally:
        if recall_span:
            recall_span.end()

    return recall_result


def _build_observations_for_llm(
    observations: "list[MemoryFact]",
    source_facts: "dict[str, MemoryFact]",
) -> list[dict[str, Any]]:
    """Serialize MemoryFact observations into dicts for the consolidation LLM prompt."""
    obs_list = []
    for obs in observations:
        obs_data: dict[str, Any] = {
            "id": obs.id,
            "text": obs.text,
            "proof_count": len(obs.source_fact_ids or []) or 1,
        }
        if obs.occurred_start:
            obs_data["occurred_start"] = obs.occurred_start
        if obs.occurred_end:
            obs_data["occurred_end"] = obs.occurred_end
        if obs.mentioned_at:
            obs_data["mentioned_at"] = obs.mentioned_at
        source_memories = []
        for sid in obs.source_fact_ids or []:
            sf = source_facts.get(sid)
            if sf is None:
                continue
            sf_data: dict[str, Any] = {"text": sf.text}
            if sf.context:
                sf_data["context"] = sf.context
            if sf.occurred_start:
                sf_data["occurred_start"] = sf.occurred_start
            if sf.occurred_end:
                sf_data["occurred_end"] = sf.occurred_end
            if sf.mentioned_at:
                sf_data["mentioned_at"] = sf.mentioned_at
            source_memories.append(sf_data)
        if source_memories:
            obs_data["source_memories"] = source_memories
        obs_list.append(obs_data)
    return obs_list


async def _consolidate_batch_with_llm(
    llm_config: Any,
    memories: list[dict[str, Any]],
    union_observations: "list[MemoryFact]",
    union_source_facts: "dict[str, MemoryFact]",
    config: Any,
    remaining_observation_slots: int | None = None,
    max_observations_per_scope: int = -1,
) -> _BatchLLMResult:
    """Single LLM call for a batch of facts against a pooled set of observations."""
    if config is None:
        raise ValueError("config is required for _consolidate_batch_with_llm")
    if union_observations:
        obs_list = _build_observations_for_llm(union_observations, union_source_facts)
        observations_text = json.dumps(obs_list, indent=2)
    else:
        observations_text = "[]"

    def _fact_line(m: dict[str, Any]) -> str:
        text = f"[{m['id']}] {m['text']}"
        temporal_parts = []
        if m.get("occurred_start"):
            temporal_parts.append(f"occurred_start={m['occurred_start']}")
        if m.get("occurred_end"):
            temporal_parts.append(f"occurred_end={m['occurred_end']}")
        if m.get("mentioned_at"):
            temporal_parts.append(f"mentioned_at={m['mentioned_at']}")
        if temporal_parts:
            text += f" ({', '.join(temporal_parts)})"
        return text

    facts_lines = "\n".join(_fact_line(m) for m in memories)

    # Build capacity note for the prompt when observation limit is configured
    observation_capacity_note: str | None = None
    if remaining_observation_slots is not None and max_observations_per_scope > 0:
        if remaining_observation_slots == 0:
            observation_capacity_note = (
                f"OBSERVATION LIMIT REACHED ({max_observations_per_scope}/{max_observations_per_scope}). "
                "Only UPDATE or DELETE existing observations. Do NOT create new ones — "
                "merge new knowledge into existing observations via UPDATE."
            )
        elif remaining_observation_slots <= len(memories):
            observation_capacity_note = (
                f"This scope has {remaining_observation_slots} observation slot(s) remaining "
                f"(out of {max_observations_per_scope}). Prefer UPDATE over CREATE when possible."
            )

    prompt_template = build_batch_consolidation_prompt(config.observations_mission, observation_capacity_note)
    prompt = prompt_template.format(
        facts_text=facts_lines,
        observations_text=observations_text,
    )

    # Use a constrained response model when observation limit is active
    response_model = _build_response_model(max_creates=remaining_observation_slots)

    max_attempts = config.consolidation_max_attempts
    inner_max_retries = config.consolidation_llm_max_retries
    last_exc: Exception | None = None
    # Pre-compute a stable identifier set for the batch so failure logs name the
    # exact memories whose consolidation is failing — without this, an opaque
    # "LLM batch call failed" line gives operators no way to find the offending
    # input until adaptive bisection narrows the batch down to a single memory.
    memory_ids = [str(m.get("id")) for m in memories]
    if len(memory_ids) <= 5:
        ids_label = ", ".join(memory_ids)
    else:
        ids_label = f"{', '.join(memory_ids[:3])}, ... +{len(memory_ids) - 3} more"
    batch_label = f"{len(memory_ids)} memories [{ids_label}]"
    for attempt in range(1, max_attempts + 1):
        try:
            call_kwargs: dict[str, Any] = {
                "messages": [{"role": "user", "content": prompt}],
                "response_format": response_model,
                "scope": "consolidation",
            }
            if inner_max_retries is not None:
                call_kwargs["max_retries"] = inner_max_retries
            response: _ConsolidationBatchResponse = await llm_config.call(**call_kwargs)
            # Defensive truncation: some LLM providers may not enforce JSON schema max_length
            creates = response.creates
            if remaining_observation_slots is not None and remaining_observation_slots >= 0:
                if len(creates) > remaining_observation_slots:
                    logger.info(
                        f"[CONSOLIDATION] Truncating {len(creates)} creates to {remaining_observation_slots} "
                        f"(max_observations_per_scope={max_observations_per_scope})"
                    )
                    creates = creates[:remaining_observation_slots]
            return _BatchLLMResult(
                creates=creates,
                updates=response.updates,
                deletes=response.deletes,
                obs_count=len(union_observations),
                prompt_chars=len(prompt),
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"[CONSOLIDATION] LLM batch call failed (attempt {attempt}/{max_attempts}) for {batch_label}: {exc}"
            )

    logger.error(
        f"[CONSOLIDATION] LLM batch call failed after {max_attempts} attempts for {batch_label}, "
        f"skipping batch. Last error: {last_exc}"
    )
    return _BatchLLMResult(obs_count=len(union_observations), prompt_chars=len(prompt), failed=True)


async def _create_observation_directly(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    observation_text: str,
    tags: list[str] | None = None,
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> dict[str, Any]:
    """Create an observation from one or more source memories with pre-processed text."""
    live_source_memory_ids = await _filter_live_source_memories(conn, bank_id, source_memory_ids)
    if not live_source_memory_ids:
        logger.debug(f"Create skipped: all {len(source_memory_ids)} source memories were deleted concurrently")
        return {"action": "skipped", "reason": "sources_deleted"}
    source_memory_ids = live_source_memory_ids

    # Generate embedding for the observation (convert to string for pgvector)
    t0 = time.time()
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [observation_text])
    embedding_str = str(embeddings[0]) if embeddings else None
    if perf:
        perf.record_timing("embedding", time.time() - t0)

    # Create the observation as a memory_unit
    now = datetime.now(timezone.utc)
    obs_event_date = event_date or now
    obs_occurred_start = occurred_start
    obs_occurred_end = occurred_end
    obs_mentioned_at = mentioned_at or now
    obs_tags = tags or []

    t0 = time.time()
    observation_id = uuid.uuid4()

    # Query varies based on text search backend
    config = get_config()
    if config.text_search_extension == "vchord":
        # VectorChord: manually tokenize and insert search_vector
        query = f"""
            INSERT INTO {fq_table("memory_units")} (
                id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids, history,
                tags, event_date, occurred_start, occurred_end, mentioned_at, search_vector
            )
            VALUES ($1, $2, $3, 'observation', $4::vector, 1, $5, '[]'::jsonb, $6, $7, $8, $9, $10,
                    tokenize($3, 'llmlingua2')::bm25_catalog.bm25vector)
            RETURNING id
        """
    else:  # native or pg_textsearch
        # Native PostgreSQL: search_vector is GENERATED ALWAYS, don't include it
        # pg_textsearch: indexes operate on base columns directly, don't populate search_vector
        query = f"""
            INSERT INTO {fq_table("memory_units")} (
                id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids, history,
                tags, event_date, occurred_start, occurred_end, mentioned_at
            )
            VALUES ($1, $2, $3, 'observation', $4::vector, 1, $5, '[]'::jsonb, $6, $7, $8, $9, $10)
            RETURNING id
        """

    row = await conn.fetchrow(
        query,
        observation_id,
        bank_id,
        observation_text,
        embedding_str,
        source_memory_ids,
        obs_tags,
        obs_event_date,
        obs_occurred_start,
        obs_occurred_end,
        obs_mentioned_at,
    )

    if perf:
        perf.record_timing("db_write", time.time() - t0)

    logger.debug(f"Created observation {observation_id} from {len(source_memory_ids)} memories (tags: {obs_tags})")

    return {"action": "created", "observation_id": str(row["id"]), "tags": obs_tags}
