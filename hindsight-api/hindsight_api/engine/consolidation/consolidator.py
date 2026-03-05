"""Consolidation engine for automatic observation creation from memories.

The consolidation engine runs as a background job after retain operations complete.
It processes new memories and either:
- Creates new observations from novel facts
- Updates existing observations when new evidence supports/contradicts/refines them

Observations are stored in memory_units with fact_type='observation' and include:
- proof_count: Number of supporting memories
- source_memory_ids: Array of memory UUIDs that contribute to this observation
- history: JSONB tracking changes over time
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ...config import get_config
from ..memory_engine import fq_table
from ..retain import embedding_utils
from .prompts import build_batch_consolidation_prompt

if TYPE_CHECKING:
    from asyncpg import Connection

    from ...api.http import RequestContext
    from ..memory_engine import MemoryEngine
    from ..response_models import MemoryFact, RecallResult

logger = logging.getLogger(__name__)


class _CreateAction(BaseModel):
    text: str
    source_fact_ids: list[str]  # memory UUIDs from the NEW FACTS list


class _UpdateAction(BaseModel):
    text: str
    observation_id: str  # UUID of the existing observation to update
    source_fact_ids: list[str]  # memory UUIDs from the NEW FACTS list


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
    }

    # Track all unique tags from consolidated memories for mental model refresh filtering
    consolidated_tags: set[str] = set()

    llm_batch_num = 0
    while True:
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
                  AND fact_type IN ('experience', 'world')
                ORDER BY created_at ASC
                LIMIT $2
                """,
                bank_id,
                max_memories_per_batch,
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

            async with pool.acquire() as conn:
                # Determine observation_scopes for this batch. All memories in a batch share
                # the same tags (enforced by tag_groups), so we only check the first memory.
                # asyncpg returns JSONB columns as raw JSON strings, so parse if needed.
                _obs_raw = llm_batch[0].get("observation_scopes") if llm_batch else None
                _obs_parsed = json.loads(_obs_raw) if isinstance(_obs_raw, str) else _obs_raw

                # Resolve the scope spec into a concrete list[list[str]] (or None for combined).
                if _obs_parsed == "per_tag":
                    _memory_tags = llm_batch[0].get("tags") or []
                    obs_tags_list = [[tag] for tag in _memory_tags] if _memory_tags else None
                elif _obs_parsed == "all_combinations":
                    _memory_tags = llm_batch[0].get("tags") or []
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

                batch_deleted: int = 0
                if obs_tags_list:
                    # Multi-pass: run one observation consolidation pass per tag set
                    results = []
                    for obs_tags in obs_tags_list:
                        pass_results, pass_deleted = await _process_memory_batch(
                            conn=conn,
                            memory_engine=memory_engine,
                            llm_config=llm_config,
                            bank_id=bank_id,
                            memories=llm_batch,
                            request_context=request_context,
                            perf=perf,
                            config=config,
                            obs_tags_override=obs_tags,
                        )
                        batch_deleted += pass_deleted
                        # Merge results: prefer non-skipped actions
                        if not results:
                            results = pass_results
                        else:
                            for i, (existing, new) in enumerate(zip(results, pass_results)):
                                if existing.get("action") == "skipped" and new.get("action") != "skipped":
                                    results[i] = new
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
                                    results[i] = {
                                        "action": "multiple",
                                        "created": existing_created + new_created,
                                        "updated": existing_updated + new_updated,
                                        "merged": 0,
                                        "total_actions": total,
                                    }
                else:
                    # Normal single pass using the memory's own tags
                    results, batch_deleted = await _process_memory_batch(
                        conn=conn,
                        memory_engine=memory_engine,
                        llm_config=llm_config,
                        bank_id=bank_id,
                        memories=llm_batch,
                        request_context=request_context,
                        perf=perf,
                        config=config,
                    )
                stats["observations_deleted"] += batch_deleted

                await conn.executemany(
                    f"UPDATE {fq_table('memory_units')} SET consolidated_at = NOW() WHERE id = $1",
                    [(m["id"],) for m in llm_batch],
                )

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
            llm_calls_made = perf.llm_calls - snap_llm_calls
            logger.info(
                f"[CONSOLIDATION] bank={bank_id} llm_batch #{llm_batch_num}"
                f" ({len(llm_batch)} memories, {llm_calls_made} llm calls)"
                f" | {stats['memories_processed']}/{total_count} processed"
                f" | {', '.join(timing_parts)}"
                f" | created={batch_created} updated={batch_updated} skipped={batch_skipped}"
                f" | input_tokens=~{input_tokens}"
                f" | avg={llm_batch_time / len(llm_batch):.3f}s/memory"
            )

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

    # Trigger mental model refreshes for models with refresh_after_consolidation=true
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

    # Find mental models with refresh_after_consolidation=true
    # SECURITY: Control which mental models get refreshed based on tags
    async with pool.acquire() as conn:
        if consolidated_tags:
            # Tagged memories were consolidated - refresh:
            # 1. Mental models with overlapping tags (security boundary)
            # 2. Untagged mental models (they're "global" and available to all contexts)
            # DO NOT refresh mental models with different tags
            rows = await conn.fetch(
                f"""
                SELECT id, name, tags
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
            # Untagged memories were consolidated - only refresh untagged mental models
            # SECURITY: Tagged mental models are NOT refreshed when untagged memories are consolidated
            rows = await conn.fetch(
                f"""
                SELECT id, name, tags
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1
                  AND (trigger->>'refresh_after_consolidation')::boolean = true
                  AND (tags IS NULL OR tags = '{{}}')
                """,
                bank_id,
            )

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
) -> tuple[list[dict[str, Any]], int]:
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

    # 3. Single LLM call
    t0 = time.time()
    llm_result = await _consolidate_batch_with_llm(
        llm_config=llm_config,
        memories=memories,
        union_observations=union_observations,
        union_source_facts=union_source_facts,
        config=config,
    )
    if perf:
        perf.record_timing("llm", time.time() - t0)
        perf.record_llm_call(llm_result.obs_count, llm_result.prompt_chars)

    # 4. Sequential execution of creates / updates / deletes
    # Track which memory indices participated so we can build per-memory results for stats
    per_memory_created: set[str] = set()
    per_memory_updated: set[str] = set()

    # Determine effective tag scope for observations.
    # When obs_tags_override is set, use it; otherwise use the memory's own tags.
    if obs_tags_override is not None:
        fact_tags = obs_tags_override
    else:
        # All memories in the batch share the same tag set (enforced by batching)
        fact_tags = memories[0].get("tags") or [] if memories else []

    mem_by_id = {str(m["id"]): m for m in memories}

    for create in llm_result.creates:
        source_mems = [mem_by_id[fid] for fid in create.source_fact_ids if fid in mem_by_id]
        if not source_mems:
            continue
        await _execute_create_action(
            conn=conn,
            memory_engine=memory_engine,
            bank_id=bank_id,
            source_memory_ids=[m["id"] for m in source_mems],
            text=create.text,
            source_fact_tags=fact_tags,
            event_date=_min_date(m.get("event_date") for m in source_mems),
            occurred_start=_min_date(m.get("occurred_start") for m in source_mems),
            occurred_end=_max_date(m.get("occurred_end") for m in source_mems),
            mentioned_at=_max_date(m.get("mentioned_at") for m in source_mems),
            perf=perf,
        )
        for m in source_mems:
            per_memory_created.add(str(m["id"]))

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
        await _execute_update_action(
            conn=conn,
            memory_engine=memory_engine,
            bank_id=bank_id,
            source_memory_ids=[m["id"] for m in source_mems],
            observation_id=update.observation_id,
            new_text=update.text,
            observations=union_observations,
            source_fact_tags=fact_tags,
            source_occurred_start=_min_date(m.get("occurred_start") for m in source_mems),
            source_occurred_end=_max_date(m.get("occurred_end") for m in source_mems),
            source_mentioned_at=_max_date(m.get("mentioned_at") for m in source_mems),
            perf=perf,
        )
        for m in source_mems:
            per_memory_updated.add(str(m["id"]))

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

    return results, deleted_count


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

    history = [
        {
            "previous_text": model.text,
            "changed_at": datetime.now(timezone.utc).isoformat(),
            "source_memory_ids": [str(mid) for mid in source_memory_ids],
        }
    ]

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

    t0 = time.time()
    await conn.execute(
        f"""
        UPDATE {fq_table("memory_units")}
        SET text = $1,
            embedding = $2::vector,
            history = $3,
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
        json.dumps(history),
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
    from ...config import get_config
    from ...tracing import get_tracer, is_tracing_enabled

    config = get_config()

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
            max_source_facts_tokens=-1,  # No token limit — we need all source facts for consolidation
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
    config: Any = None,
) -> _BatchLLMResult:
    """Single LLM call for a batch of facts against a pooled set of observations."""
    if union_observations:
        obs_list = _build_observations_for_llm(union_observations, union_source_facts)
        observations_text = json.dumps(obs_list, indent=2)
    else:
        observations_text = "[]"

    def _fact_line(m: dict[str, Any]) -> str:
        parts = [f"[{m['id']}] {m['text']}"]
        if m.get("occurred_start"):
            parts.append(f"occurred_start={m['occurred_start']}")
        if m.get("occurred_end"):
            parts.append(f"occurred_end={m['occurred_end']}")
        if m.get("mentioned_at"):
            parts.append(f"mentioned_at={m['mentioned_at']}")
        return " | ".join(parts)

    facts_lines = "\n".join(_fact_line(m) for m in memories)

    observations_mission = config.observations_mission if config is not None else None
    prompt_template = build_batch_consolidation_prompt(observations_mission)
    prompt = prompt_template.format(
        facts_text=facts_lines,
        observations_text=observations_text,
    )

    max_attempts = 3
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response: _ConsolidationBatchResponse = await llm_config.call(
                messages=[{"role": "user", "content": prompt}],
                response_format=_ConsolidationBatchResponse,
                scope="consolidation",
            )
            return _BatchLLMResult(
                creates=response.creates,
                updates=response.updates,
                deletes=response.deletes,
                obs_count=len(union_observations),
                prompt_chars=len(prompt),
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(f"[CONSOLIDATION] LLM batch call failed (attempt {attempt}/{max_attempts}): {exc}")

    logger.error(
        f"[CONSOLIDATION] LLM batch call failed after {max_attempts} attempts, skipping batch. Last error: {last_exc}"
    )
    return _BatchLLMResult(obs_count=len(union_observations), prompt_chars=len(prompt))


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
    # Generate embedding for the observation (convert to string for pgvector)
    t0 = time.time()
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [observation_text])
    embedding_str = str(embeddings[0]) if embeddings else None
    if perf:
        perf.record_timing("embedding", time.time() - t0)

    # Create the observation as a memory_unit
    now = datetime.now(timezone.utc)
    obs_event_date = event_date or now
    obs_occurred_start = occurred_start or now
    obs_occurred_end = occurred_end or now
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
