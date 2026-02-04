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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..memory_engine import fq_table
from ..retain import embedding_utils
from .prompts import (
    CONSOLIDATION_SYSTEM_PROMPT,
    CONSOLIDATION_USER_PROMPT,
)

if TYPE_CHECKING:
    from asyncpg import Connection

    from ...api.http import RequestContext
    from ..memory_engine import MemoryEngine

logger = logging.getLogger(__name__)


class ConsolidationPerfLog:
    """Performance logging for consolidation operations."""

    def __init__(self, bank_id: str):
        self.bank_id = bank_id
        self.start_time = time.time()
        self.lines: list[str] = []
        self.timings: dict[str, float] = {}

    def log(self, message: str) -> None:
        """Add a log line."""
        self.lines.append(message)

    def record_timing(self, key: str, duration: float) -> None:
        """Record a timing measurement."""
        if key in self.timings:
            self.timings[key] += duration
        else:
            self.timings[key] = duration

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
    from ...config import get_config

    config = get_config()
    perf = ConsolidationPerfLog(bank_id)
    max_memories_per_batch = config.consolidation_batch_size

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
            SELECT bank_id, name, mission
            FROM {fq_table("banks")}
            WHERE bank_id = $1
            """,
            bank_id,
        )

        if not bank_row:
            logger.warning(f"Bank {bank_id} not found for consolidation")
            return {"status": "bank_not_found", "bank_id": bank_id}

        mission = bank_row["mission"] or "General memory consolidation"
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
    stats = {
        "memories_processed": 0,
        "observations_created": 0,
        "observations_updated": 0,
        "observations_merged": 0,
        "actions_executed": 0,
        "skipped": 0,
    }

    # Track all unique tags from consolidated memories for mental model refresh filtering
    consolidated_tags: set[str] = set()

    batch_num = 0
    last_progress_timings = {}  # Track timings at last progress log
    while True:
        batch_num += 1
        batch_start = time.time()

        # Snapshot timings at batch start for per-batch calculation
        batch_start_timings = perf.timings.copy()

        # Fetch next batch of unconsolidated memories
        async with pool.acquire() as conn:
            t0 = time.time()
            memories = await conn.fetch(
                f"""
                SELECT id, text, fact_type, occurred_start, occurred_end, event_date, tags, mentioned_at
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

        for memory in memories:
            mem_start = time.time()

            # Track tags from this memory for mental model refresh filtering
            memory_tags = memory.get("tags") or []
            if memory_tags:
                consolidated_tags.update(memory_tags)

            # Process the memory (uses its own connection internally)
            async with pool.acquire() as conn:
                result = await _process_memory(
                    conn=conn,
                    memory_engine=memory_engine,
                    bank_id=bank_id,
                    memory=dict(memory),
                    mission=mission,
                    request_context=request_context,
                    perf=perf,
                )

                # Mark memory as consolidated (committed immediately)
                await conn.execute(
                    f"""
                    UPDATE {fq_table("memory_units")}
                    SET consolidated_at = NOW()
                    WHERE id = $1
                    """,
                    memory["id"],
                )

            mem_time = time.time() - mem_start
            perf.record_timing("process_memory_total", mem_time)

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

            # Log progress periodically with timing breakdown
            if stats["memories_processed"] % 10 == 0:
                # Calculate timing deltas since last progress log
                timing_parts = []
                for key in ["recall", "llm", "embedding", "db_write"]:
                    if key in perf.timings:
                        delta = perf.timings[key] - last_progress_timings.get(key, 0)
                        timing_parts.append(f"{key}={delta:.2f}s")

                timing_str = f" | {', '.join(timing_parts)}" if timing_parts else ""
                logger.info(
                    f"[CONSOLIDATION] bank={bank_id} progress: "
                    f"{stats['memories_processed']}/{total_count} memories processed{timing_str}"
                )

                # Update last progress snapshot
                last_progress_timings = perf.timings.copy()

        batch_time = time.time() - batch_start
        perf.log(
            f"[2] Batch {batch_num}: {len(memories)} memories in {batch_time:.3f}s "
            f"(avg {batch_time / len(memories):.3f}s/memory)"
        )

        # Log timing breakdown after each batch (delta from batch start)
        timing_parts = []
        for key in ["recall", "llm", "embedding", "db_write"]:
            if key in perf.timings:
                delta = perf.timings[key] - batch_start_timings.get(key, 0)
                timing_parts.append(f"{key}={delta:.3f}s")

        if timing_parts:
            avg_per_memory = batch_time / len(memories) if memories else 0
            logger.info(
                f"[CONSOLIDATION] bank={bank_id} batch {batch_num}/{len(memories)} memories: "
                f"{', '.join(timing_parts)} | avg={avg_per_memory:.3f}s/memory"
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


async def _process_memory(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    memory: dict[str, Any],
    mission: str,
    request_context: "RequestContext",
    perf: ConsolidationPerfLog | None = None,
) -> dict[str, Any]:
    """
    Process a single memory for consolidation using a SINGLE LLM call.

    This function:
    1. Finds related observations (can be empty)
    2. Uses ONE LLM call to extract durable knowledge AND decide on actions
    3. Executes array of actions (can be multiple creates/updates)

    The LLM handles all cases:
    - No related observations: returns create action(s) with extracted durable knowledge
    - Related observations exist: returns update/create actions based on tag routing
    - Purely ephemeral fact: returns empty array (skip)

    Returns:
        Dict with action summary: created/updated/merged counts
    """
    fact_text = memory["text"]
    memory_id = memory["id"]
    fact_tags = memory.get("tags") or []

    # Find related observations using the full recall system
    # SECURITY: Pass tags to ensure observations don't leak across security boundaries
    t0 = time.time()
    related_observations = await _find_related_observations(
        conn=conn,
        memory_engine=memory_engine,
        bank_id=bank_id,
        query=fact_text,
        request_context=request_context,
        tags=fact_tags,  # Pass source memory's tags for security
    )
    if perf:
        perf.record_timing("recall", time.time() - t0)

    # Single LLM call handles ALL cases (with or without existing observations)
    # Note: Tags are NOT passed to LLM - they are handled algorithmically
    t0 = time.time()
    actions = await _consolidate_with_llm(
        memory_engine=memory_engine,
        fact_text=fact_text,
        observations=related_observations,  # Can be empty list
        mission=mission,
    )
    if perf:
        perf.record_timing("llm", time.time() - t0)

    if not actions:
        # LLM returned empty array - fact is purely ephemeral, skip
        return {"action": "skipped", "reason": "no_durable_knowledge"}

    # Execute all actions and collect results
    results = []
    for action in actions:
        action_type = action.get("action")
        if action_type == "update":
            result = await _execute_update_action(
                conn=conn,
                memory_engine=memory_engine,
                bank_id=bank_id,
                memory_id=memory_id,
                action=action,
                observations=related_observations,
                source_fact_tags=fact_tags,  # Pass source fact's tags for security
                source_occurred_start=memory.get("occurred_start"),
                source_occurred_end=memory.get("occurred_end"),
                source_mentioned_at=memory.get("mentioned_at"),
                perf=perf,
            )
            results.append(result)
        elif action_type == "create":
            result = await _execute_create_action(
                conn=conn,
                memory_engine=memory_engine,
                bank_id=bank_id,
                memory_id=memory_id,
                action=action,
                source_fact_tags=fact_tags,  # Pass source fact's tags for security
                event_date=memory.get("event_date"),
                occurred_start=memory.get("occurred_start"),
                occurred_end=memory.get("occurred_end"),
                mentioned_at=memory.get("mentioned_at"),
                perf=perf,
            )
            results.append(result)

    if not results:
        # No valid actions executed
        return {"action": "skipped", "reason": "no_valid_actions"}

    # Summarize results
    created = sum(1 for r in results if r.get("action") == "created")
    updated = sum(1 for r in results if r.get("action") == "updated")
    merged = sum(1 for r in results if r.get("action") == "merged")

    if len(results) == 1:
        return results[0]

    return {
        "action": "multiple",
        "created": created,
        "updated": updated,
        "merged": merged,
        "total_actions": len(results),
    }


async def _execute_update_action(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    memory_id: uuid.UUID,
    action: dict[str, Any],
    observations: list[dict[str, Any]],
    source_fact_tags: list[str] | None = None,
    source_occurred_start: datetime | None = None,
    source_occurred_end: datetime | None = None,
    source_mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> dict[str, Any]:
    """
    Execute an update action on an existing observation.

    Updates the observation text, adds to history, increments proof_count,
    and updates temporal fields:
    - occurred_start: uses LEAST to keep the earliest start time
    - occurred_end: uses GREATEST to keep the most recent end time
    - mentioned_at: uses GREATEST to keep the most recent mention time

    SECURITY: Merges source fact's tags into the observation's existing tags.
    This ensures all contributors can see the observation they contributed to.
    For example, if Lisa's observation (tags=['user_lisa']) is updated with
    Mike's fact (tags=['user_mike']), the observation will have both tags.
    """
    learning_id = action.get("learning_id")
    new_text = action.get("text")
    reason = action.get("reason", "Updated with new fact")

    if not learning_id or not new_text:
        return {"action": "skipped", "reason": "missing_learning_id_or_text"}

    # Find the observation
    model = next((m for m in observations if str(m["id"]) == learning_id), None)
    if not model:
        return {"action": "skipped", "reason": "learning_not_found"}

    # Build history entry
    history = list(model.get("history", []))
    history.append(
        {
            "previous_text": model["text"],
            "changed_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "source_memory_id": str(memory_id),
        }
    )

    # Update source_memory_ids
    source_ids = list(model.get("source_memory_ids", []))
    source_ids.append(memory_id)

    # SECURITY: Merge source fact's tags into existing observation tags
    # This ensures all contributors can see the observation they contributed to
    existing_tags = set(model.get("tags", []) or [])
    source_tags = set(source_fact_tags or [])
    merged_tags = list(existing_tags | source_tags)  # Union of both tag sets
    if source_tags and source_tags != existing_tags:
        logger.debug(
            f"Security: Merging tags for observation {learning_id}: "
            f"existing={list(existing_tags)}, source={list(source_tags)}, merged={merged_tags}"
        )

    # Generate new embedding for updated text
    t0 = time.time()
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [new_text])
    embedding_str = str(embeddings[0]) if embeddings else None
    if perf:
        perf.record_timing("embedding", time.time() - t0)

    # Update the observation
    # - occurred_start: LEAST keeps the earliest start time across all source facts
    # - occurred_end: GREATEST keeps the most recent end time across all source facts
    # - mentioned_at: GREATEST keeps the most recent mention time
    # - tags: merged from existing + source fact (for visibility)
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
        uuid.UUID(learning_id),
        source_occurred_start,
        source_occurred_end,
        source_mentioned_at,
        merged_tags,
    )

    # Create links from memory to observation
    await _create_memory_links(conn, memory_id, uuid.UUID(learning_id))
    if perf:
        perf.record_timing("db_write", time.time() - t0)

    logger.debug(f"Updated observation {learning_id} with memory {memory_id}")

    return {"action": "updated", "observation_id": learning_id}


async def _execute_create_action(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    memory_id: uuid.UUID,
    action: dict[str, Any],
    source_fact_tags: list[str] | None = None,
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> dict[str, Any]:
    """
    Execute a create action for a new observation.

    Creates a new observation with the specified text.
    The text comes directly from the classify LLM - no second LLM call needed.

    Tags are determined algorithmically (not by LLM):
    - Observations always inherit their source fact's tags
    - This ensures visibility scope is maintained (security)
    """
    text = action.get("text")

    # Tags are determined algorithmically - always use source fact's tags
    # This ensures private memories create private observations
    tags = source_fact_tags or []

    if not text:
        return {"action": "skipped", "reason": "missing_text"}

    # Use text directly from classify - skip the redundant LLM call
    result = await _create_observation_directly(
        conn=conn,
        memory_engine=memory_engine,
        bank_id=bank_id,
        source_memory_id=memory_id,
        observation_text=text,  # Text already processed by classify LLM
        tags=tags,
        event_date=event_date,
        occurred_start=occurred_start,
        occurred_end=occurred_end,
        mentioned_at=mentioned_at,
        perf=perf,
    )

    logger.debug(f"Created observation {result.get('observation_id')} from memory {memory_id} (tags: {tags})")

    return result


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
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    query: str,
    request_context: "RequestContext",
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
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

    config = get_config()

    # SECURITY: Use all_strict matching if tags provided to prevent cross-scope consolidation
    tags_match = "all_strict" if tags else "any"

    recall_result = await memory_engine.recall_async(
        bank_id=bank_id,
        query=query,
        max_tokens=config.consolidation_max_tokens,  # Token budget for observations (configurable)
        fact_type=["observation"],  # Only retrieve observations
        request_context=request_context,
        tags=tags,  # Filter by source memory's tags
        tags_match=tags_match,  # Use strict matching for security
        _quiet=True,  # Suppress logging
    )

    # If no observations returned, return empty list
    if not recall_result.results:
        return []

    # Batch fetch all observations in a single query (no artificial limit)
    observation_ids = [uuid.UUID(obs.id) for obs in recall_result.results]

    rows = await conn.fetch(
        f"""
        SELECT id, text, proof_count, history, tags, source_memory_ids, created_at, updated_at,
               occurred_start, occurred_end, mentioned_at
        FROM {fq_table("memory_units")}
        WHERE id = ANY($1) AND bank_id = $2 AND fact_type = 'observation'
        """,
        observation_ids,
        bank_id,
    )

    # Build results list preserving recall order
    id_to_row = {row["id"]: row for row in rows}
    results = []

    for obs in recall_result.results:
        obs_id = uuid.UUID(obs.id)
        if obs_id not in id_to_row:
            continue

        row = id_to_row[obs_id]
        history = row["history"]
        if isinstance(history, str):
            history = json.loads(history)
        elif history is None:
            history = []

        # Fetch source memories to include their text and dates
        source_memory_ids = row["source_memory_ids"] or []
        source_memories = []

        if source_memory_ids:
            source_rows = await conn.fetch(
                f"""
                SELECT text, occurred_start, occurred_end, mentioned_at, event_date
                FROM {fq_table("memory_units")}
                WHERE id = ANY($1) AND bank_id = $2
                ORDER BY created_at ASC
                LIMIT 5
                """,
                source_memory_ids[:5],  # Limit to first 5 source memories for token efficiency
                bank_id,
            )

            for src_row in source_rows:
                source_memories.append(
                    {
                        "text": src_row["text"],
                        "occurred_start": src_row["occurred_start"],
                        "occurred_end": src_row["occurred_end"],
                        "mentioned_at": src_row["mentioned_at"],
                        "event_date": src_row["event_date"],
                    }
                )

        results.append(
            {
                "id": row["id"],
                "text": row["text"],
                "proof_count": row["proof_count"] or 1,
                "tags": row["tags"] or [],
                "source_memories": source_memories,
                "occurred_start": row["occurred_start"],
                "occurred_end": row["occurred_end"],
                "mentioned_at": row["mentioned_at"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    return results


async def _consolidate_with_llm(
    memory_engine: "MemoryEngine",
    fact_text: str,
    observations: list[dict[str, Any]],
    mission: str,
) -> list[dict[str, Any]]:
    """
    Single LLM call to extract durable knowledge and decide on consolidation actions.

    This handles ALL cases:
    - No related observations: extracts durable knowledge, returns create action
    - Related observations exist: compares and returns update/create actions
    - Purely ephemeral fact: returns empty array

    Note: Tags are NOT handled by the LLM. They are determined algorithmically:
    - CREATE: observation inherits source fact's tags
    - UPDATE: observation merges source fact's tags with existing tags

    Returns:
        List of actions, each being:
        - {"action": "update", "learning_id": "uuid", "text": "...", "reason": "..."}
        - {"action": "create", "text": "...", "reason": "..."}
        - [] if fact is purely ephemeral (no durable knowledge)
    """
    # Format observations as JSON with source memories and dates
    if observations:
        obs_list = []
        for obs in observations:
            obs_data = {
                "id": str(obs["id"]),
                "text": obs["text"],
                "proof_count": obs["proof_count"],
                "tags": obs["tags"],
                "created_at": obs["created_at"].isoformat() if obs.get("created_at") else None,
                "updated_at": obs["updated_at"].isoformat() if obs.get("updated_at") else None,
            }

            # Include temporal info if available
            if obs.get("occurred_start"):
                obs_data["occurred_start"] = obs["occurred_start"].isoformat()
            if obs.get("occurred_end"):
                obs_data["occurred_end"] = obs["occurred_end"].isoformat()
            if obs.get("mentioned_at"):
                obs_data["mentioned_at"] = obs["mentioned_at"].isoformat()

            # Include source memories (up to 3 for brevity)
            if obs.get("source_memories"):
                obs_data["source_memories"] = [
                    {
                        "text": sm["text"],
                        "event_date": sm["event_date"].isoformat() if sm.get("event_date") else None,
                        "occurred_start": sm["occurred_start"].isoformat() if sm.get("occurred_start") else None,
                    }
                    for sm in obs["source_memories"][:3]  # Limit to 3 for token efficiency
                ]

            obs_list.append(obs_data)

        observations_text = json.dumps(obs_list, indent=2)
    else:
        observations_text = "[]"

    # Only include mission section if mission is set and not the default
    mission_section = ""
    if mission and mission != "General memory consolidation":
        mission_section = f"""
MISSION CONTEXT: {mission}

Focus on DURABLE knowledge that serves this mission, not ephemeral state.
"""

    user_prompt = CONSOLIDATION_USER_PROMPT.format(
        mission_section=mission_section,
        fact_text=fact_text,
        observations_text=observations_text,
    )

    messages = [
        {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        result = await memory_engine._consolidation_llm_config.call(
            messages=messages,
            skip_validation=True,  # Raw JSON response
            scope="consolidation",
        )
        # Parse JSON response - should be an array
        if isinstance(result, str):
            # Strip markdown code fences (some models wrap JSON in ```json ... ```)
            clean = result.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
            result = json.loads(clean)
        # Ensure result is a list
        if isinstance(result, list):
            return result
        # Handle legacy single-action format for backward compatibility
        if isinstance(result, dict):
            if result.get("related_ids") and result.get("consolidated_text"):
                # Convert old format to new format
                return [
                    {
                        "action": "update",
                        "learning_id": result["related_ids"][0],
                        "text": result["consolidated_text"],
                        "reason": result.get("reason", ""),
                    }
                ]
            return []
        return []
    except Exception as e:
        logger.warning(f"Error in consolidation LLM call: {e}")
        return []


async def _create_observation_directly(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_id: uuid.UUID,
    observation_text: str,
    tags: list[str] | None = None,
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> dict[str, Any]:
    """
    Create an observation directly with pre-processed text (no LLM call).

    Used when the classify LLM has already provided the learning text.
    This avoids the redundant second LLM call.
    """
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
    row = await conn.fetchrow(
        f"""
        INSERT INTO {fq_table("memory_units")} (
            id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids, history,
            tags, event_date, occurred_start, occurred_end, mentioned_at
        )
        VALUES ($1, $2, $3, 'observation', $4::vector, 1, $5, '[]'::jsonb, $6, $7, $8, $9, $10)
        RETURNING id
        """,
        observation_id,
        bank_id,
        observation_text,
        embedding_str,
        [source_memory_id],
        obs_tags,
        obs_event_date,
        obs_occurred_start,
        obs_occurred_end,
        obs_mentioned_at,
    )

    # Create links between memory and observation (includes entity links, memory_links)
    await _create_memory_links(conn, source_memory_id, observation_id)
    if perf:
        perf.record_timing("db_write", time.time() - t0)

    logger.debug(f"Created observation {observation_id} from memory {source_memory_id} (tags: {obs_tags})")

    return {"action": "created", "observation_id": str(row["id"]), "tags": obs_tags}
