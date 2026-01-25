"""Consolidation engine for automatic mental model creation from memories.

The consolidation engine runs as a background job after retain operations complete.
It processes new memories and either:
- Creates new mental models from novel facts
- Updates existing mental models when new evidence supports/contradicts/refines them

Mental models are stored in memory_units with fact_type='mental_model' and include:
- proof_count: Number of supporting memories
- source_memory_ids: Array of memory UUIDs that contribute to this mental model
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
    if not config.enable_mental_models:
        logger.debug(f"Consolidation disabled for bank {bank_id}")
        return {"status": "disabled", "bank_id": bank_id}

    # Note: For background tasks (async consolidation), the schema context is already
    # set by _handle_consolidation before calling this function. For synchronous calls,
    # the caller must ensure the schema is set via _authenticate_tenant beforehand.
    # We don't call _authenticate_tenant here because background tasks pass an empty
    # RequestContext without JWT tokens.

    async with memory_engine._pool.acquire() as conn:
        # Get bank profile and last_consolidated_at
        t0 = time.time()
        bank_row = await conn.fetchrow(
            f"""
            SELECT bank_id, name, mission, last_consolidated_at
            FROM {fq_table("banks")}
            WHERE bank_id = $1
            """,
            bank_id,
        )

        if not bank_row:
            logger.warning(f"Bank {bank_id} not found for consolidation")
            return {"status": "bank_not_found", "bank_id": bank_id}

        mission = bank_row["mission"] or "General memory consolidation"
        last_consolidated_at = bank_row["last_consolidated_at"]
        perf.record_timing("fetch_bank", time.time() - t0)

        # Fetch memories created after last_consolidated_at (exclude mental_model type)
        t0 = time.time()
        if last_consolidated_at:
            memories = await conn.fetch(
                f"""
                SELECT id, text, fact_type, occurred_start, event_date, tags
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1 AND created_at > $2
                  AND fact_type IN ('experience', 'world')
                ORDER BY created_at ASC
                LIMIT $3
                """,
                bank_id,
                last_consolidated_at,
                max_memories_per_batch,
            )
        else:
            memories = await conn.fetch(
                f"""
                SELECT id, text, fact_type, occurred_start, event_date, tags
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1
                  AND fact_type IN ('experience', 'world')
                ORDER BY created_at ASC
                LIMIT $2
                """,
                bank_id,
                max_memories_per_batch,
            )
        perf.record_timing("fetch_memories", time.time() - t0)

        if not memories:
            logger.debug(f"No new memories to consolidate for bank {bank_id}")
            # Update timestamp anyway to prevent reprocessing
            await _update_last_consolidated_at(conn, bank_id)
            return {"status": "no_new_memories", "bank_id": bank_id, "memories_processed": 0}

        logger.info(
            f"[CONSOLIDATION] bank={bank_id} memories={len(memories)} "
            f"batch_size={max_memories_per_batch} since={last_consolidated_at or 'beginning'}"
        )
        perf.log(f"[1] Found {len(memories)} pending memories to consolidate")

        # Process each memory sequentially
        # Important: We process ALL pending memories before updating the watermark
        # to avoid losing memories when many have the same timestamp
        stats = {
            "memories_processed": 0,
            "mental_models_created": 0,
            "mental_models_updated": 0,
            "mental_models_merged": 0,
            "actions_executed": 0,  # Total actions (can be > memories_processed due to multiple actions per fact)
            "skipped": 0,
        }

        # Track processed memory IDs to avoid reprocessing
        processed_ids: set[uuid.UUID] = set()
        batch_num = 0

        while memories:
            batch_num += 1
            batch_start = time.time()

            for memory in memories:
                if memory["id"] in processed_ids:
                    continue

                mem_start = time.time()
                result = await _process_memory(
                    conn=conn,
                    memory_engine=memory_engine,
                    bank_id=bank_id,
                    memory=dict(memory),
                    mission=mission,
                    request_context=request_context,
                    perf=perf,
                )
                mem_time = time.time() - mem_start
                perf.record_timing("process_memory_total", mem_time)

                processed_ids.add(memory["id"])
                stats["memories_processed"] += 1

                action = result.get("action")
                if action == "created":
                    stats["mental_models_created"] += 1
                    stats["actions_executed"] += 1
                elif action == "updated":
                    stats["mental_models_updated"] += 1
                    stats["actions_executed"] += 1
                elif action == "merged":
                    stats["mental_models_merged"] += 1
                    stats["actions_executed"] += 1
                elif action == "multiple":
                    # Multiple actions from one fact (tag routing)
                    stats["mental_models_created"] += result.get("created", 0)
                    stats["mental_models_updated"] += result.get("updated", 0)
                    stats["mental_models_merged"] += result.get("merged", 0)
                    stats["actions_executed"] += result.get("total_actions", 0)
                elif action == "skipped":
                    stats["skipped"] += 1

            batch_time = time.time() - batch_start
            perf.log(
                f"[2] Batch {batch_num}: {len(memories)} memories in {batch_time:.3f}s "
                f"(avg {batch_time / len(memories):.3f}s/memory)"
            )

            # Fetch next batch of memories (excluding already processed)
            t0 = time.time()
            if last_consolidated_at:
                memories = await conn.fetch(
                    f"""
                    SELECT id, text, fact_type, occurred_start, event_date, tags
                    FROM {fq_table("memory_units")}
                    WHERE bank_id = $1 AND created_at > $2
                      AND fact_type IN ('experience', 'world')
                      AND id != ALL($4)
                    ORDER BY created_at ASC
                    LIMIT $3
                    """,
                    bank_id,
                    last_consolidated_at,
                    max_memories_per_batch,
                    list(processed_ids),
                )
            else:
                memories = await conn.fetch(
                    f"""
                    SELECT id, text, fact_type, occurred_start, event_date, tags
                    FROM {fq_table("memory_units")}
                    WHERE bank_id = $1
                      AND fact_type IN ('experience', 'world')
                      AND id != ALL($3)
                    ORDER BY created_at ASC
                    LIMIT $2
                    """,
                    bank_id,
                    max_memories_per_batch,
                    list(processed_ids),
                )
            perf.record_timing("fetch_memories", time.time() - t0)

        # Update last_consolidated_at only after ALL memories are processed
        t0 = time.time()
        await _update_last_consolidated_at(conn, bank_id)
        perf.record_timing("update_watermark", time.time() - t0)

        # Build summary
        perf.log(
            f"[3] Results: {stats['memories_processed']} memories → "
            f"{stats['actions_executed']} actions "
            f"({stats['mental_models_created']} created, "
            f"{stats['mental_models_updated']} updated, "
            f"{stats['mental_models_merged']} merged, "
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

        perf.flush()

        return {"status": "completed", "bank_id": bank_id, **stats}


async def _update_last_consolidated_at(conn: "Connection", bank_id: str) -> None:
    """Update the bank's last_consolidated_at timestamp."""
    await conn.execute(
        f"""
        UPDATE {fq_table("banks")}
        SET last_consolidated_at = $1
        WHERE bank_id = $2
        """,
        datetime.now(timezone.utc),
        bank_id,
    )


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
    1. Finds related mental models (can be empty)
    2. Uses ONE LLM call to extract durable knowledge AND decide on actions
    3. Executes array of actions (can be multiple creates/updates)

    The LLM handles all cases:
    - No related models: returns create action(s) with extracted durable knowledge
    - Related models exist: returns update/create actions based on tag routing
    - Purely ephemeral fact: returns empty array (skip)

    Returns:
        Dict with action summary: created/updated/merged counts
    """
    fact_text = memory["text"]
    memory_id = memory["id"]
    fact_tags = memory.get("tags") or []

    # Find related mental models using the full recall system (NO tag filtering)
    t0 = time.time()
    related_mental_models = await _find_related_mental_models(
        conn=conn,
        memory_engine=memory_engine,
        bank_id=bank_id,
        query=fact_text,
        request_context=request_context,
    )
    if perf:
        perf.record_timing("recall", time.time() - t0)

    # Single LLM call handles ALL cases (with or without existing models)
    t0 = time.time()
    actions = await _consolidate_with_llm(
        memory_engine=memory_engine,
        fact_text=fact_text,
        fact_tags=fact_tags,
        mental_models=related_mental_models,  # Can be empty list
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
                mental_models=related_mental_models,
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
                event_date=memory.get("event_date"),
                occurred_start=memory.get("occurred_start"),
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
    mental_models: list[dict[str, Any]],
    perf: ConsolidationPerfLog | None = None,
) -> dict[str, Any]:
    """
    Execute an update action on an existing mental model.

    Updates the mental model text, adds to history, and increments proof_count.
    """
    learning_id = action.get("learning_id")
    new_text = action.get("text")
    reason = action.get("reason", "Updated with new fact")

    if not learning_id or not new_text:
        return {"action": "skipped", "reason": "missing_learning_id_or_text"}

    # Find the mental model
    model = next((m for m in mental_models if str(m["id"]) == learning_id), None)
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

    # Generate new embedding for updated text
    t0 = time.time()
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [new_text])
    embedding_str = str(embeddings[0]) if embeddings else None
    if perf:
        perf.record_timing("embedding", time.time() - t0)

    # Update the mental model
    t0 = time.time()
    await conn.execute(
        f"""
        UPDATE {fq_table("memory_units")}
        SET text = $1,
            embedding = $2::vector,
            history = $3,
            source_memory_ids = $4,
            proof_count = $5,
            updated_at = now()
        WHERE id = $6
        """,
        new_text,
        embedding_str,
        json.dumps(history),
        source_ids,
        len(source_ids),
        uuid.UUID(learning_id),
    )

    # Create links from memory to mental model
    await _create_memory_links(conn, memory_id, uuid.UUID(learning_id))
    if perf:
        perf.record_timing("db_write", time.time() - t0)

    logger.debug(f"Updated mental model {learning_id} with memory {memory_id}")

    return {"action": "updated", "mental_model_id": learning_id}


async def _execute_create_action(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    memory_id: uuid.UUID,
    action: dict[str, Any],
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> dict[str, Any]:
    """
    Execute a create action for a new mental model.

    Creates a new mental model with the specified text and tags.
    The text comes directly from the classify LLM - no second LLM call needed.
    """
    text = action.get("text")
    tags = action.get("tags", [])

    if not text:
        return {"action": "skipped", "reason": "missing_text"}

    # Use text directly from classify - skip the redundant LLM call
    result = await _create_mental_model_directly(
        conn=conn,
        memory_engine=memory_engine,
        bank_id=bank_id,
        source_memory_id=memory_id,
        mental_model_text=text,  # Text already processed by classify LLM
        tags=tags,
        event_date=event_date,
        occurred_start=occurred_start,
        perf=perf,
    )

    logger.debug(f"Created mental model {result.get('mental_model_id')} from memory {memory_id} (tags: {tags})")

    return result


async def _create_memory_links(
    conn: "Connection",
    memory_id: uuid.UUID,
    mental_model_id: uuid.UUID,
) -> None:
    """
    Create links between a source memory and its mental model.

    This:
    1. Creates bidirectional semantic links between memory and mental model
    2. Copies existing memory_links from the source memory to the mental model
    3. Copies entity links from the source memory to the mental model

    This enables graph traversal to find related memories via their mental models.

    Note: Uses EXISTS checks to handle the case where source memory was deleted
    by a concurrent operation between fetching and link creation.
    """
    mu_table = fq_table("memory_units")
    ml_table = fq_table("memory_links")
    ue_table = fq_table("unit_entities")

    # 1. Bidirectional link between memory and mental model
    # Only insert if both units exist (handles concurrent deletion)
    await conn.execute(
        f"""
        INSERT INTO {ml_table} (from_unit_id, to_unit_id, link_type, weight)
        SELECT $1, $2, 'semantic', 1.0
        WHERE EXISTS (SELECT 1 FROM {mu_table} WHERE id = $1)
          AND EXISTS (SELECT 1 FROM {mu_table} WHERE id = $2)
        ON CONFLICT DO NOTHING
        """,
        memory_id,
        mental_model_id,
    )
    await conn.execute(
        f"""
        INSERT INTO {ml_table} (from_unit_id, to_unit_id, link_type, weight)
        SELECT $1, $2, 'semantic', 1.0
        WHERE EXISTS (SELECT 1 FROM {mu_table} WHERE id = $1)
          AND EXISTS (SELECT 1 FROM {mu_table} WHERE id = $2)
        ON CONFLICT DO NOTHING
        """,
        mental_model_id,
        memory_id,
    )

    # 2. Copy outgoing memory_links from source memory to mental model
    # If source memory links to X, mental model should also link to X
    await conn.execute(
        f"""
        INSERT INTO {ml_table} (from_unit_id, to_unit_id, link_type, entity_id, weight)
        SELECT $1, ml.to_unit_id, ml.link_type, ml.entity_id, ml.weight
        FROM {ml_table} ml
        WHERE ml.from_unit_id = $2 AND ml.to_unit_id != $1
          AND EXISTS (SELECT 1 FROM {mu_table} WHERE id = $1)
          AND EXISTS (SELECT 1 FROM {mu_table} WHERE id = ml.to_unit_id)
        ON CONFLICT DO NOTHING
        """,
        mental_model_id,
        memory_id,
    )

    # 3. Copy incoming memory_links from source memory to mental model
    # If X links to source memory, X should also link to mental model
    await conn.execute(
        f"""
        INSERT INTO {ml_table} (from_unit_id, to_unit_id, link_type, entity_id, weight)
        SELECT ml.from_unit_id, $1, ml.link_type, ml.entity_id, ml.weight
        FROM {ml_table} ml
        WHERE ml.to_unit_id = $2 AND ml.from_unit_id != $1
          AND EXISTS (SELECT 1 FROM {mu_table} WHERE id = $1)
          AND EXISTS (SELECT 1 FROM {mu_table} WHERE id = ml.from_unit_id)
        ON CONFLICT DO NOTHING
        """,
        mental_model_id,
        memory_id,
    )

    # 4. Copy entity links from source memory to mental model
    await conn.execute(
        f"""
        INSERT INTO {ue_table} (unit_id, entity_id)
        SELECT $1, ue.entity_id
        FROM {ue_table} ue
        WHERE ue.unit_id = $2
          AND EXISTS (SELECT 1 FROM {mu_table} WHERE id = $1)
        ON CONFLICT DO NOTHING
        """,
        mental_model_id,
        memory_id,
    )


async def _find_related_mental_models(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    query: str,
    request_context: "RequestContext",
) -> list[dict[str, Any]]:
    """
    Find mental models related to the given query using the full recall system.

    IMPORTANT: We do NOT filter by tags here. Consolidation needs to see ALL
    potentially related mental models regardless of scope, so the LLM can
    decide on tag routing (same scope update vs cross-scope create).

    This leverages:
    - Semantic search (embedding similarity)
    - BM25 text search (keyword matching)
    - Entity-based retrieval (shared entities)
    - Graph traversal (connected via entity links)

    Returns:
        List of related mental models with their tags for LLM tag routing
    """
    # Use recall to find related mental models
    # NO tags parameter - we want ALL mental models regardless of scope
    # Use low max_tokens since we only need mental models, not memories
    recall_result = await memory_engine.recall_async(
        bank_id=bank_id,
        query=query,
        max_tokens=5000,  # Token budget for mental models
        fact_type=["mental_model"],  # Only retrieve mental models
        request_context=request_context,
        _quiet=True,  # Suppress logging
        # NO tags parameter - intentionally get ALL mental models
    )

    # If no mental models returned, return empty list
    # When fact_type=["mental_model"], results come back in `results` field
    if not recall_result.results:
        return []

    # Trust recall's relevance filtering - fetch full data for each mental model
    results = []
    for mm in recall_result.results:
        # Fetch full mental model data from DB to get history, source_memory_ids, tags
        row = await conn.fetchrow(
            f"""
            SELECT id, text, proof_count, history, tags, source_memory_ids, created_at, updated_at
            FROM {fq_table("memory_units")}
            WHERE id = $1 AND bank_id = $2 AND fact_type = 'mental_model'
            """,
            uuid.UUID(mm.id),
            bank_id,
        )

        if row:
            history = row["history"]
            if isinstance(history, str):
                history = json.loads(history)
            elif history is None:
                history = []

            results.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "proof_count": row["proof_count"] or 1,
                    "history": history,
                    "tags": row["tags"] or [],  # Include tags for LLM tag routing
                    "source_memory_ids": row["source_memory_ids"] or [],
                    "similarity": 1.0,  # Retrieved via recall so assumed relevant
                }
            )

    return results


async def _consolidate_with_llm(
    memory_engine: "MemoryEngine",
    fact_text: str,
    fact_tags: list[str],
    mental_models: list[dict[str, Any]],
    mission: str,
) -> list[dict[str, Any]]:
    """
    Single LLM call to extract durable knowledge and decide on consolidation actions.

    This handles ALL cases:
    - No related mental models: extracts durable knowledge, returns create action
    - Related models exist: compares and returns update/create actions
    - Purely ephemeral fact: returns empty array

    Returns:
        List of actions, each being:
        - {"action": "update", "learning_id": "uuid", "text": "...", "reason": "..."}
        - {"action": "create", "tags": [...], "text": "...", "reason": "..."}
        - [] if fact is purely ephemeral (no durable knowledge)
    """
    # Format mental models WITH their tags (or "None" if empty)
    if mental_models:
        mental_models_text = "\n".join(
            f'- ID: {mm["id"]}, Tags: {json.dumps(mm["tags"])}, Text: "{mm["text"]}" (proof_count: {mm["proof_count"]})'
            for mm in mental_models
        )
    else:
        mental_models_text = "None (this is a new topic - create if fact contains durable knowledge)"

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
        fact_tags=json.dumps(fact_tags),
        mental_models_text=mental_models_text,
    )

    messages = [
        {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        result = await memory_engine._llm_config.call(
            messages=messages,
            skip_validation=True,  # Raw JSON response
            scope="consolidation",
        )
        # Parse JSON response - should be an array
        if isinstance(result, str):
            result = json.loads(result)
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


async def _create_mental_model_directly(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_id: uuid.UUID,
    mental_model_text: str,
    tags: list[str] | None = None,
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> dict[str, Any]:
    """
    Create a mental model directly with pre-processed text (no LLM call).

    Used when the classify LLM has already provided the learning text.
    This avoids the redundant second LLM call.
    """
    # Generate embedding for the mental model (convert to string for pgvector)
    t0 = time.time()
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [mental_model_text])
    embedding_str = str(embeddings[0]) if embeddings else None
    if perf:
        perf.record_timing("embedding", time.time() - t0)

    # Create the mental model as a memory_unit
    now = datetime.now(timezone.utc)
    mm_event_date = event_date or now
    mm_occurred_start = occurred_start or now
    mm_tags = tags or []

    t0 = time.time()
    mental_model_id = uuid.uuid4()
    row = await conn.fetchrow(
        f"""
        INSERT INTO {fq_table("memory_units")} (
            id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids, history,
            tags, event_date, occurred_start
        )
        VALUES ($1, $2, $3, 'mental_model', $4::vector, 1, $5, '[]'::jsonb, $6, $7, $8)
        RETURNING id
        """,
        mental_model_id,
        bank_id,
        mental_model_text,
        embedding_str,
        [source_memory_id],
        mm_tags,
        mm_event_date,
        mm_occurred_start,
    )

    # Create links between memory and mental model (includes entity links, memory_links)
    await _create_memory_links(conn, source_memory_id, mental_model_id)
    if perf:
        perf.record_timing("db_write", time.time() - t0)

    logger.debug(f"Created mental model {mental_model_id} from memory {source_memory_id} (tags: {mm_tags})")

    return {"action": "created", "mental_model_id": str(row["id"]), "tags": mm_tags}
