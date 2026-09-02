"""Whole consolidation actions, composed from the consolidator's two halves.

Consolidation applies every write derived from one LLM response — deletes, updates,
creates, dedup folds and the ``consolidated_at`` stamps for the facts consumed — in a
single transaction (#3876). A transaction must never be held open across an embedder or
an LLM call, so production splits each action in two: a connection-free *prepare*
(resolve sources, embed, adjudicate dedup) and an *apply* that runs on the batch's
connection inside the batch's transaction.

Tests that exercise one action end to end want the pair back together. These helpers do
exactly what ``_process_memory_batch`` does for a single action, and nothing else — keep
them that way, so a test asserting on an action still asserts on production behaviour.

Connections are acquired through ``C.acquire_with_retry`` (not the ``db_utils`` symbol) so a
test that stubs the consolidator's pool acquisition also stubs these helpers'.
"""

from __future__ import annotations

import uuid
from typing import Any

from hindsight_api.engine.consolidation import consolidator as C


async def create_observation(
    *,
    pool,
    memory_engine,
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    observation_text: str,
    tags: list[str] | None = None,
    event_date=None,
    occurred_start=None,
    occurred_end=None,
    mentioned_at=None,
    perf=None,
) -> dict[str, Any]:
    """Preflight + embed off-connection, then insert in one short transaction."""
    async with C.acquire_with_retry(pool) as conn:
        if not await C._any_live_source_memory(conn, bank_id, source_memory_ids):
            return {"action": "skipped", "reason": "sources_deleted"}
    embedding_str = await C._embed_observation_text(memory_engine, observation_text, perf)
    async with C.acquire_with_retry(pool) as conn:
        async with conn.transaction():
            return await C._apply_create_observation(
                conn=conn,
                memory_engine=memory_engine,
                bank_id=bank_id,
                source_memory_ids=source_memory_ids,
                observation_text=observation_text,
                embedding_str=embedding_str,
                tags=tags,
                event_date=event_date,
                occurred_start=occurred_start,
                occurred_end=occurred_end,
                mentioned_at=mentioned_at,
                perf=perf,
            )


async def execute_create_action(
    *,
    pool,
    memory_engine,
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    text: str,
    source_fact_tags: list[str] | None = None,
    event_date=None,
    occurred_start=None,
    occurred_end=None,
    mentioned_at=None,
    perf=None,
) -> str:
    """One CREATE action: returns "created" or "skipped", as the batch executor sees it."""
    created = await create_observation(
        pool=pool,
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
    new_id = created.get("observation_id")
    if new_id:
        C.record_created_memory_ids([new_id])
    return created["action"]


async def execute_update_action(
    *,
    pool,
    memory_engine,
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    observation_id: str,
    new_text: str,
    observations: list,
    source_fact_tags: list[str] | None = None,
    source_bounds: C._TemporalBounds = C._TemporalBounds(),
    perf=None,
) -> str | None:
    """One UPDATE action: returns the observation's new embedding, or None if skipped."""
    model = next((m for m in observations if str(m.id) == observation_id), None)
    if model is None:
        return None
    async with C.acquire_with_retry(pool) as conn:
        if not await C._any_live_source_memory(conn, bank_id, source_memory_ids):
            return None
    embedding_str = await C._embed_observation_text(memory_engine, new_text, perf)
    prepared = C._PreparedUpdate(
        update=C._UpdateAction(
            text=new_text,
            observation_id=observation_id,
            source_fact_ids=[str(mid) for mid in source_memory_ids],
        ),
        model=model,
        source_mems=[],
        source_memory_ids=source_memory_ids,
        source_fact_tags=source_fact_tags or [],
        source_bounds=source_bounds,
        embedding_str=embedding_str,
    )
    async with C.acquire_with_retry(pool) as conn:
        async with conn.transaction():
            return await C._apply_update_action(
                conn=conn,
                memory_engine=memory_engine,
                bank_id=bank_id,
                prepared=prepared,
                perf=perf,
            )


async def dedup_reconcile_update(
    pool,
    memory_engine,
    bank_id: str,
    config,
    dedup_llm_config,
    updated_id: str,
    updated_text: str,
    updated_emb_str: str | None,
    tags: list[str] | None,
) -> bool:
    """UPDATE-path dedup: adjudicate off-connection, then fold in one transaction."""
    outcome = await C._dedup_adjudicate(
        pool,
        memory_engine,
        bank_id,
        config,
        dedup_llm_config,
        updated_text,
        updated_emb_str,
        tags,
        exclude_id=updated_id,
    )
    async with C.acquire_with_retry(pool) as conn:
        async with conn.transaction():
            return await C._apply_dedup_update_fold(
                conn, memory_engine, bank_id, config, outcome, updated_id, updated_text
            )
