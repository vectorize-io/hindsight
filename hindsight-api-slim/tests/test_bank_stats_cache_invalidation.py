"""Regression tests: get_bank_stats cache must be invalidated by mutations.

`get_bank_stats` is served from a short-TTL per-process cache (`BankStatsCache`).
`delete_bank` already invalidates that cache after it mutates counts, but the
other operations that change the same counts — `delete_memory_unit`,
`delete_document`, `clear_observations`, and `update_document` (when a tag
change deletes observations) — did not, so a client polling stats right after a
deletion would see pre-mutation counts until the TTL expired (up to a minute).

Each test pins a long TTL on the engine's stats cache so that, *without* the
invalidation fix, the second `get_bank_stats` call would be served the stale
cached value and the assertion would fail.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.bank_stats_cache import BankStatsCache
from hindsight_api.engine.memories import FactRecord, get_memories
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table

# A TTL long enough that, absent invalidation, the warmed cache would still be
# served on the post-mutation read within the same test.
_PINNED_TTL_SECONDS = 300.0


async def _insert_memory(
    memory: MemoryEngine,
    conn,
    bank_id: str,
    text: str,
    fact_type: str = "experience",
    document_id: str | None = None,
) -> uuid.UUID:
    """Seed one memory through the store, bypassing the LLM retain pipeline.

    Goes through ``insert_facts`` rather than an ``INSERT INTO memory_units`` so the fixture
    seeds wherever memories actually live: a store that keeps them outside SQL never sees a
    raw row, which would leave the bank empty and every assertion below reading zero.
    """
    store = get_memories()
    fact = SimpleNamespace(
        fact_text=text,
        embedding=memory.embeddings.encode([text])[0],
        fact_type=fact_type,
        tags=[],
        context=None,
        document_id=document_id,
        chunk_id=None,
        metadata=None,
        observation_scopes=None,
        entities=[],
        causal_relations=[],
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
    )
    unit_ids = await store.insert_facts(
        conn=conn, ops=memory._backend.ops, bank_id=bank_id, facts=[fact], document_id=document_id
    )
    # The raw insert this replaces stamped consolidated_at, so these fixtures are not a
    # consolidation backlog. Keep that: an unconsolidated source would change the stats read below.
    await store.mark_consolidated(
        conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids, when=datetime.now(timezone.utc)
    )
    return uuid.UUID(unit_ids[0])


async def _insert_observation(
    memory: MemoryEngine, conn, bank_id: str, text: str, source_memory_ids: list[uuid.UUID]
) -> uuid.UUID:
    """Seed one observation, wherever the configured store keeps observations.

    Gated because the two stores split this differently: the SQL store's ``upsert_observation``
    is a no-op (the consolidator writes that row inline), so SQL is seeded with the insert it
    would have written; a store that owns its observations is seeded through the store.
    """
    store = get_memories()
    obs_id = uuid.uuid4()
    if store.writes_memory_rows_in_sql:
        await conn.execute(
            """
            INSERT INTO memory_units (
                id, bank_id, text, fact_type, event_date, source_memory_ids, proof_count, created_at, updated_at
            ) VALUES ($1, $2, $3, 'observation', NOW(), $4, $5, NOW(), NOW())
            """,
            obs_id,
            bank_id,
            text,
            source_memory_ids,
            len(source_memory_ids),
        )
    else:
        now = datetime.now(timezone.utc)
        await store.upsert_observation(
            conn=conn,
            bank_id=bank_id,
            record=FactRecord(
                unit_id=str(obs_id),
                text=text,
                embedding=memory.embeddings.encode([text])[0],
                fact_type="observation",
                proof_count=len(source_memory_ids),
                source_memory_ids=[str(s) for s in source_memory_ids],
                event_date=now,
                created_at=now,
            ),
        )
    return obs_id


async def _insert_document(conn, bank_id: str, doc_id: str) -> None:
    await conn.execute(
        """
        INSERT INTO documents (id, bank_id, original_text, content_hash)
        VALUES ($1, $2, $3, $4)
        """,
        doc_id,
        bank_id,
        f"text-for-{doc_id}",
        doc_id,
    )


async def _ensure_bank(memory: MemoryEngine, bank_id: str, request_context: RequestContext) -> None:
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)


def _pin_cache(memory: MemoryEngine) -> None:
    """Replace the stats cache with one that has a deterministic long TTL."""
    memory._bank_stats_cache = BankStatsCache(ttl_seconds=_PINNED_TTL_SECONDS, max_entries=128)


class TestBankStatsCacheInvalidation:
    @pytest.mark.asyncio
    async def test_delete_memory_unit_invalidates_stats_cache(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        bank_id = f"test-stats-cache-delunit-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(memory, conn, bank_id, "Alice loves hiking.")
            await _insert_memory(memory, conn, bank_id, "Bob enjoys cycling.")

        _pin_cache(memory)
        try:
            before = await memory.get_bank_stats(bank_id, request_context=request_context)
            assert before["node_counts"].get("experience") == 2

            await memory.delete_memory_unit(str(m1), bank_id=bank_id, request_context=request_context)

            after = await memory.get_bank_stats(bank_id, request_context=request_context)
            # Without invalidation the long-TTL cache would still report 2.
            assert after["node_counts"].get("experience") == 1
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_delete_document_invalidates_stats_cache(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-stats-cache-deldoc-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        document_id = f"doc-{uuid.uuid4().hex[:8]}"
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await _insert_document(conn, bank_id, document_id)
            await _insert_memory(memory, conn, bank_id, "Alice works at Acme.", document_id=document_id)

        _pin_cache(memory)
        try:
            before = await memory.get_bank_stats(bank_id, request_context=request_context)
            assert before["total_documents"] == 1
            assert before["node_counts"].get("experience") == 1

            await memory.delete_document(document_id, bank_id, request_context=request_context)

            after = await memory.get_bank_stats(bank_id, request_context=request_context)
            # Without invalidation the long-TTL cache would still report 1 document.
            assert after["total_documents"] == 0
            assert after["node_counts"].get("experience", 0) == 0
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_clear_observations_invalidates_stats_cache(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        bank_id = f"test-stats-cache-clearobs-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(memory, conn, bank_id, "Alice loves hiking.")
            await _insert_observation(memory, conn, bank_id, "Alice enjoys hiking regularly.", [m1])

        _pin_cache(memory)
        try:
            before = await memory.get_bank_stats(bank_id, request_context=request_context)
            assert before["total_observations"] == 1

            await memory.clear_observations(bank_id, request_context=request_context)

            after = await memory.get_bank_stats(bank_id, request_context=request_context)
            # Without invalidation the long-TTL cache would still report 1 observation.
            assert after["total_observations"] == 0
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)
