"""Tests for the table-backed (cross-process) get_bank_stats cache.

On PostgreSQL the engine backs `get_bank_stats` with the `bank_stats_cache`
table (`DistributedBankStatsCache`) instead of a per-process dict, so one
worker's computation is shared with every other worker. These tests verify:

* the PG engine actually selects the distributed cache,
* a computed result is written to the table and served from it on the next call,
* invalidation deletes the row so the next call recomputes, and
* an unreachable cache table degrades to computing without caching rather than
  failing the endpoint.
"""

import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.bank_stats_cache import DistributedBankStatsCache
from hindsight_api.engine.memory_engine import MemoryEngine, get_current_schema

_PINNED_TTL_SECONDS = 300.0


async def _insert_memory(conn, bank_id: str, text: str, fact_type: str = "experience") -> uuid.UUID:
    mem_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, created_at, updated_at, consolidated_at)
        VALUES ($1, $2, $3, $4, NOW(), NOW(), NOW(), NOW())
        """,
        mem_id,
        bank_id,
        text,
        fact_type,
    )
    return mem_id


async def _ensure_bank(memory: MemoryEngine, bank_id: str, request_context: RequestContext) -> None:
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)


def _pin_distributed_cache(memory: MemoryEngine) -> DistributedBankStatsCache:
    cache = DistributedBankStatsCache(backend=memory._backend, ttl_seconds=_PINNED_TTL_SECONDS)
    memory._bank_stats_cache = cache
    return cache


class TestDistributedBankStatsCache:
    @pytest.mark.asyncio
    async def test_pg_engine_selects_distributed_cache(self, memory: MemoryEngine):
        if memory._database_backend_type != "postgresql":
            pytest.skip("distributed cache is PostgreSQL-only")
        assert isinstance(memory._bank_stats_cache, DistributedBankStatsCache)

    @pytest.mark.asyncio
    async def test_result_is_written_and_served_from_table(self, memory: MemoryEngine, request_context: RequestContext):
        if memory._database_backend_type != "postgresql":
            pytest.skip("distributed cache is PostgreSQL-only")

        bank_id = f"test-dist-stats-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice loves hiking.")

        _pin_distributed_cache(memory)
        try:
            first = await memory.get_bank_stats(bank_id, request_context=request_context)
            assert first["node_counts"].get("experience") == 1

            # The computed result was persisted to the shared table.
            async with pool.acquire() as conn:
                rows = await conn.fetchval("SELECT count(*) FROM bank_stats_cache WHERE bank_id = $1", bank_id)
            assert rows == 1

            # Mutate the underlying data WITHOUT going through an invalidating
            # engine method — the long-TTL cache must serve the stale row.
            async with pool.acquire() as conn:
                await _insert_memory(conn, bank_id, "Bob enjoys cycling.")
            served = await memory.get_bank_stats(bank_id, request_context=request_context)
            assert served["node_counts"].get("experience") == 1  # still cached

            # Invalidating drops the row → next call recomputes the true count.
            await memory._bank_stats_cache.invalidate(get_current_schema(), bank_id)
            async with pool.acquire() as conn:
                rows = await conn.fetchval("SELECT count(*) FROM bank_stats_cache WHERE bank_id = $1", bank_id)
            assert rows == 0
            fresh = await memory.get_bank_stats(bank_id, request_context=request_context)
            assert fresh["node_counts"].get("experience") == 2
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_degrades_when_cache_table_unreachable(self, memory: MemoryEngine, request_context: RequestContext):
        if memory._database_backend_type != "postgresql":
            pytest.skip("distributed cache is PostgreSQL-only")

        bank_id = f"test-dist-stats-degrade-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice loves hiking.")

        # Point the cache at a table that does not exist: reads and writes fail,
        # so it must fall back to computing the real result (no real table touched).
        cache = _pin_distributed_cache(memory)
        cache._qualified = lambda schema: '"public".bank_stats_cache_does_not_exist'  # type: ignore[method-assign]
        try:
            stats = await memory.get_bank_stats(bank_id, request_context=request_context)
            assert stats["node_counts"].get("experience") == 1
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)
