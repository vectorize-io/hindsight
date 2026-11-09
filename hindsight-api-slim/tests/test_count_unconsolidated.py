"""``count_unconsolidated``: the cheap count that replaced a 100k-row fetch.

The consolidation job's "is there work?" gate and progress denominator ask the
store for a *count*, not the rows. This pins the two properties that matter: the
count matches what ``find_unconsolidated`` would return (deduped across scopes),
and it is bounded by ``limit`` so a huge backlog stays one index count.
"""

import uuid

import pytest
from asyncpg import Record
from asyncpg.pool import PoolConnectionProxy

from hindsight_api import RequestContext
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.schema import fq_table

# This module seeds `memory_units` rows directly to control the consolidated/unconsolidated split. A MEMORIES extension owns those rows in its own store and leaves the Postgres
# tables empty, so the seed lands nowhere the code under test can see it and every assertion here
# measures the storage layout rather than the behaviour. Deselected when the suite runs against an
# alternative store; unchanged, and still required, on Postgres.
pytestmark = pytest.mark.memory_backend_incompatible


async def _seed(conn, bank_id: str, *, tags: list[str], consolidated: bool = False) -> str:
    mem_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, tags, event_date, created_at, updated_at, consolidated_at)
        VALUES ($1, $2, 'a fact', 'experience', $3, NOW(), NOW(), NOW(), CASE WHEN $4 THEN NOW() ELSE NULL END)
        """,
        mem_id,
        bank_id,
        tags,
        consolidated,
    )
    return str(mem_id)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_count_dedupes_across_overlapping_scopes(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test-count-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

    store = get_memories()
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await _seed(conn, bank_id, tags=["x"])  # matches scope [x] only
        await _seed(conn, bank_id, tags=["y"])  # matches scope [y] only
        await _seed(conn, bank_id, tags=["x", "y"])  # matches BOTH scopes — must count once
        await _seed(conn, bank_id, tags=["x"], consolidated=True)  # already consolidated — excluded

        # Unscoped: every unconsolidated fact, regardless of tags.
        assert (
            await store.count_unconsolidated(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                fact_types=["experience", "world"],
                scopes=[None],
                limit=1000,
            )
            == 3
        )

        # Two overlapping scopes: the {x,y} fact matches both but is counted once (distinct rows).
        deduped = await store.count_unconsolidated(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            fact_types=["experience", "world"],
            scopes=[["x"], ["y"]],
            limit=1000,
        )
        assert deduped == 3, "overlapping scopes must not double-count the {x,y} fact"

        # And it agrees with actually fetching + deduping the rows.
        by_id: set[str] = set()
        for scope in (["x"], ["y"]):
            for m in await store.find_unconsolidated(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                fact_types=["experience", "world"],
                limit=1000,
                scope_tags=scope,
            ):
                by_id.add(m.unit_id)
        assert deduped == len(by_id)

        # Bounded by limit: never reports more than asked, however big the backlog.
        assert (
            await store.count_unconsolidated(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                fact_types=["experience", "world"],
                scopes=[None],
                limit=2,
            )
            == 2
        )

    await memory.delete_bank(bank_id, request_context=request_context)


class _RecordingConn:
    """Forwards ``fetchrow`` to a real connection and keeps the SQL it was sent."""

    def __init__(self, conn: PoolConnectionProxy) -> None:
        self._conn = conn
        self.sql = ""

    async def fetchrow(self, sql: str, *params: object) -> Record | None:
        self.sql = sql
        return await self._conn.fetchrow(sql, *params)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_count_orders_by_the_unconsolidated_index_key(memory: MemoryEngine):
    """The bounded count orders its inner rows by ``created_at`` (#6415).

    The order does not change the count. It is what lets Postgres walk
    ``idx_memory_units_unconsolidated (bank_id, created_at)``: without it the planner chose a seq
    scan of the whole ``memory_units`` table on every consolidation job, 756 ms instead of 2.5 ms on
    a 354k-row bank with nothing pending. That plan only shows on a large table with stale stats,
    which this suite cannot build, so the test pins the query shape that earns the index.
    """
    # Only the SQL is inspected, so no bank or rows are needed: the query shape does not depend on data.
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        all_scopes: list[list[list[str] | None]] = [[None], [["x"], ["y"]]]
        for scopes in all_scopes:
            recording = _RecordingConn(conn)
            await get_memories().count_unconsolidated(
                conn=recording,
                fq_table=fq_table,
                bank_id=f"test-count-{uuid.uuid4().hex[:8]}",
                fact_types=["experience", "world"],
                scopes=scopes,
                limit=1000,
            )
            inner = " ".join(recording.sql.split())
            assert "ORDER BY created_at LIMIT" in inner, inner
