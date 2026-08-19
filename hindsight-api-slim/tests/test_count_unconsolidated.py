"""``count_unconsolidated``: the cheap count that replaced a 100k-row fetch.

The consolidation job's "is there work?" gate and progress denominator ask the
store for a *count*, not the rows. This pins the two properties that matter: the
count matches what ``find_unconsolidated`` would return (deduped across scopes),
and it is bounded by ``limit`` so a huge backlog stays one index count.
"""

import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.schema import fq_table


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
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

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
