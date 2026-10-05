"""Temporal spreading keeps the strongest link when several reach the same memory.

The spreading query fetches the top-K links per source ordered by stored weight, but the
outer query has no ORDER BY — so row order is planner-dependent. Scoring a target from
whichever row arrived first made its score, and whether the walk continued from it, depend
on the query plan. `retrieve_temporal_combined_sql` now picks the strongest path per target
before scoring.

Pure mechanics (no LLM), so these assert directly.
"""

import uuid
from datetime import UTC, datetime

import pytest

from hindsight_api.engine.memories.pg.recall import retrieve_temporal_combined_sql
from hindsight_api.engine.schema import fq_store_table_explicit as fq_table

EMBED_DIM = 384

# Query vector, and a vector at cosine 1.0 to it — every unit here is maximally similar,
# so similarity gating never decides the outcome.
_QUERY = "[" + ",".join(["1.0"] + ["0.0"] * (EMBED_DIM - 1)) + "]"
_SIM_100 = _QUERY


async def _insert_unit(conn, bank_id: str, text: str, when: datetime) -> uuid.UUID:
    row = await conn.fetchrow(
        f"""
        INSERT INTO {fq_table("memory_units")} (bank_id, text, fact_type, embedding, event_date, mentioned_at)
        VALUES ($1, $2, 'world', $3::vector, $4, $4)
        RETURNING id
        """,
        bank_id,
        text,
        _SIM_100,
        when,
    )
    return row["id"]


async def _insert_link(conn, bank_id: str, src: uuid.UUID, dst: uuid.UUID, link_type: str, weight: float) -> None:
    await conn.execute(
        f"""
        INSERT INTO {fq_table("memory_links")} (from_unit_id, to_unit_id, link_type, weight, bank_id)
        VALUES ($1, $2, $3, $4, $5)
        """,
        src,
        dst,
        link_type,
        weight,
        bank_id,
    )


@pytest.mark.asyncio
async def test_strongest_relation_wins_over_heavier_weight(memory):
    """Two links reach the same target. The `caused_by` edge scores higher despite a
    *lower* stored weight, because its relation multiplier (2.0) outweighs the gap —
    and the heavier `temporal` edge is the one the inner ORDER BY returns first."""
    bank_id = "test_temporal_best_path"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 2, 1, tzinfo=UTC)

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

        # In-window entry point, at the window's midpoint → temporal score 1.0.
        source = await _insert_unit(conn, bank_id, "entry point", datetime(2025, 1, 16, 12, tzinfo=UTC))
        # Target far outside the window → its own date proximity clamps to 0.0, so its
        # score comes entirely from the link it is reached through.
        target = await _insert_unit(conn, bank_id, "linked target", datetime(2030, 1, 1, tzinfo=UTC))

        # Heavier weight on the weaker relation: 0.9 * 1.0 * 0.7 = 0.63 ...
        await _insert_link(conn, bank_id, source, target, "temporal", 0.9)
        # ... versus 0.5 * 2.0 * 0.7 = 0.70 on the stronger one.
        await _insert_link(conn, bank_id, source, target, "caused_by", 0.5)

        results = await retrieve_temporal_combined_sql(conn, _QUERY, bank_id, ["world"], start, end, budget=100)

    reached = [r for r in results["world"] if r.id == str(target)]
    # Spreading emits the target exactly once, however many links reach it.
    assert len(reached) == 1
    assert reached[0].temporal_score == pytest.approx(0.70)


@pytest.mark.asyncio
async def test_weaker_duplicate_link_does_not_consume_budget(memory):
    """A second link to an already-scored target costs nothing: the target is emitted once
    and the extra row neither re-scores it nor eats a budget slot that a distinct memory
    would otherwise get.

    This held before the best-path change too (`visited` already skipped the duplicate row);
    it is here to guard the dedup rewrite, which could plausibly have charged budget twice.
    """
    bank_id = "test_temporal_best_path_budget"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 2, 1, tzinfo=UTC)

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

        source = await _insert_unit(conn, bank_id, "entry point", datetime(2025, 1, 16, 12, tzinfo=UTC))
        doubly_linked = await _insert_unit(conn, bank_id, "doubly linked", datetime(2030, 1, 1, tzinfo=UTC))
        singly_linked = await _insert_unit(conn, bank_id, "singly linked", datetime(2030, 1, 1, tzinfo=UTC))

        await _insert_link(conn, bank_id, source, doubly_linked, "temporal", 0.9)
        await _insert_link(conn, bank_id, source, doubly_linked, "caused_by", 0.5)
        await _insert_link(conn, bank_id, source, singly_linked, "temporal", 0.9)

        # Budget 3 = 1 entry point + 2 targets. If the duplicate row consumed a slot, one
        # of the two distinct targets would be dropped.
        results = await retrieve_temporal_combined_sql(conn, _QUERY, bank_id, ["world"], start, end, budget=3)

    ids = [r.id for r in results["world"]]
    assert sorted(ids) == sorted([str(source), str(doubly_linked), str(singly_linked)])


@pytest.mark.asyncio
async def test_budget_cutoff_drops_the_weakest_paths(memory):
    """When budget runs out mid-batch, the targets that survive are the strongest-scoring
    ones, not whichever the planner returned first."""
    bank_id = "test_temporal_best_path_cutoff"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 2, 1, tzinfo=UTC)

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

        source = await _insert_unit(conn, bank_id, "entry point", datetime(2025, 1, 16, 12, tzinfo=UTC))
        # All three targets sit far outside the window, so each one's score is purely the
        # strength of the link it is reached through.
        strong = await _insert_unit(conn, bank_id, "strong path", datetime(2030, 1, 1, tzinfo=UTC))
        medium = await _insert_unit(conn, bank_id, "medium path", datetime(2030, 1, 1, tzinfo=UTC))
        weak = await _insert_unit(conn, bank_id, "weak path", datetime(2030, 1, 1, tzinfo=UTC))

        await _insert_link(conn, bank_id, source, strong, "caused_by", 0.5)  # 0.5 * 2.0 * 0.7 = 0.70
        await _insert_link(conn, bank_id, source, medium, "temporal", 0.9)  # 0.9 * 1.0 * 0.7 = 0.63
        await _insert_link(conn, bank_id, source, weak, "temporal", 0.2)  # 0.2 * 1.0 * 0.7 = 0.14

        # Budget 3 = 1 entry point + room for 2 of the 3 targets.
        results = await retrieve_temporal_combined_sql(conn, _QUERY, bank_id, ["world"], start, end, budget=3)

    ids = {r.id for r in results["world"]}
    assert ids == {str(source), str(strong), str(medium)}
    assert str(weak) not in ids
