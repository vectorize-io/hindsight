"""Tests for the temporal-recall entry-point selection (Option A: similarity-gated +
window coverage).

`retrieve_temporal_combined` selects in-window entry points by *embedding similarity*
(not recency) and then narrows them to span the window's time range via
`_select_with_temporal_coverage`. This replaced an earlier recency-ranked selection that
biased toward the end of the window and, on banks with dense/near-uniform dates, degraded
to a full scan + disk-spilling sort while dropping the most relevant in-window memory.

These are pure mechanics (no LLM), so they assert directly.
"""

from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api.engine.search.retrieval import _select_with_temporal_coverage, retrieve_temporal_combined
from hindsight_api.engine.task_backend import fq_table

EMBED_DIM = 384


def _vec(*leading: float) -> str:
    values = list(leading) + [0.0] * (EMBED_DIM - len(leading))
    return "[" + ",".join(str(v) for v in values) + "]"


# Query vector + vectors at known cosine similarities to it.
_QUERY = _vec(1.0)
_SIM_100 = _vec(1.0)  # cosine 1.0
_SIM_090 = _vec(0.9, 0.4358898943540674)  # cosine 0.9
_SIM_050 = _vec(0.5, 0.8660254037844386)  # cosine 0.5


def _row(sim: float, day: datetime) -> dict:
    """A minimal pool row for the pure selector test."""
    return {
        "id": f"{sim}-{day.isoformat()}",
        "similarity": sim,
        "occurred_start": None,
        "mentioned_at": day,
        "occurred_end": None,
    }


# ---------------------------------------------------------------------------
# Pure selector: coverage round-robin
# ---------------------------------------------------------------------------


def test_coverage_round_robin_spreads_across_buckets():
    """One item per populated bucket is taken before any bucket gets a second."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 12, 31, tzinfo=UTC)
    jan, jul = datetime(2025, 1, 15, tzinfo=UTC), datetime(2025, 7, 15, tzinfo=UTC)
    # Two time-buckets; January is denser and slightly more similar.
    pool = [_row(1.0, jan), _row(0.99, jan), _row(0.98, jan), _row(0.95, jul)]

    selected = _select_with_temporal_coverage(pool, start, end, limit=2, n_buckets=8)

    # Coverage: the single July item beats the 2nd/3rd January items despite lower similarity.
    months = {r["mentioned_at"].month for r in selected}
    assert months == {1, 7}


def test_coverage_degenerate_dates_fall_back_to_similarity():
    """When all dates land in one bucket, selection is plain top-by-similarity."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 12, 31, tzinfo=UTC)
    same_day = datetime(2025, 1, 15, tzinfo=UTC)
    pool = [_row(0.4, same_day), _row(0.9, same_day), _row(0.7, same_day), _row(0.95, same_day)]

    selected = _select_with_temporal_coverage(pool, start, end, limit=2, n_buckets=8)

    assert [r["similarity"] for r in selected] == [0.95, 0.9]


# ---------------------------------------------------------------------------
# DB-backed: similarity gating + window filter + coverage
# ---------------------------------------------------------------------------


async def _insert_unit(conn, bank_id: str, text: str, fact_type: str, when: datetime, embedding: str) -> str:
    table = fq_table("memory_units")
    row = await conn.fetchrow(
        f"""
        INSERT INTO {table} (bank_id, text, fact_type, embedding, event_date, mentioned_at)
        VALUES ($1, $2, $3, $4::vector, $5, $5)
        RETURNING id
        """,
        bank_id,
        text,
        fact_type,
        embedding,
        when,
    )
    return str(row["id"])


@pytest.mark.asyncio
async def test_temporal_recall_selects_by_similarity_not_recency(memory):
    """The most-similar in-window unit is returned even when it is the OLDEST — the inverse
    of the old recency-ranked behavior, where it would have been dropped."""
    bank_id = "test_temporal_similarity_gate"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 2, 1, tzinfo=UTC)

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

        # Oldest in-window unit, perfect similarity.
        oldest_relevant = await _insert_unit(
            conn, bank_id, "oldest relevant", "world", datetime(2025, 1, 2, tzinfo=UTC), _SIM_100
        )
        # Newer, less-similar units.
        for i in range(8):
            await _insert_unit(
                conn,
                bank_id,
                f"recent less-relevant {i}",
                "world",
                datetime(2025, 1, 20, tzinfo=UTC) + timedelta(hours=i),
                _SIM_050,
            )
        # Out-of-window, perfect similarity → must be excluded by the window.
        before = await _insert_unit(conn, bank_id, "before", "world", datetime(2024, 12, 1, tzinfo=UTC), _SIM_100)
        after = await _insert_unit(conn, bank_id, "after", "world", datetime(2025, 3, 1, tzinfo=UTC), _SIM_100)

        results = await retrieve_temporal_combined(conn, _QUERY, bank_id, ["world"], start, end, budget=100)

    by_id = {r.id: r for r in results.get("world", [])}
    # Similarity wins over recency: the oldest, most-relevant unit is selected.
    assert oldest_relevant in by_id
    # Window filter still excludes out-of-window units, even at perfect similarity.
    assert before not in by_id
    assert after not in by_id


@pytest.mark.asyncio
async def test_temporal_recall_covers_window_range(memory):
    """Entry points span the window: relevant units in distinct time-slices are all
    represented, instead of clustering in the densest slice."""
    bank_id = "test_temporal_coverage"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 12, 31, tzinfo=UTC)

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

        # A dense January cluster at the highest similarity.
        for i in range(20):
            await _insert_unit(
                conn, bank_id, f"jan {i}", "world", datetime(2025, 1, 10, tzinfo=UTC) + timedelta(hours=i), _SIM_100
            )
        # One slightly-less-similar unit in three other quarters.
        apr = await _insert_unit(conn, bank_id, "apr", "world", datetime(2025, 4, 15, tzinfo=UTC), _SIM_090)
        jul = await _insert_unit(conn, bank_id, "jul", "world", datetime(2025, 7, 15, tzinfo=UTC), _SIM_090)
        octo = await _insert_unit(conn, bank_id, "oct", "world", datetime(2025, 10, 15, tzinfo=UTC), _SIM_090)

        results = await retrieve_temporal_combined(conn, _QUERY, bank_id, ["world"], start, end, budget=100)

    ids = {r.id for r in results.get("world", [])}
    # Without coverage, the top-10 by similarity would be 10 January units and the Apr/Jul/Oct
    # units (lower similarity) would be crowded out. Coverage surfaces every populated slice.
    assert {apr, jul, octo} <= ids
    selected_months = {r.mentioned_at.month for r in results["world"] if r.mentioned_at}
    assert len(selected_months) >= 3
