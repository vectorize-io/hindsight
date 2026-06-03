"""Unit tests for the temporal recall circuit breaker.

When ``retrieve_temporal_combined`` is called against a bank whose temporal
filter would match an enormous number of rows, Phase 1's sort spills to disk
and the top-50-per-fact_type sample stops being statistically meaningful. The
circuit breaker uses the postgres planner's row estimate (``EXPLAIN FORMAT
JSON``) as a pre-flight cost gate: above ``TEMPORAL_RECALL_MAX_ESTIMATED_ROWS``
the function returns an empty result and lets semantic+BM25 carry the recall.

These tests pin three behaviours so future refactors can't silently regress
them: (a) the gate fires when the estimate exceeds the threshold, (b) the gate
allows the query through when the estimate is under the threshold, and (c)
EXPLAIN failures fail open rather than silently dropping legitimate queries.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from hindsight_api.engine.search.retrieval import (
    TEMPORAL_RECALL_MAX_ESTIMATED_ROWS,
    retrieve_temporal_combined,
)


class _FakeRecord(dict):
    """Mimic asyncpg.Record's tuple-style positional access."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class _FakeConn:
    """Capture EXPLAIN calls and stage a canned plan for the estimator.

    The retrieval code path issues two fetches: one EXPLAIN for the row
    estimate, then (if the gate allows) the real two-CTE query. The fake conn
    routes by query content so tests don't have to fake the full row shape of
    the real query when the gate is expected to fire.
    """

    backend_type = "postgresql"

    def __init__(self, *, planner_row_estimate: int | None):
        self._planner_row_estimate = planner_row_estimate
        self.fetch_calls: list[tuple[str, tuple]] = []

    async def fetch(self, query, *params):
        self.fetch_calls.append((query, params))
        if "EXPLAIN" in query:
            if self._planner_row_estimate is None:
                # Simulate EXPLAIN failure path — return no rows.
                return []
            plan = [
                {
                    "Plan": {
                        "Node Type": "Seq Scan",
                        "Plan Rows": self._planner_row_estimate,
                        "Total Cost": 1000.0,
                    }
                }
            ]
            return [_FakeRecord({"QUERY PLAN": json.dumps(plan)})]
        # Real query path — gate let it through. Return no rows; the caller
        # already handles empty.
        return []


@pytest.mark.asyncio
async def test_gate_fires_above_threshold():
    """When planner estimates > MAX rows, the temporal query is skipped and
    the function returns an empty result for every requested fact_type."""
    conn = _FakeConn(planner_row_estimate=TEMPORAL_RECALL_MAX_ESTIMATED_ROWS + 1)
    result = await retrieve_temporal_combined(
        conn=conn,
        query_emb_str="[0.1,0.2,0.3]",
        bank_id="bank-1",
        fact_types=["observation", "experience", "world"],
        start_date=datetime(2026, 5, 28, tzinfo=UTC),
        end_date=datetime(2026, 5, 28, 23, 59, 59, tzinfo=UTC),
        budget=10,
    )
    assert result == {"observation": [], "experience": [], "world": []}
    # Only EXPLAIN should have run — no real query.
    assert len(conn.fetch_calls) == 1
    assert "EXPLAIN" in conn.fetch_calls[0][0]


@pytest.mark.asyncio
async def test_gate_allows_below_threshold():
    """When planner estimate is at-or-below the limit, the temporal query
    proceeds normally. Each fact_type still gets a (possibly empty) list."""
    conn = _FakeConn(planner_row_estimate=TEMPORAL_RECALL_MAX_ESTIMATED_ROWS - 1)
    result = await retrieve_temporal_combined(
        conn=conn,
        query_emb_str="[0.1,0.2,0.3]",
        bank_id="bank-1",
        fact_types=["observation"],
        start_date=datetime(2026, 6, 1, tzinfo=UTC),
        end_date=datetime(2026, 6, 1, 23, 59, 59, tzinfo=UTC),
        budget=10,
    )
    # The gate let it through — both EXPLAIN and real query ran.
    assert len(conn.fetch_calls) == 2
    assert "EXPLAIN" in conn.fetch_calls[0][0]
    assert "EXPLAIN" not in conn.fetch_calls[1][0]
    # Empty rows from the fake; mapping has the requested fact_type key.
    assert set(result.keys()) == {"observation"}


@pytest.mark.asyncio
async def test_gate_fails_open_when_explain_returns_no_rows():
    """If EXPLAIN parsing fails (planner statistics unavailable, dialect
    quirk, permission error), the gate must NOT silently drop the query —
    fall through and let the real query run. Better a slow query than a
    silently-empty recall."""
    conn = _FakeConn(planner_row_estimate=None)
    result = await retrieve_temporal_combined(
        conn=conn,
        query_emb_str="[0.1,0.2,0.3]",
        bank_id="bank-1",
        fact_types=["observation"],
        start_date=datetime(2026, 6, 1, tzinfo=UTC),
        end_date=datetime(2026, 6, 1, 23, 59, 59, tzinfo=UTC),
        budget=10,
    )
    # EXPLAIN ran (returned []), then the real query ran.
    assert len(conn.fetch_calls) == 2
    assert "EXPLAIN" in conn.fetch_calls[0][0]
    assert "EXPLAIN" not in conn.fetch_calls[1][0]
    assert set(result.keys()) == {"observation"}


@pytest.mark.asyncio
async def test_gate_at_exact_threshold_allows():
    """``> threshold`` is the firing condition (strict inequality), so a
    query estimated exactly at the limit must pass — this pins the boundary
    in case a future refactor switches to ``>=``."""
    conn = _FakeConn(planner_row_estimate=TEMPORAL_RECALL_MAX_ESTIMATED_ROWS)
    result = await retrieve_temporal_combined(
        conn=conn,
        query_emb_str="[0.1,0.2,0.3]",
        bank_id="bank-1",
        fact_types=["observation"],
        start_date=datetime(2026, 6, 1, tzinfo=UTC),
        end_date=datetime(2026, 6, 1, 23, 59, 59, tzinfo=UTC),
        budget=10,
    )
    assert len(conn.fetch_calls) == 2  # gate allowed → real query ran
    assert set(result.keys()) == {"observation"}


@pytest.mark.asyncio
async def test_gate_skipped_on_non_postgres_backend():
    """The EXPLAIN format used here is postgres-specific; the gate opts out
    on other backends and lets the temporal query proceed (Oracle's planner
    surface is different and would need its own implementation)."""
    conn = _FakeConn(planner_row_estimate=10**9)
    conn.backend_type = "oracle"
    result = await retrieve_temporal_combined(
        conn=conn,
        query_emb_str="[0.1,0.2,0.3]",
        bank_id="bank-1",
        fact_types=["observation"],
        start_date=datetime(2026, 6, 1, tzinfo=UTC),
        end_date=datetime(2026, 6, 1, 23, 59, 59, tzinfo=UTC),
        budget=10,
    )
    # No EXPLAIN call — gate was skipped. Only the real query ran.
    assert len(conn.fetch_calls) == 1
    assert "EXPLAIN" not in conn.fetch_calls[0][0]
    assert set(result.keys()) == {"observation"}
