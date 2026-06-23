"""Regression test: `enqueue_graph_maintenance` must insert unit_ids in a
deterministic sorted order so concurrent transactions can't deadlock on the
graph_maintenance_queue unique-key check.

Symptom (production): under load, concurrent `PATCH /memories/{id}` requests
on the same bank generate overlapping `victim_ids` sets (the surviving units
whose outgoing links pointed at the updated unit). Each transaction inserts
those victims into `graph_maintenance_queue` with
`ON CONFLICT (bank_id, unit_id) DO NOTHING`. The conflict check takes a
short-lived row-level lock per (bank_id, unit_id) being inserted, and when
two transactions insert overlapping sets in different orders Postgres
detects a deadlock and aborts one of them — surfacing as
`asyncpg.exceptions.DeadlockDetectedError` from the API, which becomes a 500.

The fix sorts the input list inside both `ops_postgresql` and `ops_oracle`
before passing it to the INSERT, so every transaction acquires the per-row
locks in the same global (sorted-UUID) order. With a total order over the
lock set, deadlock is mathematically impossible — Postgres still serializes
the conflicting inserts but they queue cleanly instead of cycling.

This test pins that post-condition by capturing the array passed to the
underlying `conn.execute` (PG path) / `conn.executemany` (Oracle path) and
asserting it's sorted.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps


def _shuffled_uuids(n: int) -> list[uuid.UUID]:
    """Generate n UUIDs in a deliberately non-monotonic order. Hex literals
    avoid `uuid.uuid4()` because uuid4 is random and we want determinism."""
    raw = [
        "ffffffff-ffff-4fff-8fff-ffffffffffff",
        "00000000-0000-4000-8000-000000000001",
        "88888888-8888-4888-8888-888888888888",
        "11111111-1111-4111-8111-111111111111",
        "ccccccccc-cccc-4ccc-8ccc-cccccccccccc"[:36],
        "44444444-4444-4444-8444-444444444444",
    ]
    return [uuid.UUID(s) for s in raw[:n]]


@pytest.mark.asyncio
async def test_pg_enqueue_graph_maintenance_inserts_in_sorted_order():
    """The PostgreSQL ops impl must pass the unit_ids to the INSERT in
    sorted order, regardless of how the caller ordered them."""
    ops = PostgreSQLOps()
    conn = AsyncMock()

    unit_ids = _shuffled_uuids(6)
    assert unit_ids != sorted(unit_ids), "test inputs must be unsorted"

    await ops.enqueue_graph_maintenance(
        conn=conn,
        table="graph_maintenance_queue",
        bank_id="test-bank",
        unit_ids=unit_ids,
    )

    assert conn.execute.await_count == 1
    _sql, bank_id_arg, ids_arg = conn.execute.await_args.args
    assert bank_id_arg == "test-bank"
    assert ids_arg == sorted(unit_ids), f"expected sorted unit_ids for deadlock-free concurrent inserts, got {ids_arg}"


@pytest.mark.asyncio
async def test_oracle_enqueue_graph_maintenance_inserts_in_sorted_order():
    """The Oracle ops impl applies the same sort. `executemany` receives a
    list of (bank_id, unit_id) tuples; the unit_id projection must be
    sorted."""
    ops = OracleOps()
    conn = AsyncMock()

    unit_ids = _shuffled_uuids(6)
    assert unit_ids != sorted(unit_ids), "test inputs must be unsorted"

    await ops.enqueue_graph_maintenance(
        conn=conn,
        table="graph_maintenance_queue",
        bank_id="test-bank",
        unit_ids=unit_ids,
    )

    assert conn.executemany.await_count == 1
    _sql, rows = conn.executemany.await_args.args
    assert [r[0] for r in rows] == ["test-bank"] * len(unit_ids)
    assert [r[1] for r in rows] == sorted(unit_ids), (
        f"expected sorted unit_ids for deadlock-free concurrent inserts, got {[r[1] for r in rows]}"
    )


@pytest.mark.asyncio
async def test_pg_empty_unit_ids_short_circuits():
    """Empty input must remain a no-op — the early return predates this fix
    and must continue to skip the INSERT entirely."""
    ops = PostgreSQLOps()
    conn = AsyncMock()
    await ops.enqueue_graph_maintenance(conn, "graph_maintenance_queue", "b", [])
    conn.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_oracle_empty_unit_ids_short_circuits():
    ops = OracleOps()
    conn = AsyncMock()
    await ops.enqueue_graph_maintenance(conn, "graph_maintenance_queue", "b", [])
    conn.executemany.assert_not_awaited()
