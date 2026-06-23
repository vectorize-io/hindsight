"""Reproduces the concurrent-insert deadlock on ``graph_maintenance_queue``
that PR #2353 targets, and demonstrates that a shared insertion order cures it.

A deadlock is a *database*-level phenomenon, so unlike the PR's own tests (which
only assert that the Python list handed to ``conn.execute`` is sorted) these run
against the real Postgres test DB and drive two genuinely-concurrent
transactions, forcing the exact interleaving that produces a lock cycle.

Modelling note
--------------
Production enqueues a whole victim set in ONE statement::

    INSERT INTO graph_maintenance_queue (bank_id, unit_id)
    SELECT $1, v FROM unnest($2::uuid[]) ON CONFLICT (bank_id, unit_id) DO NOTHING

That single statement still takes the per-row unique-key locks one row at a time,
in the order ``unnest`` yields — we just can't pause *inside* a single statement.
So each worker here issues the rows one at a time with a barrier between them.
That makes the otherwise-racy interleaving deterministic while exercising the
identical lock: ``ON CONFLICT`` on the ``(bank_id, unit_id)`` primary key.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from asyncpg.exceptions import DeadlockDetectedError

from hindsight_api.engine.memory_engine import MemoryEngine

# Two keys with an unambiguous sort order (low < high as UUIDs / as text).
K_LOW = uuid.UUID("00000000-0000-4000-8000-000000000001")
K_HIGH = uuid.UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")


async def _insert_one(conn, bank_id: str, unit_id: uuid.UUID) -> None:
    """One row of the production INSERT ... ON CONFLICT DO NOTHING."""
    await conn.execute(
        """
        INSERT INTO graph_maintenance_queue (bank_id, unit_id)
        VALUES ($1, $2)
        ON CONFLICT (bank_id, unit_id) DO NOTHING
        """,
        bank_id,
        unit_id,
    )


@pytest.mark.asyncio
async def test_unordered_concurrent_enqueue_deadlocks(memory: MemoryEngine):
    """Two transactions inserting the same two keys in OPPOSITE orders deadlock.

    This is the pre-fix reality: ``enqueue_relink_victims`` feeds whatever order
    ``SELECT DISTINCT`` returns, so two overlapping victim sets can acquire the
    unique-key locks in opposite orders and cycle. Postgres aborts one with
    ``DeadlockDetectedError``, which the API surfaces as a 500.
    """
    pool = await memory._get_pool()
    bank_id = f"dl-bug-{uuid.uuid4().hex[:8]}"

    # Both transactions hold their first lock before either takes its second,
    # so the cross-wait (and thus the cycle) is guaranteed rather than racy.
    barrier = asyncio.Barrier(2)

    async def worker(order: list[uuid.UUID]) -> None:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await _insert_one(conn, bank_id, order[0])
                await barrier.wait()
                await _insert_one(conn, bank_id, order[1])

    results = await asyncio.wait_for(
        asyncio.gather(
            worker([K_LOW, K_HIGH]),
            worker([K_HIGH, K_LOW]),
            return_exceptions=True,
        ),
        timeout=30,
    )

    deadlocks = [r for r in results if isinstance(r, DeadlockDetectedError)]
    assert deadlocks, f"expected one transaction aborted with DeadlockDetectedError, got {results!r}"


@pytest.mark.asyncio
async def test_ordered_concurrent_enqueue_does_not_deadlock(memory: MemoryEngine):
    """With both transactions inserting in the SAME (sorted) order — exactly what
    PR #2353's ``sorted(unit_ids)`` guarantees per call — there is no cycle. The
    second transaction simply waits on the first shared key and proceeds once the
    first commits; both victim sets land in the queue.
    """
    pool = await memory._get_pool()
    bank_id = f"dl-fix-{uuid.uuid4().hex[:8]}"

    order = sorted([K_LOW, K_HIGH])  # identical order for both workers

    async def worker() -> None:
        async with pool.acquire() as conn:
            async with conn.transaction():
                for uid in order:
                    await _insert_one(conn, bank_id, uid)

    # Sorted order cannot cycle; the timeout only guards against an unexpected hang.
    results = await asyncio.wait_for(
        asyncio.gather(worker(), worker(), return_exceptions=True),
        timeout=30,
    )

    errors = [r for r in results if isinstance(r, BaseException)]
    assert not errors, f"sorted concurrent inserts must not deadlock, got {results!r}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT unit_id FROM graph_maintenance_queue WHERE bank_id = $1 ORDER BY unit_id",
            bank_id,
        )
    assert [r["unit_id"] for r in rows] == order
