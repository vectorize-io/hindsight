"""Concurrency tests for the ``graph_maintenance_queue`` re-enqueue race (#3034).

Before the fix the queue used ``ON CONFLICT DO NOTHING`` (PG) /
``IGNORE_ROW_ON_DUPKEY_INDEX`` (Oracle) on enqueue. Neither locks the existing
row, so a mutation that re-enqueued an already-queued unit could not serialise
against a worker that concurrently claimed (deleted) that row and processed the
unit's *pre-mutation* state. The re-enqueue signal was silently dropped and the
unit's derived temporal/semantic links were left stale with an empty queue.

The fix makes a duplicate enqueue take the existing row's lock (PG
``DO UPDATE`` no-op / Oracle ``MERGE ... WHEN MATCHED``) and makes the worker
claim lock those rows ``FOR UPDATE`` in ``(bank_id, unit_id)`` order — the same
order the enqueue takes them — so the two serialise for a given key and can
never cycle.

These run against the real Postgres test DB and drive genuinely-concurrent
transactions (the only way to observe a *database*-level lock interleaving).
The Oracle protocol is the same shape (MERGE + ordered per-row delete + the
Pass-1 retry backstop); its live integration reproduction is tracked as a
follow-up per the issue (the Python Oracle suite doesn't run in PR CI).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine

TABLE = "graph_maintenance_queue"

# Two keys with an unambiguous sort order (low < high as UUIDs and as text).
K_LOW = uuid.UUID("00000000-0000-4000-8000-000000000001")
K_HIGH = uuid.UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")


async def _queue_ids(conn, bank_id: str) -> list[str]:
    rows = await conn.fetch(
        f"SELECT unit_id FROM {TABLE} WHERE bank_id = $1 ORDER BY unit_id",
        bank_id,
    )
    return [str(r["unit_id"]) for r in rows]


@pytest.mark.asyncio
async def test_duplicate_enqueue_locks_row_and_blocks_claim(memory: MemoryEngine):
    """A duplicate enqueue holds the queue row's lock until it commits, so a
    concurrent worker claim cannot delete-and-process the stale row underneath it.

    This is the core #3034 guard. With the old ``DO NOTHING`` the duplicate
    enqueue took no lock and the claim below would return immediately (deleting
    the row while the mutation was still in flight) — so the ``TimeoutError``
    assertion here would fail. ``DO UPDATE`` makes the claim wait for the commit.
    """
    pool = await memory._get_pool()
    backend = await memory._get_backend()
    bank_id = f"gmq-lock-{uuid.uuid4().hex[:8]}"
    unit = uuid.uuid4()

    # Seed a committed queue row (an earlier, not-yet-drained maintenance request).
    async with pool.acquire() as seed:
        await backend.ops.enqueue_graph_maintenance(seed, TABLE, bank_id, [unit])

    # A mutation re-enqueues the same unit inside its (uncommitted) transaction.
    mutation = await pool.acquire()
    try:
        mut_tx = mutation.transaction()
        await mut_tx.start()
        await backend.ops.enqueue_graph_maintenance(mutation, TABLE, bank_id, [unit])

        async def claim_batch() -> list[str]:
            async with pool.acquire() as worker:
                async with worker.transaction():
                    return await backend.ops.claim_graph_maintenance_batch(worker, TABLE, bank_id, 50)

        claim_task = asyncio.create_task(claim_batch())

        # The claim must block on the row lock the mutation holds — it cannot
        # complete while the mutation is uncommitted.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(claim_task), timeout=1.0)

        # Mutation commits; the row is still present (the SET is a no-op), so the
        # worker now claims it and processes the committed state.
        await mut_tx.commit()
        claimed = await asyncio.wait_for(claim_task, timeout=10.0)
        assert claimed == [str(unit)]
    finally:
        await pool.release(mutation)

    async with pool.acquire() as conn:
        assert await _queue_ids(conn, bank_id) == []


@pytest.mark.asyncio
async def test_worker_claim_before_enqueue_reinserts_signal(memory: MemoryEngine):
    """The other interleaving: the worker claims (deletes) the row first, then
    the mutation's enqueue must land a *fresh* row rather than silently no-op'ing.

    Under the old protocol the mutation's ``INSERT ... DO NOTHING`` could observe
    the soon-to-be-deleted row and skip, losing the signal. ``DO UPDATE`` blocks
    on the worker's delete lock and, once the delete commits, re-drives the insert
    — so a new queue row survives and the follow-up drain will reprocess the unit.
    """
    pool = await memory._get_pool()
    backend = await memory._get_backend()
    bank_id = f"gmq-reins-{uuid.uuid4().hex[:8]}"
    unit = uuid.uuid4()

    async with pool.acquire() as seed:
        await backend.ops.enqueue_graph_maintenance(seed, TABLE, bank_id, [unit])

    # Worker claims the row inside an uncommitted transaction (holds the delete lock).
    worker = await pool.acquire()
    try:
        worker_tx = worker.transaction()
        await worker_tx.start()
        claimed = await backend.ops.claim_graph_maintenance_batch(worker, TABLE, bank_id, 50)
        assert claimed == [str(unit)]

        async def enqueue_dup() -> None:
            async with pool.acquire() as mutation:
                async with mutation.transaction():
                    await backend.ops.enqueue_graph_maintenance(mutation, TABLE, bank_id, [unit])

        enqueue_task = asyncio.create_task(enqueue_dup())

        # The enqueue must block on the worker's (uncommitted) delete lock.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(enqueue_task), timeout=1.0)

        # Worker commits the delete; the enqueue then inserts a fresh row.
        await worker_tx.commit()
        await asyncio.wait_for(enqueue_task, timeout=10.0)
    finally:
        await pool.release(worker)

    async with pool.acquire() as conn:
        assert await _queue_ids(conn, bank_id) == [str(unit)], (
            "re-enqueue signal must survive the worker winning the race"
        )


@pytest.mark.asyncio
async def test_claim_picks_oldest_but_drains_cleanly(memory: MemoryEngine):
    """The ordered-lock claim still selects the oldest batch by ``enqueued_at``
    (the CTE only normalises *lock* order, not *selection* order) and deletes
    exactly what it returns.
    """
    pool = await memory._get_pool()
    backend = await memory._get_backend()
    bank_id = f"gmq-oldest-{uuid.uuid4().hex[:8]}"
    # Insert three rows with explicit, distinct enqueued_at values. The two
    # oldest have the HIGHEST unit_ids, so a claim that confused lock order with
    # selection order would pick the wrong rows.
    oldest = (uuid.UUID("ffffffff-ffff-4fff-8fff-fffffffffff1"), datetime(2020, 1, 1, tzinfo=UTC))
    middle = (uuid.UUID("ffffffff-ffff-4fff-8fff-fffffffffff2"), datetime(2020, 1, 2, tzinfo=UTC))
    newest = (uuid.UUID("00000000-0000-4000-8000-000000000009"), datetime(2020, 1, 3, tzinfo=UTC))
    async with pool.acquire() as conn:
        for unit, ts in (oldest, middle, newest):
            await conn.execute(
                f"INSERT INTO {TABLE} (bank_id, unit_id, enqueued_at) VALUES ($1, $2, $3)",
                bank_id,
                unit,
                ts,
            )
        claimed = await backend.ops.claim_graph_maintenance_batch(conn, TABLE, bank_id, 2)
        assert set(claimed) == {str(oldest[0]), str(middle[0])}
        # Only the newest row is left behind.
        assert await _queue_ids(conn, bank_id) == [str(newest[0])]


@pytest.mark.asyncio
async def test_concurrent_enqueue_and_claim_never_deadlock(memory: MemoryEngine):
    """A real re-enqueue and a real worker claim of the same two units, run
    concurrently, must never cycle.

    Both the enqueue upsert and the claim are single statements, so we can't
    pause mid-statement to force a specific interleave (the way the row-at-a-time
    tests in ``test_graph_maintenance_deadlock.py`` do). Instead we stress the
    overlap across many rounds. The units are seeded with ``enqueued_at`` in the
    REVERSE of unit_id order, so the claim's oldest-first *selection* (K_HIGH,
    K_LOW) differs from its *lock* order (K_LOW, K_HIGH) — which is exactly the
    order the enqueue takes them. Shared lock order ⇒ no cycle. If the worker
    ever regressed to locking in ``enqueued_at`` order, these overlapping runs
    would deadlock against the producer.
    """
    pool = await memory._get_pool()
    backend = await memory._get_backend()

    older = datetime(2020, 1, 1, tzinfo=UTC)  # earlier enqueued_at ...
    newer = datetime(2020, 1, 2, tzinfo=UTC)  # ... but on the LOWER unit_id

    for round_no in range(20):
        bank_id = f"gmq-dl-{round_no}-{uuid.uuid4().hex[:8]}"
        async with pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO {TABLE} (bank_id, unit_id, enqueued_at) VALUES ($1, $2, $3)",
                bank_id,
                K_HIGH,
                older,
            )
            await conn.execute(
                f"INSERT INTO {TABLE} (bank_id, unit_id, enqueued_at) VALUES ($1, $2, $3)",
                bank_id,
                K_LOW,
                newer,
            )

        async def reenqueue(bid: str) -> None:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await backend.ops.enqueue_graph_maintenance(conn, TABLE, bid, [K_LOW, K_HIGH])

        async def claim(bid: str) -> list[str]:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    return await backend.ops.claim_graph_maintenance_batch(conn, TABLE, bid, 50)

        results = await asyncio.wait_for(
            asyncio.gather(reenqueue(bank_id), claim(bank_id), return_exceptions=True),
            timeout=20,
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        assert not errors, f"round {round_no}: concurrent enqueue/claim must not deadlock, got {results!r}"
