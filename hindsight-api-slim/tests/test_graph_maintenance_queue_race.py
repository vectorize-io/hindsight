"""Concurrency tests for the ``graph_maintenance_queue`` claim protocol (#3034, #6417).

A mutation re-queues the units it changed; a worker claims queued units, relinks
them and deletes their rows. The re-queue signal must never be lost: if the worker
processed a unit's pre-mutation state, the row has to stay queued for another pass.

The protocol: the claim stamps ``claimed_at`` in a short transaction of its own;
the worker deletes the rows at the end of its relink only where the stamp is still
its own; a re-queue clears the stamp. Neither side holds queue-row locks for longer
than one statement before its commit, so writers don't queue behind a relink
(#6417).

These run against the real Postgres test DB with genuinely concurrent transactions.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine

TABLE = "graph_maintenance_queue"
STALE = timedelta(minutes=10)

# Two keys with an unambiguous sort order (low < high as UUIDs and as text).
K_LOW = uuid.UUID("00000000-0000-4000-8000-000000000001")
K_HIGH = uuid.UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")


async def _queue_ids(conn, bank_id: str) -> list[str]:
    rows = await conn.fetch(
        f"SELECT unit_id FROM {TABLE} WHERE bank_id = $1 ORDER BY unit_id",
        bank_id,
    )
    return [str(r["unit_id"]) for r in rows]


async def _claim(
    backend, conn, bank_id: str, limit: int = 50, now: datetime | None = None
) -> tuple[list[str], datetime]:
    claimed_at = now or datetime.now(UTC)
    ids = await backend.ops.claim_graph_maintenance_batch(conn, TABLE, bank_id, limit, claimed_at, claimed_at - STALE)
    return ids, claimed_at


@pytest.mark.asyncio
async def test_requeue_during_relink_survives_finish(memory: MemoryEngine):
    """A unit re-queued while a worker relinks it stays queued after the worker
    finishes — the worker read the pre-mutation state, so another pass is owed."""
    pool = await memory._get_pool()
    backend = await memory._get_backend()
    bank_id = f"gmq-requeue-{uuid.uuid4().hex[:8]}"
    unit = uuid.uuid4()

    async with pool.acquire() as conn:
        await backend.ops.enqueue_graph_maintenance(conn, TABLE, bank_id, [unit])
        claimed, claimed_at = await _claim(backend, conn, bank_id)
        assert claimed == [str(unit)]
        # The mutation commits while the worker is still relinking.
        await backend.ops.enqueue_graph_maintenance(conn, TABLE, bank_id, [unit])
        await backend.ops.finish_graph_maintenance_batch(conn, TABLE, bank_id, claimed, claimed_at)
        assert await _queue_ids(conn, bank_id) == [str(unit)]
        # The next claim takes it again and finishing clears it.
        claimed, claimed_at = await _claim(backend, conn, bank_id)
        await backend.ops.finish_graph_maintenance_batch(conn, TABLE, bank_id, claimed, claimed_at)
        assert await _queue_ids(conn, bank_id) == []


@pytest.mark.asyncio
async def test_requeue_does_not_wait_for_a_running_relink(memory: MemoryEngine):
    """The claim commits before the relink starts, so a writer re-queuing a
    claimed unit goes straight through instead of waiting out the relink (#6417)."""
    pool = await memory._get_pool()
    backend = await memory._get_backend()
    bank_id = f"gmq-nowait-{uuid.uuid4().hex[:8]}"
    unit = uuid.uuid4()

    async with pool.acquire() as conn:
        await backend.ops.enqueue_graph_maintenance(conn, TABLE, bank_id, [unit])
        async with conn.transaction():
            claimed, claimed_at = await _claim(backend, conn, bank_id)

    worker = await pool.acquire()
    try:
        relink_tx = worker.transaction()
        await relink_tx.start()  # the relink is running and holds no queue lock

        async with pool.acquire() as mutation:
            async with mutation.transaction():
                await asyncio.wait_for(
                    backend.ops.enqueue_graph_maintenance(mutation, TABLE, bank_id, [unit]), timeout=2.0
                )

        await backend.ops.finish_graph_maintenance_batch(worker, TABLE, bank_id, claimed, claimed_at)
        await relink_tx.commit()
    finally:
        await pool.release(worker)

    async with pool.acquire() as conn:
        assert await _queue_ids(conn, bank_id) == [str(unit)]


@pytest.mark.asyncio
async def test_finish_racing_requeue_keeps_the_signal(memory: MemoryEngine):
    """The worker's finishing delete and a re-queue of the same row, concurrent:
    whichever commits first, the row is queued afterwards."""
    pool = await memory._get_pool()
    backend = await memory._get_backend()
    bank_id = f"gmq-finish-{uuid.uuid4().hex[:8]}"
    unit = uuid.uuid4()

    async with pool.acquire() as conn:
        await backend.ops.enqueue_graph_maintenance(conn, TABLE, bank_id, [unit])
        claimed, claimed_at = await _claim(backend, conn, bank_id)

    worker = await pool.acquire()
    try:
        tx = worker.transaction()
        await tx.start()
        await backend.ops.finish_graph_maintenance_batch(worker, TABLE, bank_id, claimed, claimed_at)

        async def requeue() -> None:
            async with pool.acquire() as mutation:
                async with mutation.transaction():
                    await backend.ops.enqueue_graph_maintenance(mutation, TABLE, bank_id, [unit])

        requeue_task = asyncio.create_task(requeue())
        # It waits on the uncommitted delete (one statement before the worker's commit) ...
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(requeue_task), timeout=1.0)
        await tx.commit()
        # ... then inserts a fresh row.
        await asyncio.wait_for(requeue_task, timeout=10.0)
    finally:
        await pool.release(worker)

    async with pool.acquire() as conn:
        assert await _queue_ids(conn, bank_id) == [str(unit)]


@pytest.mark.asyncio
async def test_claim_picks_oldest_and_skips_live_claims(memory: MemoryEngine):
    """Selection is oldest-first by ``enqueued_at``; a row another worker holds is
    skipped until its claim goes stale."""
    pool = await memory._get_pool()
    backend = await memory._get_backend()
    bank_id = f"gmq-oldest-{uuid.uuid4().hex[:8]}"
    # The two oldest have the HIGHEST unit_ids, so a claim that confused lock
    # order with selection order would pick the wrong rows.
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
        now = datetime.now(UTC)
        first, _ = await _claim(backend, conn, bank_id, limit=2, now=now)
        assert set(first) == {str(oldest[0]), str(middle[0])}
        # A second worker gets only what is left.
        second, _ = await _claim(backend, conn, bank_id, now=now + timedelta(seconds=1))
        assert second == [str(newest[0])]
        # Nothing claimable until the first claims are stale (their worker died).
        assert (await _claim(backend, conn, bank_id, now=now + timedelta(seconds=2)))[0] == []
        revived, _ = await _claim(backend, conn, bank_id, now=now + STALE + timedelta(seconds=5))
        assert set(revived) == {str(oldest[0]), str(middle[0]), str(newest[0])}
        # Claiming deletes nothing: rows leave only through finish.
        assert len(await _queue_ids(conn, bank_id)) == 3


@pytest.mark.asyncio
async def test_concurrent_enqueue_and_claim_never_deadlock(memory: MemoryEngine):
    """A re-enqueue and a claim of the same two units, run concurrently over many
    rounds, must never cycle. ``enqueued_at`` is seeded in the REVERSE of unit_id
    order, so the claim's oldest-first selection differs from its lock order —
    which is the order the enqueue takes them."""
    pool = await memory._get_pool()
    backend = await memory._get_backend()

    older = datetime(2020, 1, 1, tzinfo=UTC)
    newer = datetime(2020, 1, 2, tzinfo=UTC)

    for round_no in range(20):
        bank_id = f"gmq-dl-{round_no}-{uuid.uuid4().hex[:8]}"
        async with pool.acquire() as conn:
            for unit, ts in ((K_HIGH, older), (K_LOW, newer)):
                await conn.execute(
                    f"INSERT INTO {TABLE} (bank_id, unit_id, enqueued_at) VALUES ($1, $2, $3)",
                    bank_id,
                    unit,
                    ts,
                )

        async def reenqueue(bid: str) -> None:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await backend.ops.enqueue_graph_maintenance(conn, TABLE, bid, [K_LOW, K_HIGH])

        async def claim_and_finish(bid: str) -> None:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    ids, claimed_at = await _claim(backend, conn, bid)
                async with conn.transaction():
                    await backend.ops.finish_graph_maintenance_batch(conn, TABLE, bid, ids, claimed_at)

        results = await asyncio.wait_for(
            asyncio.gather(reenqueue(bank_id), claim_and_finish(bank_id), return_exceptions=True),
            timeout=20,
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        assert not errors, f"round {round_no}: concurrent enqueue/claim must not deadlock, got {results!r}"
