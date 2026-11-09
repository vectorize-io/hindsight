"""Queue inserts made inside a long write transaction must not block other writers (#6417).

A retain or an edit queues the units it disturbed for relinking. The queue insert
locks rows that are already queued (#3034), and those locks last until commit.
When the insert ran early, a second writer of the same bank queueing an
overlapping unit waited for the first writer's whole transaction.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from hindsight_api.engine.graph_maintenance import enqueue_relink_victims
from hindsight_api.engine.memory_engine import MemoryEngine

HOLD_S = 3.0


async def _queued(backend, bank_id: str) -> set[str]:
    async with backend.acquire() as conn:
        rows = await conn.fetch("SELECT unit_id FROM graph_maintenance_queue WHERE bank_id = $1", bank_id)
    return {str(r["unit_id"]) for r in rows}


@pytest.mark.asyncio
async def test_overlapping_writer_does_not_wait_for_long_transaction(memory: MemoryEngine):
    backend = await memory._get_backend()
    bank_id = f"gmq-commit-{uuid.uuid4().hex[:8]}"
    shared, only_a, only_b = (str(uuid.uuid4()) for _ in range(3))
    # Already queued, so both writers hit the conflicting-row lock.
    async with backend.acquire() as conn:
        await enqueue_relink_victims(conn, bank_id, [shared], include_affected_units=True)

    a_enqueued = asyncio.Event()

    async def writer_a() -> None:
        async with backend.transaction() as conn:
            await enqueue_relink_victims(conn, bank_id, [shared, only_a], include_affected_units=True)
            a_enqueued.set()
            await asyncio.sleep(HOLD_S)  # the rest of a long retain/edit

    async def writer_b() -> float:
        await a_enqueued.wait()
        start = time.monotonic()
        async with backend.transaction() as conn:
            await enqueue_relink_victims(conn, bank_id, [shared, only_b], include_affected_units=True)
        return time.monotonic() - start

    _, b_elapsed = await asyncio.gather(writer_a(), writer_b())

    assert b_elapsed < HOLD_S / 3, f"writer B waited {b_elapsed:.2f}s on writer A's transaction"
    assert await _queued(backend, bank_id) == {shared, only_a, only_b}


@pytest.mark.asyncio
async def test_rolled_back_transaction_queues_nothing(memory: MemoryEngine):
    backend = await memory._get_backend()
    bank_id = f"gmq-rollback-{uuid.uuid4().hex[:8]}"
    with pytest.raises(RuntimeError):
        async with backend.transaction() as conn:
            await enqueue_relink_victims(conn, bank_id, [str(uuid.uuid4())], include_affected_units=True)
            raise RuntimeError("abort")
    assert await _queued(backend, bank_id) == set()


@pytest.mark.asyncio
async def test_writes_held_for_commit_merge_into_one_sorted_flush(memory: MemoryEngine):
    """Several queue writes in one transaction become one sorted insert at commit:
    one statement takes every row lock in ascending order, so two transactions
    can't each hold a row the other's second insert needs."""
    backend = await memory._get_backend()
    flushed: list[list[int]] = []

    async def flush(_conn, ids: list[int]) -> None:
        flushed.append(ids)

    async with backend.transaction() as conn:
        await conn.add_before_commit("k", [5, 1], flush)
        await conn.add_before_commit("k", [3, 1], flush)
        async with conn.transaction():  # a savepoint flushes nothing on its own
            await conn.add_before_commit("k", [2], flush)
        assert flushed == []
    assert flushed == [[1, 2, 3, 5]]

    # Outside a transaction there is no commit to wait for.
    async with backend.acquire() as conn:
        await conn.add_before_commit("k", [9, 7], flush)
    assert flushed[-1] == [7, 9]
