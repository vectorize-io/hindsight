"""Reproduces the concurrent-insert deadlock on ``graph_maintenance_queue``
that PR #2353 targets, and demonstrates that a shared insertion order cures it.

Also covers the ``entity_cooccurrences`` deadlock between the Pass 2/3 sweep
(``prune_stale_cooccurrences``) and retain's concurrent cooccurrence upserts —
the dominant ``DeadlockDetectedError`` cause in production logs (39 of 41
occurrences in one week) — and the ``retry_with_backoff`` fix around that
sweep in ``run_graph_maintenance_job``.

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
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from asyncpg.exceptions import DeadlockDetectedError

from hindsight_api import RequestContext
from hindsight_api.engine.graph_maintenance import run_graph_maintenance_job
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


async def _insert_entity(conn, bank_id: str, entity_id: uuid.UUID, name: str) -> None:
    await conn.execute(
        """
        INSERT INTO entities (id, bank_id, canonical_name, first_seen, last_seen, mention_count)
        VALUES ($1, $2, $3, NOW(), NOW(), 1)
        """,
        entity_id,
        bank_id,
        name,
    )


async def _insert_cooccurrence(conn, entity_a: uuid.UUID, entity_b: uuid.UUID) -> None:
    first, second = sorted([entity_a, entity_b])
    await conn.execute(
        """
        INSERT INTO entity_cooccurrences (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
        VALUES ($1, $2, 1, NOW())
        """,
        first,
        second,
    )


@pytest.mark.asyncio
async def test_unordered_concurrent_sweep_and_upsert_deadlocks(memory: MemoryEngine):
    """Reproduces the ``prune_stale_cooccurrences`` deadlock seen in production
    (``asyncpg.exceptions.DeadlockDetectedError`` on ``entity_cooccurrences``,
    39 of 41 deadlock occurrences in one week of logs).

    ``prune_stale_cooccurrences`` (graph maintenance) scans/locks
    ``entity_cooccurrences`` rows for a bank in whatever order its join/NOT
    EXISTS plan visits them. Concurrently, retain's cooccurrence upserts
    (``entity_resolver._flush_pending``) lock rows in sorted
    ``(entity_id_1, entity_id_2)`` order — sorted specifically to avoid
    deadlocking against *other* concurrent upserts, but that guarantee says
    nothing about the maintenance sweep's scan order. Two transactions
    touching the same two cooccurrence rows in opposite orders cycle, and
    Postgres aborts one with ``DeadlockDetectedError``.

    This models each side as a sequence of single-row statements with a
    barrier between them (same technique as the queue-insert test above) so
    the interleaving that produces the cycle is deterministic rather than
    racy.
    """
    pool = await memory._get_pool()
    bank_id = f"dl-sweep-{uuid.uuid4().hex[:8]}"

    # Two cooccurrence rows. entity_cooccurrence_order_check pins each row's
    # own (entity_id_1, entity_id_2) internally, but the two ROWS themselves
    # sort as pair_low < pair_high by their first UUID.
    pair_low = (uuid.UUID("00000000-0000-4000-8000-000000000001"), uuid.UUID("00000000-0000-4000-8000-000000000002"))
    pair_high = (uuid.UUID("ffffffff-ffff-4fff-8fff-fffffffffffe"), uuid.UUID("ffffffff-ffff-4fff-8fff-ffffffffffff"))

    async with pool.acquire() as conn:
        for entity_id, name in [
            (pair_low[0], "low-a"),
            (pair_low[1], "low-b"),
            (pair_high[0], "high-a"),
            (pair_high[1], "high-b"),
        ]:
            await _insert_entity(conn, bank_id, entity_id, name)
        await _insert_cooccurrence(conn, *pair_low)
        await _insert_cooccurrence(conn, *pair_high)

    barrier = asyncio.Barrier(2)

    async def touch(conn, entity_a: uuid.UUID, entity_b: uuid.UUID) -> None:
        first, second = sorted([entity_a, entity_b])
        await conn.execute(
            "UPDATE entity_cooccurrences SET cooccurrence_count = cooccurrence_count + 1 "
            "WHERE entity_id_1 = $1 AND entity_id_2 = $2",
            first,
            second,
        )

    async def sweep_worker() -> None:
        """Mimics the maintenance scan visiting pair_high before pair_low."""
        async with pool.acquire() as conn:
            async with conn.transaction():
                await touch(conn, *pair_high)
                await barrier.wait()
                await touch(conn, *pair_low)

    async def upsert_worker() -> None:
        """Mimics retain's sorted upsert — always ascending order."""
        async with pool.acquire() as conn:
            async with conn.transaction():
                await touch(conn, *pair_low)
                await barrier.wait()
                await touch(conn, *pair_high)

    results = await asyncio.wait_for(
        asyncio.gather(sweep_worker(), upsert_worker(), return_exceptions=True),
        timeout=30,
    )

    deadlocks = [r for r in results if isinstance(r, DeadlockDetectedError)]
    assert deadlocks, f"expected one transaction aborted with DeadlockDetectedError, got {results!r}"


@pytest.mark.asyncio
async def test_graph_maintenance_sweep_retries_on_deadlock(memory: MemoryEngine, request_context: RequestContext):
    """``run_graph_maintenance_job``'s Pass 2/3 sweep must survive a deadlock.

    Fix for the production bug reproduced above: the sweep
    (``prune_orphan_entities`` + ``prune_stale_cooccurrences``) now runs
    inside ``retry_with_backoff``, which already special-cases
    ``DeadlockDetectedError``. Both prunes are idempotent bank-wide deletes,
    so retrying the whole transaction after a deadlock is safe. Simulates the
    deadlock via a mock (real two-connection reproduction is covered above;
    this asserts the *application* behaviour — one transient
    DeadlockDetectedError must not fail the job) and asserts the retry
    actually happens and the job still returns correct prune counts.
    """
    bank_id = f"dl-fix-sweep-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    pool = await memory._get_pool()
    ent_a = uuid.uuid4()
    ent_b = uuid.uuid4()
    async with pool.acquire() as conn:
        await _insert_entity(conn, bank_id, ent_a, "alice")
        await _insert_entity(conn, bank_id, ent_b, "bob")
        # Both entities are referenced by units (so prune_orphan_entities
        # leaves them alone), but by DIFFERENT units — no unit references
        # both, so the cooccurrence itself is stale and eligible for
        # prune_stale_cooccurrences.
        for entity_id, text in [(ent_a, "with_alice"), (ent_b, "with_bob")]:
            unit_id = uuid.uuid4()
            await conn.execute(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, created_at, updated_at)
                VALUES ($1, $2, $3, 'experience', NOW(), NOW(), NOW())
                """,
                unit_id,
                bank_id,
                text,
            )
            await conn.execute(
                "INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)",
                unit_id,
                entity_id,
            )
        await conn.execute(
            """
            INSERT INTO entity_cooccurrences (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
            VALUES ($1, $2, 1, $3)
            """,
            *sorted([ent_a, ent_b]),
            datetime.now(UTC),
        )

    backend = await memory._get_backend()
    real_prune = backend.ops.prune_stale_cooccurrences
    call_count = 0

    async def flaky_prune(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise DeadlockDetectedError("deadlock detected")
        return await real_prune(*args, **kwargs)

    backend.ops.prune_stale_cooccurrences = AsyncMock(side_effect=flaky_prune)
    try:
        result = await run_graph_maintenance_job(memory, bank_id, request_context)
    finally:
        backend.ops.prune_stale_cooccurrences = real_prune

    assert call_count == 2, "expected exactly one deadlock retry"
    assert result["orphan_entities_pruned"] == 0
    assert result["stale_cooccurrences_pruned"] == 1


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
