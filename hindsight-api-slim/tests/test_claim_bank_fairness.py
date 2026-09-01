"""Per-bank rotation of the worker's slots (#3861).

Claiming used to be a strict global FIFO on ``created_at``, so a bank under
sustained bulk ingest held every slot for as long as its queue lasted: measured
on a six-bank instance, one bulk bank owned 17 of 17 retain slots in 90.6% of
daily samples, and a write to a bank with an *empty* queue timed out behind it.

The claim query now takes one row for the bank after a cursor the poller keeps
per schema — deficit round robin with a quantum of one slot, one level below the
tenant rotation the poller already had — and fills the rest of the pool
oldest-first exactly as before. So no bank is throttled (a bank alone with work
still takes every slot) and no slot is held open for an idle bank; all the
rotation decides is *which* row wins a slot being handed out anyway.

Both tiers are one statement. The tests below pin the three properties that
depend on that: the rotation tier reaches a starved bank, the FIFO tier still
fills the pool behind it, and the FIFO tier doubles as the wrap so the end of a
round costs neither a second query nor an empty claim.

The cursor is a *range* over bank ids, not a set of known banks, because the
starved bank is by definition one this worker has never claimed for.

These call ``ops`` directly rather than ``WorkerPoller.claim_batch`` so the slot
limits under test are exact and not a function of ambient in-flight work in the
shared test database; the poller's own cursor bookkeeping is driven through
``WorkerPoller`` in ``test_poller_cycles_banks_across_claims``.
"""

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

# Use loadgroup to ensure these tests run in the same worker
# since they share database state
pytestmark = pytest.mark.xdist_group("worker_tests")

_TABLE = "async_operations"


@pytest.fixture(scope="session")
def isolated_ops_schema(pg0_db_url):
    """A private, migrated Postgres schema for this file's claim tests.

    ``ops.claim_tasks`` scans the whole schema on the connection's search_path,
    and every test here asserts on an *exact* claim, so another pytest-xdist
    worker's pending rows would make them meaningless. One schema per worker,
    created + migrated once and dropped at session end — the same isolation
    ``test_claim_bank_serialization.py`` uses, under its own name so the two
    files never share rows.
    """
    from hindsight_api.engine.db import create_database_backend
    from hindsight_api.pg0 import resolve_database_url

    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    schema = f"bankfair_iso_{worker}"

    async def _provision() -> str:
        url = await resolve_database_url(pg0_db_url)
        b = create_database_backend("postgresql")
        await b.initialize(url, min_size=1, max_size=2)
        try:
            async with b.get_pool().acquire() as conn:
                # Rebuild from scratch so a schema left by a crashed prior run
                # can't carry stale state into this session.
                await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                await conn.execute(f'CREATE SCHEMA "{schema}"')
            b.run_migrations(url, schema=schema)
        finally:
            await b.shutdown()
        return url

    async def _drop(url: str) -> None:
        b = create_database_backend("postgresql")
        await b.initialize(url, min_size=1, max_size=2)
        try:
            async with b.get_pool().acquire() as conn:
                await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await b.shutdown()

    loop = asyncio.new_event_loop()
    try:
        url = loop.run_until_complete(_provision())
    finally:
        loop.close()

    yield schema

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_drop(url))
    finally:
        loop.close()


@pytest_asyncio.fixture
async def backend(pg0_db_url, isolated_ops_schema):
    """Create a DatabaseBackend whose pool is pinned to this file's private schema."""
    from hindsight_api.engine.db import create_database_backend
    from hindsight_api.pg0 import resolve_database_url

    resolved_url = await resolve_database_url(pg0_db_url)

    async def _use_isolated_schema(conn):
        await conn.execute(f'SET search_path TO "{isolated_ops_schema}", public')

    b = create_database_backend("postgresql")
    await b.initialize(resolved_url, min_size=2, max_size=10, command_timeout=30, init_callback=_use_isolated_schema)
    yield b
    await b.shutdown()


@pytest_asyncio.fixture
async def pool(backend):
    """Expose the raw asyncpg pool from the backend for direct DB access in tests."""
    yield backend.get_pool()


@pytest_asyncio.fixture
async def clean_operations(pool):
    """Clear leftovers from a prior test in this group.

    Safe to be broad: the pool is pinned to this file's private schema, so this
    only ever touches its own rows.
    """
    await pool.execute(f"DELETE FROM {_TABLE}")
    yield
    await pool.execute(f"DELETE FROM {_TABLE} WHERE bank_id LIKE 'test-bankfair-%'")


async def _make_bank(pool) -> str:
    """Create a bank with a unique id so concurrent runs can't collide."""
    bank_id = f"test-bankfair-{uuid.uuid4().hex[:8]}"
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )
    return bank_id


async def _insert_op(
    pool,
    bank_id: str,
    op_type: str = "retain",
    status: str = "pending",
    *,
    age_seconds: float = 0.0,
    worker_id: str | None = None,
    retry_in_seconds: float | None = None,
    serialization_key: str | None = None,
) -> uuid.UUID:
    """Insert one claimable operation row, ``age_seconds`` old."""
    op_id = uuid.uuid4()
    created_at = datetime.now(UTC) - timedelta(seconds=age_seconds)
    payload = json.dumps({"type": op_type, "bank_id": bank_id, "operation_id": str(op_id)})
    claimed_at = created_at if status == "processing" else None
    next_retry_at = datetime.now(UTC) + timedelta(seconds=retry_in_seconds) if retry_in_seconds else None
    await pool.execute(
        f"""
        INSERT INTO {_TABLE}
            (operation_id, bank_id, operation_type, status, task_payload, worker_id,
             created_at, claimed_at, next_retry_at, serialization_key, updated_at)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10, now())
        """,
        op_id,
        bank_id,
        op_type,
        status,
        payload,
        worker_id,
        created_at,
        claimed_at,
        next_retry_at,
        serialization_key,
    )
    return op_id


async def _claim(
    backend,
    *,
    shared: int = 10,
    reserved: dict[str, int] | None = None,
    cursor: str = "",
):
    """Run one claim cycle. Returns the ClaimedOperations as-is."""
    async with backend.acquire() as conn:
        async with conn.transaction():
            return await backend.ops.claim_tasks(
                conn,
                _TABLE,
                "test-bankfair-worker",
                reserved or {},
                shared,
                bank_cursor=cursor,
            )


def _ids(claimed) -> list[str]:
    """Claimed operation ids, in claim order (rotation row first)."""
    return [str(row["operation_id"]) for row in claimed.rows]


@pytest.mark.asyncio
async def test_rotation_reaches_a_bank_buried_behind_a_backlog(pool, backend, clean_operations):
    """#3861 itself: the starved bank's write wins the free slot.

    ``bulk`` is mid-backfill with a queue of older work; ``quiet`` submitted one
    write a moment ago. Under a global FIFO every freed slot went back to
    ``bulk`` until its queue drained, so ``quiet`` waited behind the whole
    backlog. With the cursor on ``bulk``, the rotation tier takes ``quiet``.
    """
    bulk, quiet = sorted([await _make_bank(pool), await _make_bank(pool)])

    backlog = [await _insert_op(pool, bulk, age_seconds=300 - i) for i in range(5)]
    quiet_write = await _insert_op(pool, quiet, age_seconds=1)

    claimed = await _claim(backend, shared=1, cursor=bulk)

    assert _ids(claimed) == [str(quiet_write)]
    assert not {str(op) for op in backlog} & set(_ids(claimed))


@pytest.mark.asyncio
async def test_backfill_behind_the_rotation_is_still_oldest_first(pool, backend, clean_operations):
    """One slot to the bank whose turn it is, the rest by age — in one claim.

    The rotation is a quantum of one. Everything behind it is the claim query
    that was always there, so the backlog still drains oldest-first.
    """
    bulk, quiet = sorted([await _make_bank(pool), await _make_bank(pool)])

    backlog = [await _insert_op(pool, bulk, age_seconds=300 - i) for i in range(5)]
    quiet_write = await _insert_op(pool, quiet, age_seconds=1)

    claimed = await _claim(backend, shared=3, cursor=bulk)

    assert _ids(claimed)[0] == str(quiet_write)
    assert _ids(claimed)[1:] == [str(op) for op in backlog[:2]]


@pytest.mark.asyncio
async def test_a_bank_alone_with_work_is_never_throttled(pool, backend, clean_operations):
    """Rotation is a tier, not a cap: with no one else waiting, one bank takes it all."""
    bulk = await _make_bank(pool)
    backlog = [await _insert_op(pool, bulk, age_seconds=300 - i) for i in range(5)]

    claimed = await _claim(backend, shared=3, cursor=bulk)

    assert _ids(claimed) == [str(op) for op in backlog[:3]]


@pytest.mark.asyncio
async def test_end_of_a_round_costs_neither_a_query_nor_an_empty_claim(pool, backend, clean_operations):
    """A cursor past the last bank still fills the pool, and resets to start a round.

    This is what lets both tiers live in one statement: the FIFO tier *is* the
    wrap, so nothing has to notice the round ended and re-ask. A single-bank
    deployment sits in this state permanently — every claim has a cursor past
    its only bank — and must not pay an empty claim for it.
    """
    bank = await _make_bank(pool)
    queue = [await _insert_op(pool, bank, age_seconds=300 - i) for i in range(4)]

    claimed = await _claim(backend, shared=3, cursor="zzz-past-every-bank")

    assert _ids(claimed) == [str(op) for op in queue[:3]]
    assert claimed.next_bank_cursor == ""


@pytest.mark.asyncio
async def test_cursor_reports_the_bank_the_rotation_served(pool, backend, clean_operations):
    """The new cursor rides back with the rows, so learning it costs no query."""
    first, second = sorted([await _make_bank(pool), await _make_bank(pool)])
    await _insert_op(pool, first, age_seconds=300)
    await _insert_op(pool, second, age_seconds=1)

    assert (await _claim(backend, shared=1, cursor="")).next_bank_cursor == first
    assert (await _claim(backend, shared=1, cursor=first)).next_bank_cursor == second


@pytest.mark.asyncio
async def test_poller_cycles_banks_across_claims(pool, backend, clean_operations):
    """Through the poller: successive claims serve each bank, then start over.

    The cursor is the poller's state, carried between claims the way
    ``_next_schema_idx`` is between tenants.
    """
    from hindsight_api.worker import WorkerPoller

    # Staggered so a plain FIFO would take both of the first bank's rows before
    # touching the second — otherwise same-age queues make FIFO and the rotation
    # visit banks in the same order, and the test passes either way.
    banks = sorted([await _make_bank(pool) for _ in range(3)])
    for offset, bank in enumerate(banks):
        for age in (300 - offset * 100, 290 - offset * 100):
            await _insert_op(pool, bank, age_seconds=age)

    poller = WorkerPoller(backend=backend, worker_id="test-bankfair-rotation", executor=lambda task: None)

    served = []
    async with backend.acquire() as conn:
        for _ in range(len(banks) + 1):
            tasks = await poller._claim_batch_for_schema(conn, None, {}, 1)
            served.append(tasks[0].task_dict["bank_id"])

    # Three banks, then the round restarts at the first.
    assert served == [*banks, banks[0]]


@pytest.mark.asyncio
async def test_rotation_row_is_not_claimed_twice(pool, backend, clean_operations):
    """The rotation's row is also the oldest, so both tiers select it.

    The tiers are unioned, so without the FIFO tier excluding what the rotation
    took, one operation would come back as two claimed tasks and be executed
    twice.
    """
    first, second = sorted([await _make_bank(pool), await _make_bank(pool)])
    oldest = await _insert_op(pool, second, age_seconds=900)
    await _insert_op(pool, first, age_seconds=10)

    claimed = await _claim(backend, shared=10, cursor=first)

    assert _ids(claimed)[0] == str(oldest)
    assert len(_ids(claimed)) == len(set(_ids(claimed)))


@pytest.mark.asyncio
async def test_rotation_skips_banks_with_nothing_claimable(pool, backend, clean_operations):
    """A bank is only owed a turn if it has a claimable row.

    Rows that are processing, deferred, or consolidation don't count —
    consolidation is claimed by its own phase against its own reserved slots.
    An empty or blocked queue leaving the circle is what stops the rotation
    spending visits on banks it cannot serve.
    """
    processing, deferred, consolidating, real = sorted([await _make_bank(pool) for _ in range(4)])

    await _insert_op(pool, processing, status="processing", age_seconds=10, worker_id="other")
    await _insert_op(pool, deferred, age_seconds=10, retry_in_seconds=600)
    await _insert_op(pool, consolidating, op_type="consolidation", age_seconds=10)
    real_row = await _insert_op(pool, real, age_seconds=10)

    claimed = await _claim(backend, shared=1, cursor="")

    assert _ids(claimed) == [str(real_row)]
    assert claimed.next_bank_cursor == real


@pytest.mark.asyncio
async def test_rotation_spans_operation_types(pool, backend, clean_operations):
    """A slot is a slot: the turn goes to the bank, whatever type it queued.

    ``file_convert_retain`` and ``refresh_mental_model`` compete for the same
    shared pool as ``retain``, so the rotation is over banks rather than over
    one type's queue.
    """
    bulk, quiet = sorted([await _make_bank(pool), await _make_bank(pool)])

    bulk_convert = await _insert_op(pool, bulk, op_type="file_convert_retain", age_seconds=300)
    quiet_refresh = await _insert_op(pool, quiet, op_type="refresh_mental_model", age_seconds=1)

    claimed = await _claim(backend, shared=1, cursor=bulk)

    assert _ids(claimed) == [str(quiet_refresh)]
    assert str(bulk_convert) not in _ids(claimed)


@pytest.mark.asyncio
async def test_rotation_does_not_break_document_serialization(pool, backend, clean_operations):
    """A bank's turn cannot hand out a retain for a document already in flight.

    The rotation tier carries the same predicates as the FIFO tier; appends to
    one document stay serialised, so the bank's turn passes to its next
    claimable row rather than racing the in-flight retain.
    """
    bulk, quiet = sorted([await _make_bank(pool), await _make_bank(pool)])

    await _insert_op(pool, bulk, age_seconds=300)
    await _insert_op(pool, quiet, status="processing", age_seconds=100, worker_id="other", serialization_key="doc-1")
    blocked = await _insert_op(pool, quiet, age_seconds=50, serialization_key="doc-1")
    free = await _insert_op(pool, quiet, age_seconds=40, serialization_key="doc-2")

    claimed = await _claim(backend, shared=1, cursor=bulk)

    assert _ids(claimed) == [str(free)]
    assert str(blocked) not in _ids(claimed)
