"""Per-bank serialisation of graph_maintenance at claim time (#3230).

Every graph_maintenance run for a bank is interchangeable — the payload carries
only ``bank_id`` and ``run_graph_maintenance_job`` drains the whole queue — so a
second concurrent run for one bank adds no work while convoying on the first
run's queue-row locks (``claim_graph_maintenance_batch`` locks ``FOR UPDATE``
with no ``SKIP LOCKED``) and holding a worker slot.

``claim_tasks`` therefore refuses to claim a graph_maintenance row for a bank
that already has one in flight, and takes at most one per bank per batch. It
does that with a predicate on the ordinary shared-pool query rather than a
claim phase of its own, so graph_maintenance keeps competing by ``created_at``
instead of dropping below every other operation type — it has no reserved-slot
floor, and the poller's fairness pass claims with ``shared_limit=1``.

These call ``ops.claim_tasks`` directly rather than ``WorkerPoller.claim_batch``
so the slot limits under test are exact and not a function of ambient in-flight
work in the shared test database.
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
    and tests like ``test_not_starved_by_newer_pending_work`` assert on an *exact*
    single-slot claim — both are meaningless if another pytest-xdist worker's
    pending rows are visible. The previous fixture kept itself clean with a global
    ``DELETE FROM async_operations WHERE status = 'pending'``, which under xdist
    deleted those other workers' in-flight operations mid-run (e.g. a refresh op
    sitting ``pending`` for the window between ``_submit_async_operation``
    committing it and ``SyncTaskBackend`` marking it ``completed``, which then
    read back as ``not_found`` and flaked an unrelated test).

    So give this file its own schema: "the whole schema" is then only its own
    rows, and its cleanup can never touch ``public``. One schema per worker,
    created + migrated once and dropped at session end. ``search_path`` is set on
    the pool in :func:`backend`, so every table reference here resolves into it.
    """
    from hindsight_api.engine.db import create_database_backend
    from hindsight_api.pg0 import resolve_database_url

    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    schema = f"gmclaim_iso_{worker}"

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
            # run_migrations is sync; call it with the loop running (as elsewhere
            # in the suite) — it builds banks/async_operations/etc. in the schema.
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
        # init runs once per new connection, setup runs on every acquire (after
        # asyncpg's release-time RESET ALL), so this pins search_path for the
        # pool's whole lifetime — every unqualified table resolves into the
        # private schema, so claims and cleanup never see the shared public one.
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

    Safe to be broad now: the pool is pinned to this file's private schema
    (:func:`isolated_ops_schema`), so this only ever touches its own rows.
    """
    await pool.execute("DELETE FROM async_operations WHERE status = 'pending'")
    yield
    await pool.execute("DELETE FROM async_operations WHERE bank_id LIKE 'test-gmclaim-%'")


async def _make_bank(pool) -> str:
    """Create a bank with a unique id so concurrent runs can't collide."""
    bank_id = f"test-gmclaim-{uuid.uuid4().hex[:8]}"
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )
    return bank_id


async def _insert_op(
    pool,
    bank_id: str,
    op_type: str,
    status: str = "pending",
    *,
    created_at: datetime | None = None,
    claimed_at: datetime | None = None,
    next_retry_at: datetime | None = None,
    worker_id: str | None = None,
) -> uuid.UUID:
    """Insert one operation row with a claimable payload."""
    op_id = uuid.uuid4()
    payload = json.dumps({"type": op_type, "bank_id": bank_id, "operation_id": str(op_id)})
    await pool.execute(
        f"""
        INSERT INTO {_TABLE}
            (operation_id, bank_id, operation_type, status, task_payload, worker_id, next_retry_at,
             created_at, claimed_at, updated_at)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7,
                COALESCE($8, now()), $9, now())
        """,
        op_id,
        bank_id,
        op_type,
        status,
        payload,
        worker_id,
        next_retry_at,
        created_at,
        claimed_at,
    )
    return op_id


async def _claim(backend, *, shared: int = 10, reserved: dict[str, int] | None = None) -> set[str]:
    """Run one claim cycle and return the claimed operation ids as strings."""
    async with backend.acquire() as conn:
        async with conn.transaction():
            rows = await backend.ops.claim_tasks(
                conn,
                _TABLE,
                "test-gmclaim-worker",
                reserved or {},
                shared,
            )
    return {str(row["operation_id"]) for row in rows}


async def _status_of(pool, op_id: uuid.UUID) -> str:
    return await pool.fetchval(f"SELECT status FROM {_TABLE} WHERE operation_id = $1", op_id)


@pytest.mark.asyncio
async def test_not_claimed_while_bank_has_run_in_flight(pool, backend, clean_operations):
    """A pending graph_maintenance is left alone while its bank has one processing."""
    bank = await _make_bank(pool)
    await _insert_op(pool, bank, "graph_maintenance", "processing", claimed_at=datetime.now(UTC), worker_id="other")
    pending = await _insert_op(pool, bank, "graph_maintenance")

    claimed = await _claim(backend)

    assert str(pending) not in claimed
    assert await _status_of(pool, pending) == "pending"


@pytest.mark.asyncio
async def test_single_batch_claims_at_most_one_per_bank(pool, backend, clean_operations):
    """One batch takes a single graph_maintenance per bank — the oldest.

    Multiple pending rows for one bank are reachable despite submit-time dedup:
    ``recover_own_tasks`` resets every processing row for a worker back to
    pending in one statement, and _schedule_retry / _defer_operation / the admin
    recover command each restore rows independently.
    """
    bank = await _make_bank(pool)
    base = datetime.now(UTC) - timedelta(minutes=10)
    op_ids = [
        await _insert_op(pool, bank, "graph_maintenance", created_at=base + timedelta(seconds=i)) for i in range(5)
    ]

    claimed = await _claim(backend)

    ours = [op for op in op_ids if str(op) in claimed]
    assert len(ours) == 1, f"expected exactly one same-bank claim, got {len(ours)}"
    assert ours[0] == op_ids[0], "the oldest pending row should be the one claimed"


@pytest.mark.asyncio
async def test_idle_bank_still_claimed_while_another_is_busy(pool, backend, clean_operations):
    """The guard is per bank, not global: a different bank is unaffected."""
    busy_bank = await _make_bank(pool)
    idle_bank = await _make_bank(pool)
    await _insert_op(pool, busy_bank, "graph_maintenance", "processing", claimed_at=datetime.now(UTC))
    blocked = await _insert_op(pool, busy_bank, "graph_maintenance")
    claimable = await _insert_op(pool, idle_bank, "graph_maintenance")

    claimed = await _claim(backend)

    assert str(claimable) in claimed
    assert str(blocked) not in claimed


@pytest.mark.asyncio
async def test_other_operation_types_unaffected(pool, backend, clean_operations):
    """A busy bank's non-graph_maintenance work is still claimed."""
    bank = await _make_bank(pool)
    await _insert_op(pool, bank, "graph_maintenance", "processing", claimed_at=datetime.now(UTC))
    retain = await _insert_op(pool, bank, "retain")

    claimed = await _claim(backend)

    assert str(retain) in claimed


@pytest.mark.asyncio
async def test_not_starved_by_newer_pending_work(pool, backend, clean_operations):
    """graph_maintenance still wins the shared slot when it is the oldest row.

    The poller's fairness pass claims with ``shared_limit=1`` and
    graph_maintenance has no reserved-slot floor, so claiming it in a phase
    *after* the generic shared-pool query would let any single pending retain
    starve it indefinitely. As a predicate on that same query it keeps its place
    in the created_at ordering.
    """
    bank = await _make_bank(pool)
    older = await _insert_op(pool, bank, "graph_maintenance", created_at=datetime.now(UTC) - timedelta(minutes=5))
    await _insert_op(pool, bank, "retain")

    claimed = await _claim(backend, shared=1)

    assert claimed == {str(older)}


@pytest.mark.asyncio
async def test_retry_blocked_older_row_does_not_block_a_claimable_one(pool, backend, clean_operations):
    """An older row still in retry backoff must not hold up its bank.

    It cannot be claimed itself, so counting it as "goes first" would stall the
    bank's graph maintenance for the whole backoff window.
    """
    bank = await _make_bank(pool)
    await _insert_op(
        pool,
        bank,
        "graph_maintenance",
        created_at=datetime.now(UTC) - timedelta(minutes=5),
        next_retry_at=datetime.now(UTC) + timedelta(hours=1),
    )
    claimable = await _insert_op(pool, bank, "graph_maintenance")

    claimed = await _claim(backend)

    assert str(claimable) in claimed


def test_guard_survives_the_oracle_sql_rewrite():
    """The shared predicate must still be valid Oracle after db/oracle.py rewrites it.

    Oracle tests need an ORACLE_TEST_DSN this suite does not have, so the dialect
    parity of this guard is otherwise unverified. The rewriter is pure text
    substitution, so assert on its output directly: PG-only spellings must be
    gone, and the ROWNUM row limit must land on the *outer* WHERE rather than the
    subquery's (the rewriter replaces only the first WHERE it sees).
    """
    from hindsight_api.engine.db.ops import graph_maintenance_bank_serialization_sql
    from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

    table = "async_operations"
    rewritten = _rewrite_pg_to_oracle(
        f"""
        SELECT o.operation_id FROM {table} o
        WHERE o.status = 'pending'
          AND (o.next_retry_at IS NULL OR o.next_retry_at <= NOW())
          AND o.operation_id != ALL($1::uuid[])
          AND {graph_maintenance_bank_serialization_sql(table, "o")}
        ORDER BY o.created_at
        LIMIT $2
        FOR UPDATE SKIP LOCKED
        """
    ).query

    assert "NOW()" not in rewritten and "SYSTIMESTAMP" in rewritten
    assert "!= ALL" not in rewritten and "::uuid[]" not in rewritten
    assert "LIMIT" not in rewritten
    assert "FOR UPDATE SKIP LOCKED" in rewritten
    assert "WHERE ROWNUM <= :2 AND o.status = 'pending'" in rewritten
    # The correlated subquery keeps its own unmodified WHERE.
    assert "WHERE gm_peer.bank_id = o.bank_id" in rewritten


@pytest.mark.asyncio
async def test_reserved_pool_is_serialised_too(pool, backend, clean_operations):
    """The guard also applies when graph_maintenance has reserved slots.

    WORKER_SLOT_TYPE_DEFAULTS gives it 0 by default, but an operator can raise
    HINDSIGHT_API_WORKER_GRAPH_MAINTENANCE_RESERVED_SLOTS, which routes claims
    through the reserved-pool query instead of the shared one.
    """
    bank = await _make_bank(pool)
    base = datetime.now(UTC) - timedelta(minutes=10)
    op_ids = [
        await _insert_op(pool, bank, "graph_maintenance", created_at=base + timedelta(seconds=i)) for i in range(3)
    ]

    claimed = await _claim(backend, shared=0, reserved={"graph_maintenance": 5})

    ours = [op for op in op_ids if str(op) in claimed]
    assert len(ours) == 1, f"expected exactly one same-bank claim, got {len(ours)}"


@pytest.mark.asyncio
async def test_reserved_and_shared_phases_do_not_double_claim(pool, backend, clean_operations):
    """A reserved-pool claim blocks the shared pool from taking a second one.

    The two phases run in one transaction, so the row claimed in phase 1 is still
    'pending' when the shared query runs and is excluded from it by operation_id.
    It has to keep blocking through the guard's older-pending branch instead,
    otherwise one cycle hands the same bank two concurrent runs.
    """
    bank = await _make_bank(pool)
    base = datetime.now(UTC) - timedelta(minutes=10)
    op_ids = [
        await _insert_op(pool, bank, "graph_maintenance", created_at=base + timedelta(seconds=i)) for i in range(3)
    ]

    claimed = await _claim(backend, shared=5, reserved={"graph_maintenance": 1})

    ours = [op for op in op_ids if str(op) in claimed]
    assert len(ours) == 1, f"expected exactly one same-bank claim across both phases, got {len(ours)}"
