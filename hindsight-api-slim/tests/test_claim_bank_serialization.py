"""Per-bank serialisation of claims, for graph_maintenance (#3230) and consolidation (#3700).

Every run of either type for a bank is interchangeable — the payload carries
only ``bank_id`` and the job drains that bank's whole backlog — so a second
concurrent run for one bank recomputes the first one's work. graph_maintenance
additionally convoys on the first run's queue-row locks
(``claim_graph_maintenance_batch`` locks ``FOR UPDATE`` with no ``SKIP
LOCKED``); consolidation hands the same memories to the LLM twice.

``claim_tasks`` therefore refuses to claim a row for a bank that already has one
of that type in flight, and takes at most one per bank per batch. Excluding
banks with a *processing* row is not enough on its own: several pending rows and
nothing yet processing is reachable through every recovery path, and one batch
would take them all (#3700 observed exactly that — two consolidations for one
bank claimed at the same microsecond after a restart).

It does that with a predicate on the ordinary claim queries rather than a claim
phase of its own, so graph_maintenance keeps competing by ``created_at`` instead
of dropping below every other operation type — it has no reserved-slot floor,
and the poller's fairness pass claims with ``shared_limit=1``.

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
    schema = f"bankclaim_iso_{worker}"

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
    await pool.execute("DELETE FROM async_operations WHERE bank_id LIKE 'test-bankclaim-%'")


async def _make_bank(pool) -> str:
    """Create a bank with a unique id so concurrent runs can't collide."""
    bank_id = f"test-bankclaim-{uuid.uuid4().hex[:8]}"
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
    payload_extra: dict | None = None,
) -> uuid.UUID:
    """Insert one operation row with a claimable payload."""
    op_id = uuid.uuid4()
    payload = json.dumps({"type": op_type, "bank_id": bank_id, "operation_id": str(op_id), **(payload_extra or {})})
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


async def _claim(
    backend,
    *,
    shared: int = 10,
    reserved: dict[str, int] | None = None,
    consolidation_bank_priority: dict[str, int] | None = None,
) -> set[str]:
    """Run one claim cycle and return the claimed operation ids as strings."""
    async with backend.acquire() as conn:
        async with conn.transaction():
            claimed = await backend.ops.claim_tasks(
                conn,
                _TABLE,
                "test-bankclaim-worker",
                reserved or {},
                shared,
                consolidation_bank_priority=consolidation_bank_priority,
            )
    return {str(row["operation_id"]) for row in claimed.rows}


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
    """A busy bank's other work is still claimed — the guard is per operation type.

    The peer subquery matches on the candidate's own operation_type, so a
    running graph_maintenance holds back only graph_maintenance: retains and
    consolidations for that bank keep flowing.
    """
    bank = await _make_bank(pool)
    await _insert_op(pool, bank, "graph_maintenance", "processing", claimed_at=datetime.now(UTC))
    retain = await _insert_op(pool, bank, "retain")
    consolidation = await _insert_op(pool, bank, "consolidation")

    claimed = await _claim(backend)

    assert str(retain) in claimed
    assert str(consolidation) in claimed


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


def test_fixed_type_call_sites_drop_the_candidate_guard():
    """A query already restricted to one type gets the constant form.

    The consolidation claim queries pin ``operation_type = 'consolidation'`` in
    their own WHERE, so the guard on the candidate's type can only ever be true
    there and the peer match is a constant, not a correlation. Emitting the
    mixed-type form anyway reads as though a consolidation could be held back by
    a graph_maintenance peer, which it never could.
    """
    from hindsight_api.engine.db.ops import bank_serialization_sql

    fixed = bank_serialization_sql("async_operations", "o", "consolidation")
    assert "NOT IN" not in fixed
    assert "bank_peer.operation_type = 'consolidation'" in fixed

    mixed = bank_serialization_sql("async_operations", "o")
    assert "o.operation_type NOT IN ('graph_maintenance', 'consolidation')" in mixed
    assert "bank_peer.operation_type = o.operation_type" in mixed


def test_guard_survives_the_oracle_sql_rewrite():
    """The shared predicate must still be valid Oracle after db/oracle.py rewrites it.

    Oracle tests need an ORACLE_TEST_DSN this suite does not have, so the dialect
    parity of this guard is otherwise unverified. The rewriter is pure text
    substitution, so assert on its output directly: PG-only spellings must be
    gone, and the ROWNUM row limit must land on the *outer* WHERE rather than the
    subquery's (the rewriter replaces only the first WHERE it sees).
    """
    from hindsight_api.engine.db.ops import bank_serialization_sql
    from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

    table = "async_operations"
    rewritten = _rewrite_pg_to_oracle(
        f"""
        SELECT o.operation_id FROM {table} o
        WHERE o.status = 'pending'
          AND (o.next_retry_at IS NULL OR o.next_retry_at <= NOW())
          AND o.operation_id != ALL($1::uuid[])
          AND {bank_serialization_sql(table, "o")}
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
    assert "WHERE bank_peer.bank_id = o.bank_id" in rewritten

    # The consolidation tiers pair the guard with LIKE ANY, whose rewrite is a
    # regex on a *bare* column name — an "o." prefix would survive it as
    # "o.(bank_id LIKE :p0 ...)", so that one column stays unqualified.
    tiered = _rewrite_pg_to_oracle(
        f"""
        SELECT o.operation_id FROM {table} o
        WHERE o.status = 'pending'
          AND {bank_serialization_sql(table, "o", "consolidation")}
          AND bank_id LIKE ANY($1::text[])
        ORDER BY o.created_at
        LIMIT $2
        FOR UPDATE SKIP LOCKED
        """
    ).query

    assert "LIKE ANY" not in tiered
    assert "o./*LIKE_ANY" not in tiered
    assert "WHERE ROWNUM <= :2 AND o.status = 'pending'" in tiered


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


# --- consolidation (#3700) ---------------------------------------------------
#
# Consolidation is claimed by its own path (bank priority tiers), so every
# guarantee above has to be re-checked against those queries.


@pytest.mark.asyncio
async def test_consolidation_not_claimed_while_bank_has_run_in_flight(pool, backend, clean_operations):
    """A pending consolidation is left alone while its bank has one processing."""
    bank = await _make_bank(pool)
    await _insert_op(pool, bank, "consolidation", "processing", claimed_at=datetime.now(UTC), worker_id="other")
    pending = await _insert_op(pool, bank, "consolidation")

    claimed = await _claim(backend)

    assert str(pending) not in claimed
    assert await _status_of(pool, pending) == "pending"


@pytest.mark.asyncio
async def test_consolidation_single_batch_claims_at_most_one_per_bank(pool, backend, clean_operations):
    """One batch takes a single consolidation per bank — the oldest (#3700).

    This is the case the busy-bank exclusion never covered: nothing is
    processing, so no bank is busy, and one query took every pending row. The
    reporter hit it after a restart, where ``_reclaim_own_processing_tasks``
    returns all of a worker's processing rows to pending in one statement.
    """
    bank = await _make_bank(pool)
    base = datetime.now(UTC) - timedelta(minutes=10)
    op_ids = [await _insert_op(pool, bank, "consolidation", created_at=base + timedelta(seconds=i)) for i in range(5)]

    claimed = await _claim(backend)

    ours = [op for op in op_ids if str(op) in claimed]
    assert len(ours) == 1, f"expected exactly one same-bank claim, got {len(ours)}"
    assert ours[0] == op_ids[0], "the oldest pending row should be the one claimed"


@pytest.mark.asyncio
async def test_consolidation_reserved_pool_is_serialised_too(pool, backend, clean_operations):
    """The reserved pool is serialised as well — the shape #3700 was reported in.

    WORKER_SLOT_TYPE_DEFAULTS reserves 2 slots for consolidation, which is why
    the reported worker logged "claimed 2 tasks (2 consolidation)" for one bank.
    """
    bank = await _make_bank(pool)
    base = datetime.now(UTC) - timedelta(minutes=10)
    op_ids = [await _insert_op(pool, bank, "consolidation", created_at=base + timedelta(seconds=i)) for i in range(5)]

    claimed = await _claim(backend, shared=0, reserved={"consolidation": 2})

    ours = [op for op in op_ids if str(op) in claimed]
    assert len(ours) == 1, f"expected exactly one same-bank claim, got {len(ours)}"


@pytest.mark.asyncio
async def test_consolidation_reserved_and_shared_phases_do_not_double_claim(pool, backend, clean_operations):
    """The reserved claim blocks the shared pool from taking a second one.

    Both phases recomputed the busy-bank list from rows that are still 'pending'
    inside the claim transaction, so neither saw the other's claim.
    """
    bank = await _make_bank(pool)
    base = datetime.now(UTC) - timedelta(minutes=10)
    op_ids = [await _insert_op(pool, bank, "consolidation", created_at=base + timedelta(seconds=i)) for i in range(3)]

    claimed = await _claim(backend, shared=5, reserved={"consolidation": 1})

    ours = [op for op in op_ids if str(op) in claimed]
    assert len(ours) == 1, f"expected exactly one same-bank claim across both phases, got {len(ours)}"


@pytest.mark.asyncio
async def test_consolidation_priority_tiers_are_serialised(pool, backend, clean_operations):
    """Both tiered claim queries carry the guard, and tiers cannot stack claims.

    ``consolidation_bank_priority`` routes claims through the LIKE / NOT LIKE
    variants instead of the plain one, and runs them once per priority level.
    """
    high_bank = await _make_bank(pool)
    low_bank = await _make_bank(pool)
    base = datetime.now(UTC) - timedelta(minutes=10)
    high_ops = [
        await _insert_op(pool, high_bank, "consolidation", created_at=base + timedelta(seconds=i)) for i in range(3)
    ]
    low_ops = [
        await _insert_op(pool, low_bank, "consolidation", created_at=base + timedelta(seconds=i)) for i in range(3)
    ]

    claimed = await _claim(backend, consolidation_bank_priority={f"{high_bank}*": 10, "*": 1})

    assert len([op for op in high_ops if str(op) in claimed]) == 1
    assert len([op for op in low_ops if str(op) in claimed]) == 1


@pytest.mark.asyncio
async def test_consolidation_other_banks_keep_their_parallelism(pool, backend, clean_operations):
    """The guard is per bank: a second bank's consolidation is claimed in the same batch."""
    busy_bank = await _make_bank(pool)
    idle_bank = await _make_bank(pool)
    await _insert_op(pool, busy_bank, "consolidation", "processing", claimed_at=datetime.now(UTC))
    blocked = await _insert_op(pool, busy_bank, "consolidation")
    claimable = await _insert_op(pool, idle_bank, "consolidation")

    claimed = await _claim(backend)

    assert str(claimable) in claimed
    assert str(blocked) not in claimed


@pytest.mark.asyncio
async def test_consolidation_retry_blocked_older_row_does_not_block_a_claimable_one(pool, backend, clean_operations):
    """An older consolidation still in retry backoff must not hold up its bank."""
    bank = await _make_bank(pool)
    await _insert_op(
        pool,
        bank,
        "consolidation",
        created_at=datetime.now(UTC) - timedelta(minutes=5),
        next_retry_at=datetime.now(UTC) + timedelta(hours=1),
    )
    claimable = await _insert_op(pool, bank, "consolidation")

    claimed = await _claim(backend)

    assert str(claimable) in claimed


@pytest.mark.asyncio
async def test_scoped_consolidation_queues_behind_an_unscoped_one(pool, backend, clean_operations):
    """A scoped consolidation is deferred, not dropped, while the bank is busy.

    Submit-time dedup exempts scoped runs (``observation_scopes`` covers only a
    tag subset, so an unscoped pending op must not swallow it), but claim-time
    serialisation needs no such carve-out: nothing is discarded here, the scoped
    run simply waits for the bank and is claimed on a later cycle — with a fresh
    watermark, which is the whole reason consolidation is not deduped at submit.
    """
    bank = await _make_bank(pool)
    running = await _insert_op(pool, bank, "consolidation", "processing", claimed_at=datetime.now(UTC))
    scoped = await _insert_op(pool, bank, "consolidation", payload_extra={"observation_scopes": [["project"]]})

    assert str(scoped) not in await _claim(backend)

    await pool.execute(f"UPDATE {_TABLE} SET status = 'completed' WHERE operation_id = $1", running)

    assert str(scoped) in await _claim(backend)
