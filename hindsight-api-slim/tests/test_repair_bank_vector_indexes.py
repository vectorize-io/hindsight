"""Per-bank vector index reconcile against the size threshold (issues #2645, #3485).

A (bank, fact_type) earns a partial vector index by size. Below the threshold
the planner answers the same ANN query exactly, and faster, from the
``(bank_id, fact_type)`` B-tree plus a top-N sort; the index it would otherwise
carry is paid for by every *other* bank, because indexes on the shared
``memory_units`` table are locked and planned against by every query on it. Three
per bank exhausts the lock table at a few thousand banks (#3485).

What is proven here:

* the policy — build at or above the threshold, drop below the hysteresis floor,
  leave partitions between the two alone;
* the recovery path — indexes orphaned by a deleted bank are collected by the
  admin command, the only path that can still see them;
* the escape hatch — ``repair-bank`` is re-runnable, rebuilds invalid coverage
  (an index whose shape drifted counts as missing, unlike a name-only check),
  is a no-op on non-per-bank backends, and validates its target flags.

Everything asserted is deterministic (index presence/shape via the catalog) —
no LLM is needed, so memory_units are inserted directly. Tests lower the
threshold rather than inserting ten thousand rows.
"""

import uuid

import pytest
from asyncpg.exceptions import DeadlockDetectedError

from hindsight_api import RequestContext
from hindsight_api.admin import cli
from hindsight_api.admin.cli import _run_repair_bank
from hindsight_api.engine import vector_index_health
from hindsight_api.engine.db_utils import acquire_with_retry, retry_with_backoff
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain.bank_utils import _BANK_INDEX_FACT_TYPES, _bank_index_name, _vector_index_clause
from hindsight_api.engine.transfer import export_bank
from hindsight_api.engine.vector_index_health import (
    drop_orphaned_bank_indexes,
    plan_bank_vector_indexes,
    reconcile_bank_vector_indexes,
)

# Serialized onto one xdist worker. Every test here issues CREATE/DROP INDEX
# CONCURRENTLY against the single shared public.memory_units, and concurrent
# index DDL on one relation deadlocks by design — CONCURRENTLY holds
# ShareUpdateExclusive while waiting out every session whose snapshot could
# still see the index, including other sessions' queued index DDL. Eight workers
# doing that to one table outlasts any retry budget (the same storm as
# f9cef24cb). Advisory locks are banned, so the isolation has to come from the
# scheduler.
pytestmark = pytest.mark.xdist_group("vector_index_reconcile")

_TEST_SCHEMA = "public"

# The shipped default threshold is 0 — every bank holding rows is indexed. These
# tests exercise the *raised*-threshold behaviour a large deployment configures,
# with small numbers so crossing it takes a handful of inserts rather than ten
# thousand. The drop floor is half the build floor, which puts _BETWEEN inside
# the hysteresis gap.
_BUILD_AT = 4
_DROP_BELOW = 2
_BETWEEN = 3


@pytest.fixture
def low_threshold(monkeypatch):
    """Shrink the size threshold so tests need 4 rows, not 10,000.

    Patched on ``vector_index_health``'s namespace because it imports both
    helpers by name; patching the config would not reach the already-bound
    references. ``raising=True`` (the default) is deliberate — if the reconcile
    stops using one of these, this fixture must fail loudly rather than silently
    leave the production 10,000-row threshold in place, which would turn every
    test below into a vacuous pass.
    """
    monkeypatch.setattr(vector_index_health, "qualifies_for_per_bank_index", lambda rows: rows >= _BUILD_AT)
    monkeypatch.setattr(
        vector_index_health, "should_keep_per_bank_index", lambda rows: rows > 0 and rows >= _DROP_BELOW
    )


@pytest.fixture
def default_threshold(monkeypatch):
    """Restore the *shipped* default (0 = no minimum) for this test.

    conftest raises the threshold out of reach for the whole suite so thousands
    of throwaway banks don't each queue an index build; a test asserting the
    default behaviour has to put it back.
    """
    monkeypatch.setattr(vector_index_health, "qualifies_for_per_bank_index", lambda rows: rows > 0)
    monkeypatch.setattr(vector_index_health, "should_keep_per_bank_index", lambda rows: rows > 0)


async def _bank_internal_id(conn, bank_id: str) -> str:
    row = await conn.fetchrow("SELECT internal_id FROM banks WHERE bank_id = $1", bank_id)
    assert row is not None, f"bank {bank_id} not found"
    return str(row["internal_id"])


async def _index_exists(conn, idx_name: str) -> bool:
    return bool(
        await conn.fetchval(
            "SELECT 1 FROM pg_indexes WHERE schemaname = $1 AND indexname = $2",
            _TEST_SCHEMA,
            idx_name,
        )
    )


async def _index_is_partial_vector(conn, idx_name: str) -> bool:
    """True only if the index carries our per-(bank, fact_type) partial predicate."""
    indexdef = await conn.fetchval(
        "SELECT pg_get_indexdef(c.oid) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = $1 AND c.relname = $2",
        _TEST_SCHEMA,
        idx_name,
    )
    return bool(indexdef) and "WHERE ((fact_type = " in indexdef


async def _expected_index_names(conn, bank_id: str) -> list[str]:
    internal_id = await _bank_internal_id(conn, bank_id)
    return [_bank_index_name(ft, internal_id) for ft in _BANK_INDEX_FACT_TYPES]


# A whole-bank export/import round-trip only carries facts that have an
# embedding, so seeds destined for one need a real vector. 384 dims matches the
# default local embedding model the schema is built for.
_EMBEDDING = "[" + ",".join(["0.01"] * 384) + "]"


async def _insert_memory(conn, bank_id: str, fact_type: str, text: str) -> None:
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, event_date, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5::vector, NOW(), NOW(), NOW())
        """,
        uuid.uuid4(),
        bank_id,
        text,
        fact_type,
        _EMBEDDING,
    )


async def _seed_bank(memory: MemoryEngine, request_context: RequestContext, rows_per_fact_type: int) -> str:
    """Create a bank and give every fact_type ``rows_per_fact_type`` memories.

    Bank creation issues no index DDL under the size policy, so whatever indexes
    the bank ends up with are the ones a reconcile decided it earned.
    """
    bank_id = f"test-repair-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        for ft in _BANK_INDEX_FACT_TYPES:
            for n in range(rows_per_fact_type):
                await _insert_memory(conn, bank_id, ft, f"seed {ft} fact {n}")
    return bank_id


async def _reconcile(conn, bank_id: str, *, dry_run: bool = False):
    """Reconcile one bank in the test schema."""
    index_clause = _vector_index_clause()
    assert index_clause is not None  # per-bank-index backend
    return await reconcile_bank_vector_indexes(conn, _TEST_SCHEMA, bank_id, index_clause, dry_run=dry_run)


async def _build_indexes_for(conn, bank_id: str) -> list[str]:
    """Give ``bank_id`` its three partial indexes directly, bypassing the policy.

    Used to set up drop-side tests: the index has to exist before the reconcile
    can be asked to remove it.
    """
    index_clause = _vector_index_clause()
    assert index_clause is not None
    names = await _expected_index_names(conn, bank_id)
    literal = await conn.fetchval("SELECT quote_literal($1::text)", bank_id)
    for ft, name in zip(_BANK_INDEX_FACT_TYPES, names, strict=True):
        await retry_with_backoff(
            lambda name=name, ft=ft: conn.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON {_TEST_SCHEMA}.memory_units "
                f"{index_clause} WHERE fact_type = '{ft}' AND bank_id = {literal}"
            )
        )
    return names


async def _drop_bank_indexes(conn, bank_id: str) -> list[str]:
    """Drop every per-(bank, fact_type) index for ``bank_id``.

    CONCURRENTLY so the drop never takes ACCESS EXCLUSIVE on the shared
    ``memory_units`` table: the suite runs 8 xdist workers against one pg0
    database, and a blocking DDL here stalls unrelated workers' DML. Retried
    because CONCURRENTLY still takes ShareUpdateExclusive, which conflicts with
    another worker's concurrent index DDL on the same table.
    """
    names = await _expected_index_names(conn, bank_id)
    for name in names:
        await retry_with_backoff(
            lambda name=name: conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_TEST_SCHEMA}.{name}")
        )
    return names


class _DeadlockOnceOnCreate:
    """Wrap a real asyncpg connection and raise a single deadlock on the first
    ``CREATE INDEX CONCURRENTLY``, delegating everything else.

    Simulates the transient deadlock that CI's 8 xdist workers hit when a
    concurrent build on the shared ``memory_units`` table is picked as the
    victim, so the retry path can be exercised deterministically.
    """

    def __init__(self, real):
        self._real = real
        self.create_calls = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    async def execute(self, query, *args, **kwargs):
        if "CREATE INDEX CONCURRENTLY" in query:
            self.create_calls += 1
            if self.create_calls == 1:
                raise DeadlockDetectedError("deadlock detected")
        return await self._real.execute(query, *args, **kwargs)


class TestSizeThresholdPolicy:
    """Build above the threshold, drop below the floor, leave the gap alone."""

    @pytest.mark.asyncio
    async def test_bank_at_the_threshold_gets_its_indexes(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _expected_index_names(conn, bank_id)
                for name in names:
                    assert not await _index_exists(conn, name), "bank creation must not build indexes"

                result = await _reconcile(conn, bank_id)

                assert result.failed == 0, result.failed_indexes
                assert result.created == len(_BANK_INDEX_FACT_TYPES)
                for name in names:
                    assert await _index_is_partial_vector(conn, name), f"{name} should be built as a partial index"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_bank_below_the_threshold_gets_nothing(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """The common shape at scale: thousands of small banks, zero indexes between them."""
        bank_id = await _seed_bank(memory, request_context, _DROP_BELOW - 1)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                result = await _reconcile(conn, bank_id)

                assert result.created == 0
                assert result.failed == 0, result.failed_indexes
                for name in await _expected_index_names(conn, bank_id):
                    assert not await _index_exists(conn, name)
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_shrunk_bank_loses_its_indexes(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """Consolidation can prune a bank back under the floor; the index must go."""
        bank_id = await _seed_bank(memory, request_context, _DROP_BELOW - 1)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _build_indexes_for(conn, bank_id)
                for name in names:
                    assert await _index_exists(conn, name)

                result = await _reconcile(conn, bank_id)

                assert result.dropped == len(names)
                for name in names:
                    assert not await _index_exists(conn, name), f"{name} should have been dropped"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_hysteresis_gap_leaves_an_existing_index_alone(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """Between the drop floor and the build threshold, nothing happens either way.

        Without the gap, a bank hovering at a single boundary would rebuild and
        drop the same ANN index on alternating sweeps.
        """
        bank_id = await _seed_bank(memory, request_context, _BETWEEN)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _build_indexes_for(conn, bank_id)

                result = await _reconcile(conn, bank_id)

                assert result.created == 0, "already inside the gap — nothing to build"
                assert result.dropped == 0, "inside the gap the existing index is kept"
                for name in names:
                    assert await _index_exists(conn, name)
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_hysteresis_gap_does_not_build_a_missing_index(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """The gap keeps what exists; it does not entitle a partition to a new index."""
        bank_id = await _seed_bank(memory, request_context, _BETWEEN)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                result = await _reconcile(conn, bank_id)

                assert result.created == 0
                for name in await _expected_index_names(conn, bank_id):
                    assert not await _index_exists(conn, name)
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)


class TestRecoveryPath:
    """Shedding indexes is what rescues a deployment that hit the lock-table wall."""

    @pytest.mark.asyncio
    async def test_index_orphaned_by_a_deleted_bank_is_dropped(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """An index whose bank row is gone is unreachable from every bank-scoped path.

        Planning starts from the bank's internal_id, so a deleted bank yields an
        empty plan and its leftovers would live forever. Normally there are none —
        delete_bank drops them while it still knows their names — but an instance
        at the #3485 wall could not run delete_bank at all, which is exactly the
        state this has to clean up.
        """
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            names = await _build_indexes_for(conn, bank_id)

        # Remove the bank the way a wedged operator would: rows and profile gone,
        # indexes left behind.
        async with acquire_with_retry(backend) as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)
            for name in names:
                assert await _index_exists(conn, name), "setup: the index should outlive the bank here"

            assert (await plan_bank_vector_indexes(conn, _TEST_SCHEMA, bank_id)).is_empty, (
                "a bank-scoped plan cannot name an orphan — that is why the sweep below exists"
            )

            dropped = await drop_orphaned_bank_indexes(conn, _TEST_SCHEMA)

            assert set(names) <= set(dropped)
            for name in names:
                assert not await _index_exists(conn, name), f"orphan {name} should be dropped"

    @pytest.mark.asyncio
    async def test_orphan_sweep_leaves_live_banks_alone(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """The orphan sweep is catalog-wide, so it must key off the live bank set.

        It is the one path here that is not bank-scoped; a name-suffix match that
        did not check `banks` would drop every index in the schema.
        """
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _build_indexes_for(conn, bank_id)

                dropped = await drop_orphaned_bank_indexes(conn, _TEST_SCHEMA)

                assert not (set(names) & set(dropped)), "a live bank's indexes must not be collected"
                for name in names:
                    assert await _index_exists(conn, name)
        finally:
            async with acquire_with_retry(backend) as conn:
                await _drop_bank_indexes(conn, bank_id)
            await memory.delete_bank(bank_id, request_context=request_context)


class TestPlan:
    """The plan is the cheap pre-check the write path runs on every insert."""

    @pytest.mark.asyncio
    async def test_plan_is_empty_when_coverage_already_matches(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """An empty plan is what keeps every write from queueing a worker task."""
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                await _build_indexes_for(conn, bank_id)

                plan = await plan_bank_vector_indexes(conn, _TEST_SCHEMA, bank_id)

                assert plan.is_empty
                assert plan.already_present == len(_BANK_INDEX_FACT_TYPES)
        finally:
            async with acquire_with_retry(backend) as conn:
                await _drop_bank_indexes(conn, bank_id)
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_plan_names_the_fact_types_that_need_building(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                plan = await plan_bank_vector_indexes(conn, _TEST_SCHEMA, bank_id)

                assert set(plan.to_build) == set(_BANK_INDEX_FACT_TYPES)
                assert plan.to_drop == []
                assert not plan.is_empty
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_plan_for_a_missing_bank_is_empty(self, memory: MemoryEngine):
        """A bank deleted between the write and the operation must not error."""
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            plan = await plan_bank_vector_indexes(conn, _TEST_SCHEMA, f"never-existed-{uuid.uuid4().hex[:8]}")

        assert plan.is_empty
        assert plan.to_build == []


class TestDefaultThresholdIsBackwardsCompatible:
    """Shipped default: 0 rows — every bank holding memories is indexed.

    The threshold is opt-in, so an existing deployment that upgrades keeps the
    coverage it had. These use ``default_threshold`` rather than
    ``low_threshold``: the suite-wide setting in conftest puts the threshold out
    of reach, so asserting the shipped default means restoring it explicitly.
    """

    @pytest.mark.asyncio
    async def test_a_single_row_earns_an_index_at_the_default(
        self, memory: MemoryEngine, request_context: RequestContext, default_threshold
    ):
        bank_id = await _seed_bank(memory, request_context, 1)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                result = await _reconcile(conn, bank_id)

                assert result.failed == 0, result.failed_indexes
                assert result.created == len(_BANK_INDEX_FACT_TYPES)
                for name in await _expected_index_names(conn, bank_id):
                    assert await _index_is_partial_vector(conn, name)
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_an_empty_bank_gets_nothing(
        self, memory: MemoryEngine, request_context: RequestContext, default_threshold
    ):
        """Zero rows is not "at least zero rows".

        An index over no rows serves nothing, and bank creation no longer builds
        one — so a bank that exists but has never been written to stays clean.
        """
        bank_id = await _seed_bank(memory, request_context, 0)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                plan = await plan_bank_vector_indexes(conn, _TEST_SCHEMA, bank_id)
                result = await _reconcile(conn, bank_id)

                assert plan.is_empty
                assert result.created == 0
                for name in await _expected_index_names(conn, bank_id):
                    assert not await _index_exists(conn, name)
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)


class TestReconcileMechanics:
    @pytest.mark.asyncio
    @pytest.mark.memory_backend_incompatible
    async def test_invalid_shape_index_is_rebuilt(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """A name-colliding index that lacks the partial predicate is unhealthy → rebuilt.

        The differentiator over a name-only existence check, which would treat
        the collision — or a stale INVALID leftover — as 'already present' and
        never repair it.
        """
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _expected_index_names(conn, bank_id)
                # Recreate the FIRST expected name with the WRONG definition: a
                # plain btree with no partial predicate. Name matches, shape does
                # not. CONCURRENTLY so the decoy never takes ACCESS EXCLUSIVE.
                bogus = names[0]
                await conn.execute(f"CREATE INDEX CONCURRENTLY {bogus} ON memory_units (bank_id)")
                assert not await _index_is_partial_vector(conn, bogus)

                result = await _reconcile(conn, bank_id)
                assert result.failed == 0, result.failed_indexes

                for name in names:
                    assert await _index_is_partial_vector(conn, name), f"{name} should now be the partial index"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_transient_deadlock_is_retried_not_failed(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """A deadlock during the CONCURRENTLY build is retried, not a permanent failure.

        The exact CI flake: 8 xdist workers share one memory_units table, so a
        concurrent build gets picked as the deadlock victim. The reconcile must
        converge rather than leave result.failed > 0.
        """
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _expected_index_names(conn, bank_id)
                flaky = _DeadlockOnceOnCreate(conn)

                result = await _reconcile(flaky, bank_id)

                assert flaky.create_calls >= 2, "expected a retry after the injected deadlock"
                assert result.failed == 0, result.failed_indexes
                assert result.created == len(_BANK_INDEX_FACT_TYPES)
                for name in names:
                    assert await _index_exists(conn, name), f"{name} should be built after the retry"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_dry_run_changes_nothing(self, memory: MemoryEngine, request_context: RequestContext, low_threshold):
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _expected_index_names(conn, bank_id)

                result = await _reconcile(conn, bank_id, dry_run=True)

                assert result.created == 0
                assert result.skipped == len(_BANK_INDEX_FACT_TYPES)
                for name in names:
                    assert not await _index_exists(conn, name), f"{name} must NOT exist after a dry run"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_rerun_is_idempotent(self, memory: MemoryEngine, request_context: RequestContext, low_threshold):
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _expected_index_names(conn, bank_id)

                first = await _reconcile(conn, bank_id)
                assert first.created == len(_BANK_INDEX_FACT_TYPES)

                second = await _reconcile(conn, bank_id)
                assert second.created == 0
                assert second.dropped == 0
                assert second.failed == 0
                assert second.already_present == len(_BANK_INDEX_FACT_TYPES)

                for name in names:
                    count = await conn.fetchval(
                        "SELECT count(*) FROM pg_indexes WHERE schemaname = $1 AND indexname = $2",
                        _TEST_SCHEMA,
                        name,
                    )
                    assert count == 1, f"{name} should exist exactly once"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)


class TestRepairBankCommand:
    @pytest.mark.asyncio
    async def test_command_builds_what_qualifies(
        self, memory: MemoryEngine, request_context: RequestContext, pg0_db_url: str, low_threshold
    ):
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            results = await _run_repair_bank(
                pg0_db_url, base_schema=_TEST_SCHEMA, schema=_TEST_SCHEMA, bank_id=bank_id, dry_run=False
            )

            assert len(results) == 1
            result = results[0]
            assert result.failed == 0, result.failed_indexes
            assert result.created == len(_BANK_INDEX_FACT_TYPES)

            async with acquire_with_retry(backend) as conn:
                for name in await _expected_index_names(conn, bank_id):
                    assert await _index_is_partial_vector(conn, name)
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    def test_requires_exactly_one_target(self):
        """Neither / both of --bank and --all is a usage error (exit 2)."""
        from typer.testing import CliRunner

        runner = CliRunner()
        neither = runner.invoke(cli.app, ["repair-bank"])
        assert neither.exit_code == 2, neither.output
        both = runner.invoke(cli.app, ["repair-bank", "--bank", "b1", "--all"])
        assert both.exit_code == 2, both.output

    @pytest.mark.asyncio
    async def test_backend_without_per_bank_indexes_is_noop(
        self, memory: MemoryEngine, request_context: RequestContext, monkeypatch, low_threshold
    ):
        """AlloyDB ScaNN / Oracle keep a single global index — nothing to reconcile."""
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            monkeypatch.setattr(cli, "_vector_index_clause", lambda: None)

            from typer.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(cli.app, ["repair-bank", "--all"])
            assert result.exit_code == 0, result.output
            assert "does not use per-bank vector indexes" in result.output

            async with acquire_with_retry(backend) as conn:
                for name in await _expected_index_names(conn, bank_id):
                    assert not await _index_exists(conn, name), f"{name} must NOT be built for a no-op backend"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)


class TestRestoredBankCoverage:
    """The #2645 guarantee, re-established through the size policy.

    #2645 was: a bank that arrives already populated — logical restore,
    cross-version upgrade, backend switch — bypassed the fresh-INSERT gate that
    created its indexes, because ``get_or_create_bank_profile`` took the SELECT
    branch for a bank row that already existed. It was then left permanently
    without coverage, silently falling back to a global index plus post-filter.

    That whole class of bug is now structurally impossible: nothing creates
    indexes at bank-creation time, so there is no gate left to bypass.
    Entitlement is recomputed from live row counts whenever a write queues a
    reconcile, and how the rows got there is not something it can observe.
    """

    @pytest.mark.asyncio
    async def test_import_builds_no_indexes_inline(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """Import must issue no index DDL: it runs in a transaction on the shared table.

        The bank is empty at the point the old code built its indexes (facts are
        replayed afterwards), so the build was both useless and a ShareLock on
        ``memory_units`` taken inside the import transaction.
        """
        bank_id = f"test-import-{uuid.uuid4().hex[:8]}"
        backend = await memory._get_backend()
        try:
            await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
            async with acquire_with_retry(backend) as conn:
                archive = await export_bank(conn, bank_id)

            await memory.delete_bank(bank_id, request_context=request_context)
            result = await memory.import_bank_async(archive, request_context)
            assert result.bank_id == bank_id

            async with acquire_with_retry(backend) as conn:
                for name in await _expected_index_names(conn, bank_id):
                    assert not await _index_exists(conn, name), f"import must not build {name} inline"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_rows_arriving_after_bank_creation_still_get_coverage(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """The #2645 shape: the bank row exists first, the rows land afterwards.

        Every populated-bank-without-coverage report reduces to this ordering.
        The reconcile counts rows and cannot tell the difference, so a restored
        bank, an upgraded one and an ordinarily-grown one all converge the same
        way.
        """
        bank_id = f"test-restore-{uuid.uuid4().hex[:8]}"
        backend = await memory._get_backend()
        try:
            # Bank row first — this is the SELECT branch that #2645 fell through.
            await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
            async with acquire_with_retry(backend) as conn:
                for name in await _expected_index_names(conn, bank_id):
                    assert not await _index_exists(conn, name)

                # Rows afterwards, as a restore or an upgrade delivers them.
                for ft in _BANK_INDEX_FACT_TYPES:
                    for n in range(_BUILD_AT):
                        await _insert_memory(conn, bank_id, ft, f"restored {ft} fact {n}")

                reconciled = await _reconcile(conn, bank_id)

                assert reconciled.failed == 0, reconciled.failed_indexes
                assert reconciled.created == len(_BANK_INDEX_FACT_TYPES)
                for name in await _expected_index_names(conn, bank_id):
                    assert await _index_is_partial_vector(conn, name), (
                        f"{name} should exist after the reconcile (restore coverage, #2645)"
                    )
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)


class TestScopeSafety:
    """A reconcile must never touch a bank other than the one it was given.

    Now structural — planning starts from one bank's ``internal_id`` and can only
    name that bank's three indexes — but it was not always: an earlier cut
    derived the drop set by subtracting one bank's partitions from everything the
    schema owned, which silently covered every *other* bank's indexes. Keeping the
    guard means a future refactor back toward a set-difference gets caught.
    """

    @pytest.mark.asyncio
    async def test_reconcile_leaves_other_banks_indexes_alone(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        target = await _seed_bank(memory, request_context, _DROP_BELOW - 1)
        bystander = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                target_names = await _build_indexes_for(conn, target)
                bystander_names = await _build_indexes_for(conn, bystander)

                result = await _reconcile(conn, target)

                assert result.dropped == len(target_names), "the targeted bank should be reconciled"
                for name in target_names:
                    assert not await _index_exists(conn, name), f"{name} was below the floor and should be dropped"
                for name in bystander_names:
                    assert await _index_exists(conn, name), (
                        f"{name} belongs to another bank and must survive a scoped repair"
                    )
        finally:
            async with acquire_with_retry(backend) as conn:
                await _drop_bank_indexes(conn, bystander)
            await memory.delete_bank(target, request_context=request_context)
            await memory.delete_bank(bystander, request_context=request_context)


class TestMaintenanceSubmission:
    """The write path queues coverage work, and only when there is work.

    Replaces a periodic sweep: only a write moves a bank across the threshold, so
    the writer already knows everything a sweep could have discovered. What a
    sweep bought was a bound on how often it ran; here that bound has to come
    from the pre-check being honest, or every insert queues an empty operation.
    """

    @pytest.mark.asyncio
    async def test_submit_is_a_no_op_when_coverage_already_matches(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                await _build_indexes_for(conn, bank_id)

            result = await memory.submit_async_vector_index_maintenance(
                bank_id=bank_id, request_context=request_context
            )

            assert result["no_work"] is True
            assert result["operation_id"] is None
        finally:
            async with acquire_with_retry(backend) as conn:
                await _drop_bank_indexes(conn, bank_id)
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_submit_queues_work_when_an_index_is_missing(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        try:
            result = await memory.submit_async_vector_index_maintenance(
                bank_id=bank_id, request_context=request_context
            )

            assert result.get("no_work") is not True
            assert result["operation_id"] is not None
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_an_empty_bank_never_queues_anything(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """The common case at scale: banks that exist but hold nothing yet.

        Every bank creation would otherwise pay for an async_operations row that
        has nothing to do.
        """
        bank_id = await _seed_bank(memory, request_context, 0)
        try:
            result = await memory.submit_async_vector_index_maintenance(
                bank_id=bank_id, request_context=request_context
            )

            assert result["no_work"] is True
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_handler_builds_the_indexes(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """End to end: the queued operation is what actually creates the coverage."""
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            await memory._handle_vector_index_maintenance({"bank_id": bank_id})

            async with acquire_with_retry(backend) as conn:
                for name in await _expected_index_names(conn, bank_id):
                    assert await _index_is_partial_vector(conn, name), f"{name} should be built by the handler"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_backend_without_per_bank_indexes_never_queues(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold, monkeypatch
    ):
        """ScaNN keeps one global index; there is no per-bank coverage to maintain."""
        from hindsight_api.engine.retain import bank_utils

        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        try:
            monkeypatch.setattr(bank_utils, "_vector_index_clause", lambda: None)

            result = await memory.submit_async_vector_index_maintenance(
                bank_id=bank_id, request_context=request_context
            )

            assert result["no_work"] is True
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)


class TestDeletionDropsCoverage:
    """Losing rows has to take the indexes with it, not just gaining them.

    Every drop test above raises the threshold and leaves the rows alone, which
    is why a whole class of bug survived: the drop side was dead at the shipped
    default, and no delete path queued a reconcile at all. These come at it from
    the other direction — the rows go away and the threshold does not move.
    """

    @pytest.mark.asyncio
    async def test_emptied_bank_drops_its_indexes_at_the_default_threshold(
        self, memory: MemoryEngine, request_context: RequestContext, default_threshold
    ):
        """Regression: at threshold 0 the drop check read `0 < 0` and never fired.

        An emptied bank kept three ANN indexes over nothing forever, because
        nothing writes to an emptied bank — the exact accumulation the threshold
        exists to prevent, in the configuration almost everyone runs.
        """
        bank_id = await _seed_bank(memory, request_context, 3)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _build_indexes_for(conn, bank_id)
                for name in names:
                    assert await _index_exists(conn, name), "setup: the bank should start with coverage"

                await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)

                plan = await plan_bank_vector_indexes(conn, _TEST_SCHEMA, bank_id)
                assert set(plan.to_drop) == set(names), "an emptied partition must be planned for dropping"

                result = await _reconcile(conn, bank_id)

                assert result.dropped == len(names)
                for name in names:
                    assert not await _index_exists(conn, name), f"{name} covers no rows and should be gone"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_partially_emptied_bank_keeps_only_the_covered_fact_types(
        self, memory: MemoryEngine, request_context: RequestContext, default_threshold
    ):
        """Coverage is per (bank, fact_type), so a partial delete is a partial drop."""
        bank_id = await _seed_bank(memory, request_context, 3)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                await _build_indexes_for(conn, bank_id)
                internal_id = await _bank_internal_id(conn, bank_id)
                emptied, kept = "world", "experience"

                await conn.execute("DELETE FROM memory_units WHERE bank_id = $1 AND fact_type = $2", bank_id, emptied)
                await _reconcile(conn, bank_id)

                assert not await _index_exists(conn, _bank_index_name(emptied, internal_id))
                assert await _index_exists(conn, _bank_index_name(kept, internal_id))
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_clearing_a_bank_drops_its_indexes(
        self, memory: MemoryEngine, request_context: RequestContext, default_threshold
    ):
        """`DELETE /memories` keeps the bank, so nothing else can drop its indexes.

        Regression for the worst case of the lot. The full-delete path drops a
        bank's indexes by name while it still knows the internal_id; the
        clear-memories path (``delete_bank_profile=False``) deliberately does
        not, and queued no reconcile either — so clearing a bank left three
        indexes over zero rows permanently.
        """
        bank_id = await _seed_bank(memory, request_context, 3)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _build_indexes_for(conn, bank_id)

            await memory.delete_bank(bank_id, request_context=request_context, delete_bank_profile=False)

            async with acquire_with_retry(backend) as conn:
                assert await conn.fetchval("SELECT count(*) FROM memory_units WHERE bank_id = $1", bank_id) == 0, (
                    "setup: clearing should remove every row"
                )
                assert await conn.fetchval("SELECT 1 FROM banks WHERE bank_id = $1", bank_id), (
                    "setup: the bank itself must survive — that is what makes this path distinct"
                )
                for name in names:
                    assert not await _index_exists(conn, name), (
                        f"{name} outlived every row it indexed; nothing else would ever collect it"
                    )
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)


class TestHandlerConnectionSource:
    """The maintenance job must reach the database the engine is attached to.

    It needs its own connection — CREATE/DROP INDEX CONCURRENTLY cannot run on a
    pooled one inside a transaction — and the tempting way to get one is to read
    HINDSIGHT_API_DATABASE_URL back out of config. That is wrong whenever the
    engine was handed a DSN directly instead of reading the env var: embedders do
    that, and so does this very test suite, which resolves pg0 in a fixture. CI
    caught it where local runs could not, because the local harness exported the
    env var and CI does not — every reconcile there died on
    `relation "public.banks" does not exist`.
    """

    @pytest.mark.asyncio
    async def test_handler_ignores_a_wrong_database_url_in_config(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold, monkeypatch
    ):
        from types import SimpleNamespace

        import hindsight_api.engine.memory_engine as engine_mod

        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            # Config points somewhere that does not exist; the engine's own pool
            # is fine. Reading config would raise or reconcile the wrong database.
            monkeypatch.setattr(
                engine_mod,
                "get_config",
                lambda: SimpleNamespace(
                    migration_database_url=None,
                    database_url="postgresql://nobody@127.0.0.1:1/does-not-exist",
                ),
            )

            await memory._handle_vector_index_maintenance({"bank_id": bank_id})

            async with acquire_with_retry(backend) as conn:
                for name in await _expected_index_names(conn, bank_id):
                    assert await _index_is_partial_vector(conn, name), (
                        f"{name} was not built — the handler used config.database_url instead of the engine's own DSN"
                    )
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_backend_exposes_the_dsn_it_was_opened_with(self, memory: MemoryEngine):
        """The property the handler depends on; without it the fallback is silent."""
        backend = await memory._get_backend()

        assert getattr(backend, "dsn", None), "the pool's DSN must be reachable for out-of-pool DDL"
