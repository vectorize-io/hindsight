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
* the recovery path — indexes orphaned by a deleted bank are dropped, which is
  how a deployment that hit the wall sheds them;
* the budgets — one pass is bounded and reports a backlog, and the bound is
  pass-global rather than per schema;
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
from hindsight_api.engine.schema import fq_routine
from hindsight_api.engine.transfer import export_bank
from hindsight_api.engine.vector_index_health import (
    VectorIndexBudget,
    _reconcile_schema,
    discover_partitions,
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

# Row counts used in place of the production 10_000, so a test can cross the
# threshold with a handful of inserts. The drop floor is half the build floor,
# which puts _BETWEEN inside the hysteresis gap.
_BUILD_AT = 4
_DROP_BELOW = 2
_BETWEEN = 3


@pytest.fixture
def low_threshold(monkeypatch):
    """Shrink the size threshold so tests need 4 rows, not 10,000.

    Patched on ``vector_index_health``'s namespace because it imports both
    helpers by name; patching the config would not reach the already-bound
    references.
    """
    monkeypatch.setattr(vector_index_health, "per_bank_index_min_rows", lambda: _BUILD_AT)
    monkeypatch.setattr(vector_index_health, "per_bank_index_drop_rows", lambda: _DROP_BELOW)


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


async def _reconcile(conn, bank_id: str | None, *, dry_run: bool = False, budget: VectorIndexBudget | None = None):
    """Run one reconcile pass over the test schema, scoped to ``bank_id``."""
    index_clause = _vector_index_clause()
    assert index_clause is not None  # per-bank-index backend
    partitions = await discover_partitions(conn, fq_routine("banks_needing_vector_index"), bank_id=bank_id)
    return await _reconcile_schema(
        conn,
        _TEST_SCHEMA,
        index_clause,
        partitions.get(_TEST_SCHEMA, []),
        dry_run=dry_run,
        budget=budget or VectorIndexBudget(),
        bank_scope=bank_id,
    )


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
        """An index whose bank no longer exists appears in no row count, only in the catalog.

        It can therefore only be found by scanning the catalog and subtracting —
        which is also why the drop side works on an instance that cannot plan a
        query against memory_units at all.
        """
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            names = await _build_indexes_for(conn, bank_id)

        # Delete the bank's rows and profile, but leave the indexes behind, as a
        # deployment whose delete path could not run would.
        async with acquire_with_retry(backend) as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)
            for name in names:
                assert await _index_exists(conn, name), "setup: index should outlive the bank here"

            result = await _reconcile(conn, bank_id=None)

            assert result.dropped >= len(names)
            for name in names:
                assert not await _index_exists(conn, name), f"orphan {name} should be dropped"

    @pytest.mark.asyncio
    async def test_drop_budget_bounds_one_pass_and_reports_a_backlog(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """A bounded pass must say so, or the loop would wait out the full interval."""
        bank_id = await _seed_bank(memory, request_context, _DROP_BELOW - 1)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _build_indexes_for(conn, bank_id)

                result = await _reconcile(conn, bank_id, budget=VectorIndexBudget(max_drops=1))

                assert result.dropped == 1
                assert result.backlog is True
                remaining = [n for n in names if await _index_exists(conn, n)]
                assert len(remaining) == len(names) - 1
        finally:
            async with acquire_with_retry(backend) as conn:
                await _drop_bank_indexes(conn, bank_id)
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_build_budget_is_pass_global_not_per_schema(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """One budget instance threaded through every schema in the pass.

        A per-schema cap would let a deployment with a thousand tenant schemas
        issue a thousand times the intended load in a single sweep.
        """
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                budget = VectorIndexBudget(max_builds=2)

                first = await _reconcile(conn, bank_id, budget=budget)
                assert first.created == 2
                assert first.backlog is True

                # Same budget object, as reconcile_vector_indexes threads it: the
                # second schema in a pass inherits the spend, so nothing is left.
                second = await _reconcile(conn, bank_id, budget=budget)
                assert second.created == 0
                assert second.backlog is True
        finally:
            async with acquire_with_retry(backend) as conn:
                await _drop_bank_indexes(conn, bank_id)
            await memory.delete_bank(bank_id, request_context=request_context)


class TestReconcileMechanics:
    @pytest.mark.asyncio
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


class TestDiscoveryRoutine:
    @pytest.mark.asyncio
    async def test_returns_counts_at_or_above_the_floor_only(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """The routine is handed the drop floor, so the gap is visible to the caller.

        A partition between the two bounds must come back — the reconcile needs
        its row count to decide to leave it alone. If the routine filtered on the
        build threshold instead, the gap would look like 'no rows' and its index
        would be dropped every sweep.
        """
        bank_id = await _seed_bank(memory, request_context, _BETWEEN)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                by_schema = await discover_partitions(conn, fq_routine("banks_needing_vector_index"), bank_id=bank_id)
                partitions = by_schema.get(_TEST_SCHEMA, [])

                assert {p.fact_type for p in partitions} == set(_BANK_INDEX_FACT_TYPES)
                assert all(p.row_count == _BETWEEN for p in partitions)
                assert all(p.bank_id == bank_id for p in partitions)
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_partition_under_the_floor_is_absent(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        bank_id = await _seed_bank(memory, request_context, _DROP_BELOW - 1)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                by_schema = await discover_partitions(conn, fq_routine("banks_needing_vector_index"), bank_id=bank_id)
                assert by_schema.get(_TEST_SCHEMA, []) == []
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
            assert result.banks_scanned == 1  # only the targeted bank

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
    Entitlement is recomputed from row counts on every sweep, and how the rows
    got there is not something the reconcile can observe.
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
    """A reconcile must never act outside the scope it was given.

    The drop side is defined by subtraction — "every index this schema owns that
    the partition list does not account for" — which makes scoping a correctness
    property rather than an optimisation. Narrow the partition list without
    narrowing the drop set and the subtraction quietly widens to everything else.
    """

    @pytest.mark.asyncio
    async def test_scoped_reconcile_leaves_other_banks_indexes_alone(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """Repairing one bank must not drop a bystander's indexes.

        `repair-bank --bank X` discovers only X's partitions, so every other
        bank's index is absent from the keep set. Without an explicit scope on
        the drop side, one targeted repair would strip the entire schema.
        """
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

    @pytest.mark.asyncio
    async def test_schema_skipped_by_discovery_is_not_reconciled(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """A schema discovery could not scan is left alone, not treated as empty.

        The routine skips a schema that vanished or is under concurrent DDL. Read
        as "scanned, owns nothing", that schema's every vector index would be
        dropped and rebuilt on the following pass — the most expensive possible
        response to a transient lock.
        """
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _build_indexes_for(conn, bank_id)
                index_clause = _vector_index_clause()
                assert index_clause is not None

                # Discovery returned nothing for this schema: it has no key, so
                # reconcile must not visit it at all.
                results = await vector_index_health.reconcile_vector_indexes(conn, [_TEST_SCHEMA], index_clause, {})

                assert results == [], "an unscanned schema must not be reconciled"
                for name in names:
                    assert await _index_exists(conn, name), f"{name} must survive a pass that never saw its schema"
        finally:
            async with acquire_with_retry(backend) as conn:
                await _drop_bank_indexes(conn, bank_id)
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_scanned_but_empty_schema_is_reconciled(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """The other half: a present-but-empty list does mean "drop what is there".

        Distinguishing this from the skipped case is the entire point of the
        routine's sentinel row, so both directions are asserted.
        """
        bank_id = await _seed_bank(memory, request_context, _BUILD_AT)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _build_indexes_for(conn, bank_id)
                index_clause = _vector_index_clause()
                assert index_clause is not None

                results = await vector_index_health.reconcile_vector_indexes(
                    conn, [_TEST_SCHEMA], index_clause, {_TEST_SCHEMA: []}, bank_id=bank_id
                )

                assert len(results) == 1
                assert results[0].dropped == len(names)
                for name in names:
                    assert not await _index_exists(conn, name)
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_discovery_reports_a_scanned_schema_with_no_qualifying_banks(
        self, memory: MemoryEngine, request_context: RequestContext, low_threshold
    ):
        """The sentinel row: a scanned schema appears as a key with an empty list."""
        bank_id = await _seed_bank(memory, request_context, _DROP_BELOW - 1)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                by_schema = await discover_partitions(conn, fq_routine("banks_needing_vector_index"), bank_id=bank_id)

                assert _TEST_SCHEMA in by_schema, "a scanned schema must be reported even with nothing to build"
                assert by_schema[_TEST_SCHEMA] == []
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)
