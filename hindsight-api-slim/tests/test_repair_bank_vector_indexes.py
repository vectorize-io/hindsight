"""Regression tests for per-bank vector index coverage repair (issue #2645).

Per-(bank, fact_type) partial vector indexes are only created at fresh-bank
creation time. Banks that arrive already populated — via logical restore, a
cross-version upgrade, or a vector-extension switch — never hit that path, so
their recall silently falls back to a global index + post-filter (slower,
~30% recall@10 miss).

Two things are proven here:

* the `import-bank` leak is plugged — a restored bank gets its per-bank indexes;
* the `repair-bank` command is the re-runnable escape hatch — it rebuilds
  missing OR invalid coverage (a name-colliding index that lacks the partial
  predicate counts as invalid and is rebuilt, unlike a name-only check), is a
  no-op on non-per-bank backends, and validates its target flags.

Everything asserted is deterministic (index presence/shape via the catalog) —
no LLM is needed, so memory_units are inserted directly.
"""

import uuid

import pytest
from asyncpg.exceptions import DeadlockDetectedError

from hindsight_api import RequestContext
from hindsight_api.admin import cli
from hindsight_api.admin.cli import _run_repair_bank
from hindsight_api.engine.db_utils import acquire_with_retry, retry_with_backoff
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain.bank_utils import _BANK_INDEX_FACT_TYPES, _bank_index_name, _vector_index_clause
from hindsight_api.engine.transfer import export_bank
from hindsight_api.engine.vector_index_health import _repair_schema

_TEST_SCHEMA = "public"


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


async def _seed_bank(memory: MemoryEngine, request_context: RequestContext) -> str:
    """Create a bank (auto-creates internal_id + per-bank indexes) and populate it."""
    bank_id = f"test-repair-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        for ft in _BANK_INDEX_FACT_TYPES:
            await conn.execute(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, created_at, updated_at)
                VALUES ($1, $2, $3, $4, NOW(), NOW(), NOW())
                """,
                uuid.uuid4(),
                bank_id,
                f"seed {ft} fact",
                ft,
            )
    return bank_id


async def _drop_bank_indexes(conn, bank_id: str) -> list[str]:
    """Drop every per-(bank, fact_type) index to simulate the restore/upgrade gap.

    CONCURRENTLY so the drop never takes ACCESS EXCLUSIVE on the shared
    ``memory_units`` table: the test suite runs 8 xdist workers against one pg0
    database, and a blocking DDL here deadlocks unrelated workers' DML.

    Retried on deadlock: CONCURRENTLY still takes ShareUpdateExclusive, which
    conflicts with the ShareLock a *fresh bank's* plain CREATE INDEX holds (that
    one cannot be made concurrent — it runs inside the bank-create transaction).
    So a drop here can still be picked as the victim while another worker seeds
    a bank. That is transient, and the drop is idempotent.
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
    victim, so the repair's retry path can be exercised deterministically.
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


class TestRepairBankCommand:
    @pytest.mark.asyncio
    async def test_repair_single_bank_recreates_dropped_indexes(
        self, memory: MemoryEngine, request_context: RequestContext, pg0_db_url: str
    ):
        bank_id = await _seed_bank(memory, request_context)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _drop_bank_indexes(conn, bank_id)
                for name in names:
                    assert not await _index_exists(conn, name), f"{name} should be dropped"

            results = await _run_repair_bank(
                pg0_db_url, base_schema=_TEST_SCHEMA, schema=_TEST_SCHEMA, bank_id=bank_id, dry_run=False
            )

            assert len(results) == 1
            result = results[0]
            assert result.failed == 0
            assert result.created >= len(_BANK_INDEX_FACT_TYPES)
            # Only the targeted bank was scanned.
            assert result.banks_scanned == 1

            async with acquire_with_retry(backend) as conn:
                for name in names:
                    assert await _index_is_partial_vector(conn, name), f"{name} should be rebuilt as a partial index"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_repair_all_scans_every_bank(
        self, memory: MemoryEngine, request_context: RequestContext, pg0_db_url: str
    ):
        bank_id = await _seed_bank(memory, request_context)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _drop_bank_indexes(conn, bank_id)

            results = await _run_repair_bank(
                pg0_db_url, base_schema=_TEST_SCHEMA, schema=_TEST_SCHEMA, bank_id=None, dry_run=False
            )

            result = results[0]
            # --all scans more than just our bank (other banks may exist in the shared db).
            assert result.banks_scanned >= 1
            assert result.created >= len(_BANK_INDEX_FACT_TYPES)
            async with acquire_with_retry(backend) as conn:
                for name in names:
                    assert await _index_exists(conn, name), f"{name} should be rebuilt"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_invalid_shape_index_is_rebuilt(
        self, memory: MemoryEngine, request_context: RequestContext, pg0_db_url: str
    ):
        """A name-colliding index that lacks the partial predicate is unhealthy → rebuilt.

        This is the differentiator over a name-only existence check (which would
        treat the collision — or a stale INVALID leftover — as 'already present'
        and never repair it).
        """
        bank_id = await _seed_bank(memory, request_context)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _drop_bank_indexes(conn, bank_id)
                # Recreate the FIRST expected index name with the WRONG definition:
                # a plain btree with no partial predicate. Name matches, shape does not.
                # CONCURRENTLY so building the decoy never takes ACCESS EXCLUSIVE on
                # the shared memory_units table (see _drop_bank_indexes).
                bogus = names[0]
                await conn.execute(f"CREATE INDEX CONCURRENTLY {bogus} ON memory_units (bank_id)")
                assert await _index_exists(conn, bogus)
                assert not await _index_is_partial_vector(conn, bogus)

            results = await _run_repair_bank(
                pg0_db_url, base_schema=_TEST_SCHEMA, schema=_TEST_SCHEMA, bank_id=bank_id, dry_run=False
            )
            assert results[0].failed == 0

            async with acquire_with_retry(backend) as conn:
                for name in names:
                    assert await _index_is_partial_vector(conn, name), f"{name} should now be the partial vector index"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_transient_deadlock_is_retried_not_failed(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """A transient deadlock during the CONCURRENTLY build is retried, not
        recorded as a permanent failure.

        This is the exact CI flake: 8 xdist workers share one memory_units
        table, so a concurrent build gets picked as the deadlock victim. Repair
        must converge (rebuild the index) rather than leave result.failed > 0.
        """
        bank_id = await _seed_bank(memory, request_context)
        backend = await memory._get_backend()
        index_clause = _vector_index_clause()
        assert index_clause is not None  # per-bank-index backend (see other tests)
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _drop_bank_indexes(conn, bank_id)
                flaky = _DeadlockOnceOnCreate(conn)
                result = await _repair_schema(flaky, _TEST_SCHEMA, index_clause, dry_run=False, bank_id=bank_id)
                assert flaky.create_calls >= 2, "expected a retry after the injected deadlock"
                assert result.failed == 0, result.failed_indexes
                assert result.created == len(_BANK_INDEX_FACT_TYPES)
                for name in names:
                    assert await _index_exists(conn, name), f"{name} should be rebuilt after the retry"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_dry_run_creates_nothing(
        self, memory: MemoryEngine, request_context: RequestContext, pg0_db_url: str
    ):
        bank_id = await _seed_bank(memory, request_context)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _drop_bank_indexes(conn, bank_id)

            results = await _run_repair_bank(
                pg0_db_url, base_schema=_TEST_SCHEMA, schema=_TEST_SCHEMA, bank_id=bank_id, dry_run=True
            )
            assert results[0].created == 0
            assert results[0].skipped >= len(_BANK_INDEX_FACT_TYPES)

            async with acquire_with_retry(backend) as conn:
                for name in names:
                    assert not await _index_exists(conn, name), f"{name} must NOT exist after dry-run"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_rerun_is_idempotent(self, memory: MemoryEngine, request_context: RequestContext, pg0_db_url: str):
        bank_id = await _seed_bank(memory, request_context)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _drop_bank_indexes(conn, bank_id)

            first = (
                await _run_repair_bank(
                    pg0_db_url, base_schema=_TEST_SCHEMA, schema=_TEST_SCHEMA, bank_id=bank_id, dry_run=False
                )
            )[0]
            assert first.created >= len(_BANK_INDEX_FACT_TYPES)

            second = (
                await _run_repair_bank(
                    pg0_db_url, base_schema=_TEST_SCHEMA, schema=_TEST_SCHEMA, bank_id=bank_id, dry_run=False
                )
            )[0]
            assert second.created == 0
            assert second.failed == 0
            assert second.already_present >= len(_BANK_INDEX_FACT_TYPES)

            async with acquire_with_retry(backend) as conn:
                for name in names:
                    count = await conn.fetchval(
                        "SELECT count(*) FROM pg_indexes WHERE schemaname = $1 AND indexname = $2",
                        _TEST_SCHEMA,
                        name,
                    )
                    assert count == 1, f"{name} should exist exactly once"
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
        self, memory: MemoryEngine, request_context: RequestContext, pg0_db_url: str, monkeypatch
    ):
        bank_id = await _seed_bank(memory, request_context)
        backend = await memory._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                names = await _drop_bank_indexes(conn, bank_id)

            # Simulate a backend (AlloyDB ScaNN / Oracle) with a single global index.
            monkeypatch.setattr(cli, "_vector_index_clause", lambda: None)

            from typer.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(cli.app, ["repair-bank", "--all"])
            assert result.exit_code == 0, result.output
            assert "does not use per-bank vector indexes" in result.output

            async with acquire_with_retry(backend) as conn:
                for name in names:
                    assert not await _index_exists(conn, name), f"{name} must NOT exist for no-op backend"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)


class TestImportBankCreatesIndexes:
    @pytest.mark.asyncio
    async def test_import_bank_creates_per_bank_indexes(self, memory: MemoryEngine, request_context: RequestContext):
        """The import-bank leak (#2645): a restored bank must get its per-bank indexes.

        export → delete → import round-trip, then assert the per-bank partial
        indexes exist for the restored bank. Before the fix, import took the
        SELECT branch of get_or_create_bank_profile and skipped index creation.
        """
        bank_id = f"test-import-{uuid.uuid4().hex[:8]}"
        backend = await memory._get_backend()
        try:
            # Populate directly (deterministic; no LLM) so the bank has real rows.
            await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
            async with acquire_with_retry(backend) as conn:
                for ft in _BANK_INDEX_FACT_TYPES:
                    await conn.execute(
                        """
                        INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, NOW(), NOW(), NOW())
                        """,
                        uuid.uuid4(),
                        bank_id,
                        f"seed {ft} fact",
                        ft,
                    )
                archive = await export_bank(conn, bank_id)

            await memory.delete_bank(bank_id, request_context=request_context)
            result = await memory.import_bank_async(archive, request_context)
            assert result.bank_id == bank_id

            async with acquire_with_retry(backend) as conn:
                for name in await _expected_index_names(conn, bank_id):
                    assert await _index_is_partial_vector(conn, name), (
                        f"{name} should exist after import-bank (the restore leak, #2645)"
                    )
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)
