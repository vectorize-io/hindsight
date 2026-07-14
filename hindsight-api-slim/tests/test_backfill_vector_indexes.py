"""Regression tests for `hindsight-admin backfill-vector-indexes` (issue #2645).

Per-(bank, fact_type) partial vector indexes are only created at fresh-bank
creation time. Banks that arrive already populated — via restore, a cross-version
upgrade, or a vector-extension switch — never hit that path, so their queries
silently fall back to a global index + post-filter (slower, ~30% recall@10 miss).

These tests prove the operator escape hatch:

* the core regression — a populated bank whose per-bank indexes were dropped
  (simulating restore/upgrade) gets them rebuilt by the backfill;
* `--dry-run` creates nothing;
* a re-run is idempotent (no error, no duplicates);
* a backend that doesn't use per-bank indexes is a no-op.

Everything asserted is deterministic (index presence via pg_indexes) — no LLM is
needed, so memory_units are inserted directly with the `memory` (MockLLM) fixture.
"""

import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.admin import cli
from hindsight_api.admin.cli import _run_backfill_vector_indexes
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain.bank_utils import _BANK_INDEX_FACT_TYPES, _bank_index_name

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


async def _expected_index_names(conn, bank_id: str) -> list[str]:
    internal_id = await _bank_internal_id(conn, bank_id)
    return [_bank_index_name(ft, internal_id) for ft in _BANK_INDEX_FACT_TYPES]


async def _seed_bank(memory: MemoryEngine, request_context: RequestContext) -> str:
    """Create a bank (auto-creates internal_id + per-bank indexes) and populate it."""
    bank_id = f"test-backfill-{uuid.uuid4().hex[:8]}"
    # get_bank_profile lazily creates the bank row, which also creates the
    # per-bank vector indexes (instant on an empty bank).
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
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
    """Drop every per-(bank, fact_type) index to simulate the restore/upgrade gap."""
    names = await _expected_index_names(conn, bank_id)
    for name in names:
        await conn.execute(f"DROP INDEX IF EXISTS {_TEST_SCHEMA}.{name}")
    return names


class TestBackfillVectorIndexes:
    @pytest.mark.asyncio
    async def test_backfill_recreates_dropped_indexes(
        self, memory: MemoryEngine, request_context: RequestContext, pg0_db_url: str
    ):
        bank_id = await _seed_bank(memory, request_context)
        pool = await memory._get_pool()

        try:
            async with pool.acquire() as conn:
                names = await _drop_bank_indexes(conn, bank_id)
                # Simulated uncovered state: none of the per-bank indexes exist.
                for name in names:
                    assert not await _index_exists(conn, name), f"{name} should be dropped"

            results = await _run_backfill_vector_indexes(pg0_db_url, schema=_TEST_SCHEMA)

            # One schema scanned; our seeded fact types were all rebuilt.
            assert len(results) == 1
            result = results[0]
            assert result.schema == _TEST_SCHEMA
            assert result.failed == 0
            assert result.created >= len(_BANK_INDEX_FACT_TYPES)

            async with pool.acquire() as conn:
                for name in names:
                    assert await _index_exists(conn, name), f"{name} should be rebuilt"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_dry_run_creates_nothing(
        self, memory: MemoryEngine, request_context: RequestContext, pg0_db_url: str
    ):
        bank_id = await _seed_bank(memory, request_context)
        pool = await memory._get_pool()

        try:
            async with pool.acquire() as conn:
                names = await _drop_bank_indexes(conn, bank_id)

            results = await _run_backfill_vector_indexes(pg0_db_url, schema=_TEST_SCHEMA, dry_run=True)

            result = results[0]
            assert result.created == 0
            assert result.skipped >= len(_BANK_INDEX_FACT_TYPES)

            # Dry run must not have created any index.
            async with pool.acquire() as conn:
                for name in names:
                    assert not await _index_exists(conn, name), f"{name} must NOT exist after dry-run"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_rerun_is_idempotent(self, memory: MemoryEngine, request_context: RequestContext, pg0_db_url: str):
        bank_id = await _seed_bank(memory, request_context)
        pool = await memory._get_pool()

        try:
            async with pool.acquire() as conn:
                names = await _drop_bank_indexes(conn, bank_id)

            first = (await _run_backfill_vector_indexes(pg0_db_url, schema=_TEST_SCHEMA))[0]
            assert first.created >= len(_BANK_INDEX_FACT_TYPES)

            # Second run: everything already present, nothing created, no error.
            second = (await _run_backfill_vector_indexes(pg0_db_url, schema=_TEST_SCHEMA))[0]
            assert second.created == 0
            assert second.failed == 0
            assert second.already_present >= len(_BANK_INDEX_FACT_TYPES)

            async with pool.acquire() as conn:
                for name in names:
                    # Exactly one index per name — no duplicates.
                    count = await conn.fetchval(
                        "SELECT count(*) FROM pg_indexes WHERE schemaname = $1 AND indexname = $2",
                        _TEST_SCHEMA,
                        name,
                    )
                    assert count == 1, f"{name} should exist exactly once"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_backend_without_per_bank_indexes_is_noop(
        self, memory: MemoryEngine, request_context: RequestContext, pg0_db_url: str, monkeypatch
    ):
        bank_id = await _seed_bank(memory, request_context)
        pool = await memory._get_pool()

        try:
            async with pool.acquire() as conn:
                names = await _drop_bank_indexes(conn, bank_id)

            # Simulate a backend (AlloyDB ScaNN / Oracle) that uses a single
            # global vector index — the command must be a no-op.
            monkeypatch.setattr(cli, "_vector_index_clause", lambda: None)

            from typer.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(cli.app, ["backfill-vector-indexes", "--schema", _TEST_SCHEMA])
            assert result.exit_code == 0, result.output
            assert "does not use per-bank vector indexes" in result.output

            # No indexes were created.
            async with pool.acquire() as conn:
                for name in names:
                    assert not await _index_exists(conn, name), f"{name} must NOT exist for no-op backend"
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)
