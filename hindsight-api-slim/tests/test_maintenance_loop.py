"""Tests for the MaintenanceLoop: due-timer logic, consolidation reconcile
gating, and cross-schema retention purge."""

import time
import uuid

import pytest

from hindsight_api.engine.maintenance import MaintenanceLoop
from hindsight_api.engine.memory_engine import MemoryEngine


def test_start_is_noop_on_oracle(monkeypatch):
    """The loop is PostgreSQL-only (PG-only tables + routines); it must not start on Oracle."""
    import hindsight_api.engine.maintenance as maintenance_mod

    monkeypatch.setattr(maintenance_mod, "_is_oracle", lambda: True)
    loop = MaintenanceLoop(engine=None)
    loop.start()
    assert loop._task is None


def test_is_due_runs_at_start_then_waits_interval():
    """A job is due on first check (run-at-start), then not until its interval elapses."""
    loop = MaintenanceLoop(engine=None)  # _is_due needs no engine

    assert loop._is_due("job", 3600) is True  # never run -> due
    assert loop._is_due("job", 3600) is False  # just ran -> not due

    # Simulate the interval having elapsed.
    loop._last_run["job"] = time.monotonic() - 4000
    assert loop._is_due("job", 3600) is True


async def _make_bank(memory: MemoryEngine, request_context, suffix: str, config_json: str | None = None) -> str:
    bank_id = f"recon-{suffix}-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    if config_json is not None:
        async with memory._pool.acquire() as conn:
            await conn.execute("UPDATE banks SET config = $2::jsonb WHERE bank_id = $1", bank_id, config_json)
    return bank_id


async def _insert_fact(conn, bank_id: str) -> None:
    await conn.execute(
        "INSERT INTO memory_units (id, bank_id, text, fact_type, created_at) VALUES ($1, $2, 'a fact', 'experience', now())",
        uuid.uuid4(),
        bank_id,
    )


@pytest.mark.asyncio
async def test_reconcile_submits_eligible_skips_disabled_and_in_flight(
    memory: MemoryEngine, request_context, monkeypatch
):
    """Reconcile enqueues consolidation for eligible banks and skips banks that
    disabled auto-consolidation or already have an in-flight consolidation."""
    eligible = await _make_bank(
        memory, request_context, "eligible", '{"enable_observations": true, "enable_auto_consolidation": true}'
    )
    disabled = await _make_bank(memory, request_context, "disabled", '{"enable_auto_consolidation": false}')
    in_flight = await _make_bank(memory, request_context, "inflight")

    async with memory._pool.acquire() as conn:
        await _insert_fact(conn, eligible)
        await _insert_fact(conn, disabled)
        await _insert_fact(conn, in_flight)
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'consolidation', 'processing', '{}'::jsonb)
            """,
            uuid.uuid4(),
            in_flight,
        )

    submitted: list[str] = []

    async def _record(*, bank_id, request_context, observation_scopes=None):
        submitted.append(bank_id)
        return {"operation_id": str(uuid.uuid4())}

    monkeypatch.setattr(memory, "submit_async_consolidation", _record)

    await MaintenanceLoop(memory)._run_reconcile()

    # Shared pg0 may contain other eligible banks, so assert on membership.
    assert eligible in submitted
    assert disabled not in submitted
    assert in_flight not in submitted


@pytest.mark.asyncio
async def test_purge_expired_deletes_old_rows_across_schema(memory: MemoryEngine):
    """_purge_expired deletes rows older than the cutoff and keeps recent ones."""
    tag = f"maint-purge-{uuid.uuid4().hex[:8]}"
    async with memory._pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO audit_log (action, transport, started_at) VALUES ($1, 'system', now() - INTERVAL '10 days')",
            tag,
        )
        await conn.execute(
            "INSERT INTO audit_log (action, transport, started_at) VALUES ($1, 'system', now())",
            tag,
        )

    await MaintenanceLoop(memory)._purge_expired("audit_log", "started_at", 7)

    async with memory._pool.acquire() as conn:
        remaining = await conn.fetchval("SELECT COUNT(*) FROM audit_log WHERE action = $1", tag)
    assert remaining == 1  # only the recent row survives


class TestOperationCleanupJob:
    """Terminal-operation cleanup runs as a scheduled maintenance job.

    It previously rode the worker's task-claiming loop, firing only when that
    loop happened to iterate. It is periodic housekeeping like the retention
    sweeps, so it belongs on the same tick with its own interval.

    Discovery is one cross-tenant round-trip (``schemas_with_expired_operations``)
    instead of a connection plus prune transaction per tenant, so a schema with
    nothing expired costs nothing.
    """

    @staticmethod
    def _make_engine(*, expired=None, tenants=(), fetch_error=None):
        from contextlib import asynccontextmanager
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        conn = MagicMock()

        @asynccontextmanager
        async def transaction():
            yield conn

        conn.transaction = transaction
        conn.fetch = AsyncMock(side_effect=fetch_error, return_value=[(s,) for s in (expired or [])])

        backend = MagicMock()
        backend.backend_type = "postgresql"
        backend.ops.prune_terminal_operations = AsyncMock(return_value=0)

        @asynccontextmanager
        async def acquire(*_args, **_kwargs):
            yield conn

        backend.acquire = acquire

        engine = MagicMock()
        engine._backend = backend
        engine._tenant_extension.list_tenants = AsyncMock(return_value=[SimpleNamespace(schema=s) for s in tenants])
        return engine, backend, conn

    @staticmethod
    def _cfg(days=30, batch=1000):
        from types import SimpleNamespace

        return SimpleNamespace(operation_retention_days=days, operation_cleanup_batch_size=batch)

    @staticmethod
    def _pruned_tables(backend):
        return [call.args[1] for call in backend.ops.prune_terminal_operations.await_args_list]

    @pytest.mark.asyncio
    async def test_only_schemas_reported_as_expired_are_pruned(self, monkeypatch):
        engine, backend, conn = self._make_engine(
            expired=["public", "tenant_b"], tenants=("public", "tenant_a", "tenant_b")
        )
        loop = MaintenanceLoop(engine)

        await loop._run_operation_cleanup(self._cfg())

        # One discovery round-trip, carrying the configured retention window.
        conn.fetch.assert_awaited_once()
        # Schema-qualified via fq_routine, so a non-public deployment calls its own
        # copy rather than a public one that may not exist (#2638).
        assert '"public".schemas_with_expired_operations' in conn.fetch.await_args.args[0]
        assert conn.fetch.await_args.args[1] == 30
        # tenant_a has nothing expired, so it costs no connection or transaction.
        assert self._pruned_tables(backend) == ['"public".async_operations', '"tenant_b".async_operations']

    @pytest.mark.asyncio
    async def test_nothing_expired_anywhere_skips_pruning_entirely(self):
        engine, backend, _conn = self._make_engine(expired=[], tenants=("public", "tenant_a"))
        loop = MaintenanceLoop(engine)

        await loop._run_operation_cleanup(self._cfg())

        backend.ops.prune_terminal_operations.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_schema_no_tenant_claims_is_skipped(self):
        """The routine reports every schema owning an async_operations table,
        including ones tenant discovery doesn't claim; pruning stays scoped."""
        engine, backend, _conn = self._make_engine(expired=["public", "stranger"], tenants=("public",))
        loop = MaintenanceLoop(engine)

        await loop._run_operation_cleanup(self._cfg())

        assert self._pruned_tables(backend) == ['"public".async_operations']

    @pytest.mark.asyncio
    async def test_missing_routine_skips_the_sweep(self):
        """Without the routine there is no sweep — deliberately no full-scan
        fallback, so a deployment that never ran the migration fails loudly in
        the logs rather than silently paying the per-tenant cost."""
        engine, backend, _conn = self._make_engine(
            fetch_error=RuntimeError("function schemas_with_expired_operations(integer) does not exist"),
            tenants=("public", "tenant_a"),
        )
        loop = MaintenanceLoop(engine)

        await loop._run_operation_cleanup(self._cfg())

        backend.ops.prune_terminal_operations.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tick_schedules_cleanup_only_when_retention_enabled(self, monkeypatch):
        """Retention 0 (the default) disables the job entirely."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import hindsight_api.engine.maintenance as maintenance_mod

        cfg = SimpleNamespace(
            consolidation_reconcile_interval_seconds=0,
            audit_log_enabled=False,
            audit_log_retention_days=0,
            llm_trace_enabled=False,
            llm_trace_retention_days=0,
            mental_model_refresh_tick_seconds=0,
            operation_retention_days=0,
            operation_cleanup_batch_size=1000,
        )
        monkeypatch.setattr(maintenance_mod, "get_config", lambda: cfg)
        loop = MaintenanceLoop(engine=None)
        loop._run_operation_cleanup = AsyncMock()

        await loop._tick()
        loop._run_operation_cleanup.assert_not_awaited()

        cfg.operation_retention_days = 30
        await loop._tick()
        loop._run_operation_cleanup.assert_awaited_once()
