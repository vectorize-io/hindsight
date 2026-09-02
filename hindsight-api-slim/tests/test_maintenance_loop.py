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


def _tick_cfg(**overrides):
    """A config stub for ``_tick``, with every job disabled unless overridden."""
    from types import SimpleNamespace

    fields = {
        "consolidation_reconcile_interval_seconds": 0,
        "audit_log_enabled": False,
        "audit_log_retention_days": 0,
        "llm_trace_enabled": False,
        "llm_trace_retention_days": 0,
        "mental_model_refresh_tick_seconds": 0,
        "retention_sweep_interval_seconds": 3600,
        "operation_retention_days": 0,
        "operation_cleanup_interval_seconds": 900,
        "operation_cleanup_batch_size": 1000,
        "maintenance_start_jitter_seconds": 0,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_is_due_runs_at_start_then_waits_interval():
    """A job is due on first check (run-at-start), then not until its interval elapses."""
    loop = MaintenanceLoop(engine=None)  # _is_due needs no engine

    assert loop._is_due("job", 3600) is True  # never run -> due
    assert loop._is_due("job", 3600) is False  # just ran -> not due

    # Simulate the interval having elapsed.
    loop._last_run["job"] = time.monotonic() - 4000
    assert loop._is_due("job", 3600) is True


@pytest.mark.asyncio
async def test_retention_sweep_cadence_is_configurable(monkeypatch):
    """The retention sweep's interval is server config, not a constant, and 0 disables it."""
    from unittest.mock import AsyncMock

    import hindsight_api.engine.maintenance as maintenance_mod

    cfg = _tick_cfg(audit_log_retention_days=7, retention_sweep_interval_seconds=0)
    monkeypatch.setattr(maintenance_mod, "get_config", lambda: cfg)
    loop = MaintenanceLoop(engine=None)
    loop._run_retention = AsyncMock()

    await loop._tick()
    loop._run_retention.assert_not_awaited()

    cfg.retention_sweep_interval_seconds = 3600
    await loop._tick()
    loop._run_retention.assert_awaited_once()


class TestStartJitter:
    """The first tick is offset by a random delay per process.

    Every job is due the first time ``_is_due`` sees it, so a fleet started
    together (deploy, rolling restart) would otherwise run every cross-tenant
    sweep in every process at the same instant.
    """

    @pytest.mark.asyncio
    async def test_zero_jitter_starts_immediately(self, monkeypatch):
        """0 disables the offset, so tests and single-process deployments stay deterministic."""
        import hindsight_api.engine.maintenance as maintenance_mod

        monkeypatch.setattr(maintenance_mod, "get_config", lambda: _tick_cfg(maintenance_start_jitter_seconds=0))
        loop = MaintenanceLoop(engine=None)

        assert await loop._wait_start_jitter() is True

    @pytest.mark.asyncio
    async def test_delay_is_drawn_from_the_configured_window(self, monkeypatch):
        """The offset is a random draw over [0, jitter], not a fixed sleep — a fixed
        one would keep the fleet aligned, just later."""
        import hindsight_api.engine.maintenance as maintenance_mod

        monkeypatch.setattr(maintenance_mod, "get_config", lambda: _tick_cfg(maintenance_start_jitter_seconds=60))
        draws: list[tuple[float, float]] = []

        def _uniform(low, high):
            draws.append((low, high))
            return 0.0

        monkeypatch.setattr(maintenance_mod.random, "uniform", _uniform)
        loop = MaintenanceLoop(engine=None)

        assert await loop._wait_start_jitter() is True
        assert draws == [(0, 60)]

    @pytest.mark.asyncio
    async def test_stop_during_jitter_aborts_the_loop(self, monkeypatch):
        """A process shut down inside its start offset must not go on to tick."""
        import hindsight_api.engine.maintenance as maintenance_mod

        monkeypatch.setattr(maintenance_mod, "get_config", lambda: _tick_cfg(maintenance_start_jitter_seconds=60))
        monkeypatch.setattr(maintenance_mod.random, "uniform", lambda _low, _high: 30.0)
        loop = MaintenanceLoop(engine=None)
        loop._stop.set()

        assert await loop._wait_start_jitter() is False


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
async def test_cron_discovery_predicate_can_use_the_partial_index(memory: MemoryEngine):
    """``idx_mental_models_cron``'s predicate must match the discovery routine's WHERE.

    ``mental_models_with_cron()`` probes every tenant schema on every sweep, so a
    predicate mismatch is expensive and *silent*: the index exists, the sweep still
    returns the right rows, and every tenant keeps paying a sequential scan of its
    mental_models table. Disabling seqscan proves the planner considers the index
    usable for that predicate, which is the property a future edit to either side
    can regress. It says nothing about which plan wins on cost.
    """
    async with memory._pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SET LOCAL enable_seqscan = off")
            rows = await conn.fetch(
                "EXPLAIN SELECT bank_id, id FROM mental_models WHERE COALESCE(trigger->>'refresh_cron', '') <> ''"
            )
    plan = "\n".join(r[0] for r in rows)
    assert "idx_mental_models_cron" in plan, plan


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


class TestRetentionSweepPacing:
    """The retention sweep must never be one long unbounded DELETE, and must stay
    safe when every process runs it at once.

    The maintenance loop runs in every API/worker process with no leader election,
    so the previous single `DELETE ... WHERE started_at < cutoff` per schema ran
    concurrently on every pod: minutes-long statements pinned on disk reads,
    blocking each other on row locks and starving foreground queries. The sweep now
    deletes in bounded chunks whose rows are claimed with FOR UPDATE SKIP LOCKED,
    so concurrent sweepers take disjoint chunks instead of waiting on each other.
    """

    @staticmethod
    async def _make_probe_table(memory: MemoryEngine, expired: int, recent: int) -> str:
        """A throwaway table shaped like the retention tables (id + started_at).

        Purpose-made so the assertions are exact: sweeping the real audit_log would
        also delete rows other tests left behind in the shared database.
        """
        table = f"retention_probe_{uuid.uuid4().hex[:8]}"
        async with memory._pool.acquire() as conn:
            await conn.execute(
                f"CREATE TABLE {table} (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), started_at TIMESTAMPTZ NOT NULL)"
            )
            await conn.execute(
                f"INSERT INTO {table} (started_at) SELECT now() - INTERVAL '10 days' FROM generate_series(1, $1)",
                expired,
            )
            await conn.execute(
                f"INSERT INTO {table} (started_at) SELECT now() FROM generate_series(1, $1)",
                recent,
            )
        return table

    @staticmethod
    async def _count(memory: MemoryEngine, table: str) -> int:
        async with memory._pool.acquire() as conn:
            return await conn.fetchval(f"SELECT COUNT(*) FROM {table}")

    @staticmethod
    async def _drop(memory: MemoryEngine, table: str) -> None:
        async with memory._pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {table}")

    @pytest.mark.asyncio
    async def test_chunked_delete_drains_every_expired_row(self, memory: MemoryEngine, monkeypatch):
        """Chunking is a pacing device, not a cap: the sweep keeps going until the
        expired range is gone, and never touches rows inside the window."""
        import hindsight_api.engine.maintenance as maintenance_mod

        monkeypatch.setattr(maintenance_mod, "_RETENTION_BATCH_SIZE", 2)
        table = await self._make_probe_table(memory, expired=5, recent=1)
        try:
            deleted = await MaintenanceLoop(memory)._purge_expired(table, "started_at", 7)

            assert deleted == 5  # three chunks of 2, 2, 1
            assert await self._count(memory, table) == 1  # only the row inside the window survives
        finally:
            await self._drop(memory, table)

    @pytest.mark.asyncio
    async def test_per_run_batch_ceiling_bounds_one_sweep(self, memory: MemoryEngine, monkeypatch):
        """A backlog is drained over several runs rather than in one long statement."""
        import hindsight_api.engine.maintenance as maintenance_mod

        monkeypatch.setattr(maintenance_mod, "_RETENTION_BATCH_SIZE", 2)
        monkeypatch.setattr(maintenance_mod, "_RETENTION_MAX_BATCHES", 1)
        table = await self._make_probe_table(memory, expired=5, recent=1)
        try:
            await MaintenanceLoop(memory)._purge_expired(table, "started_at", 7)

            # One chunk of 2 removed; the rest waits for the next run.
            assert await self._count(memory, table) == 4
        finally:
            await self._drop(memory, table)

    @pytest.mark.asyncio
    async def test_concurrent_sweepers_split_the_work_without_blocking(self, memory: MemoryEngine, monkeypatch):
        """The point of SKIP LOCKED: with no leader, several pods sweep at once and
        each row is deleted exactly once — the deletes partition the backlog instead
        of waiting on each other's row locks."""
        import asyncio

        import hindsight_api.engine.maintenance as maintenance_mod

        monkeypatch.setattr(maintenance_mod, "_RETENTION_BATCH_SIZE", 2)
        table = await self._make_probe_table(memory, expired=20, recent=1)
        try:
            sweepers = [MaintenanceLoop(memory)._purge_expired(table, "started_at", 7) for _ in range(3)]
            deleted = await asyncio.wait_for(asyncio.gather(*sweepers), timeout=60)

            # Every expired row accounted for exactly once across the three pods:
            # a row double-counted would mean a sweeper waited for, then re-deleted,
            # another's rows; a missing row would mean SKIP LOCKED dropped it.
            assert sum(deleted) == 20
            assert await self._count(memory, table) == 1
        finally:
            await self._drop(memory, table)


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
        # The sweep purges expired export archives before pruning each schema's rows;
        # stub it as a no-op so these tests stay focused on discovery + prune scoping.
        engine.purge_expired_export_archives = AsyncMock(return_value=0)
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
        from unittest.mock import AsyncMock

        import hindsight_api.engine.maintenance as maintenance_mod

        cfg = _tick_cfg(operation_retention_days=0)
        monkeypatch.setattr(maintenance_mod, "get_config", lambda: cfg)
        loop = MaintenanceLoop(engine=None)
        loop._run_operation_cleanup = AsyncMock()

        await loop._tick()
        loop._run_operation_cleanup.assert_not_awaited()

        cfg.operation_retention_days = 30
        await loop._tick()
        loop._run_operation_cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_cadence_is_configurable(self, monkeypatch):
        """The interval is server config, not a constant: 0 disables the job even
        with retention enabled, so a large deployment can turn off the per-tick
        cross-tenant probe without giving up retention."""
        from unittest.mock import AsyncMock

        import hindsight_api.engine.maintenance as maintenance_mod

        cfg = _tick_cfg(operation_retention_days=30, operation_cleanup_interval_seconds=0)
        monkeypatch.setattr(maintenance_mod, "get_config", lambda: cfg)
        loop = MaintenanceLoop(engine=None)
        loop._run_operation_cleanup = AsyncMock()

        await loop._tick()
        loop._run_operation_cleanup.assert_not_awaited()

        cfg.operation_cleanup_interval_seconds = 900
        await loop._tick()
        loop._run_operation_cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_archive_purge_shares_the_prune_batch_bound(self):
        """The archive purge is bounded by the same batch size as the row prune.

        Unbounded, it re-selects every expired export and re-deletes blobs that are
        already gone on every cycle — ``storage_key`` survives in the row until the
        row itself is pruned, so nothing marks the blob as handled.
        """
        engine, _backend, _conn = self._make_engine(expired=["public"], tenants=("public",))
        loop = MaintenanceLoop(engine)

        await loop._run_operation_cleanup(self._cfg(batch=250))

        assert engine.purge_expired_export_archives.await_args.kwargs["batch_size"] == 250
