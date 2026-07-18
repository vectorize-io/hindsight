"""Tests for the MaintenanceLoop: due-timer logic, consolidation reconcile
gating, and cross-schema retention purge."""

import time
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

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


def test_reconcile_request_marks_job_due_and_wakes_loop():
    loop = MaintenanceLoop(engine=None)
    loop._last_run["vector_index_reconcile"] = time.monotonic()

    loop.request_vector_index_reconcile()

    assert "vector_index_reconcile" not in loop._last_run
    assert loop._wakeup.is_set()


@pytest.mark.asyncio
async def test_tick_runs_vector_index_reconcile_at_start(monkeypatch):
    loop = MaintenanceLoop(engine=None)
    called = 0

    async def _record():
        nonlocal called
        called += 1

    async def _noop(*args, **kwargs):
        return None

    cfg = SimpleNamespace(
        audit_log_enabled=False,
        audit_log_retention_days=-1,
        llm_trace_enabled=False,
        llm_trace_retention_days=-1,
        consolidation_reconcile_interval_seconds=0,
        mental_model_refresh_tick_seconds=0,
        vector_index_reconcile_interval_seconds=3600,
    )
    monkeypatch.setattr("hindsight_api.engine.maintenance.get_config", lambda: cfg)
    monkeypatch.setattr(loop, "_run_vector_index_reconcile", _record)
    monkeypatch.setattr(loop, "_run_retention", _noop)

    await loop._tick()
    await loop._tick()

    assert called == 1


@pytest.mark.asyncio
async def test_vector_index_reconcile_uses_raw_autocommit_connection(monkeypatch):
    import hindsight_api.engine.maintenance as maintenance_mod

    events: list[object] = []

    class FakeRawConnection:
        pass

    @asynccontextmanager
    async def fake_raw_connection(_engine):
        events.append("opened")
        try:
            yield FakeRawConnection()
        finally:
            events.append("closed")

    async def fake_reconcile(conn, schemas, index_clause):
        events.append((conn.__class__.__name__, schemas, index_clause))
        return [SimpleNamespace(created=2, failed=0)]

    engine = SimpleNamespace(
        _tenant_extension=SimpleNamespace(
            list_tenants=lambda: _async_result([SimpleNamespace(schema="public"), SimpleNamespace(schema="tenant_a")])
        )
    )
    monkeypatch.setattr(maintenance_mod, "_raw_postgres_connection", fake_raw_connection)
    monkeypatch.setattr(maintenance_mod, "_vector_index_clause", lambda: "USING hnsw (embedding vector_cosine_ops)")
    monkeypatch.setattr(maintenance_mod, "reconcile_vector_indexes", fake_reconcile)

    await MaintenanceLoop(engine)._run_vector_index_reconcile()

    assert events[0] == "opened"
    assert events[1] == (
        "FakeRawConnection",
        ["public", "tenant_a"],
        "USING hnsw (embedding vector_cosine_ops)",
    )
    assert events[-1] == "closed"


async def _async_result(value):
    return value


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
