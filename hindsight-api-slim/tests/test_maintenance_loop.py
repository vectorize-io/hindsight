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
