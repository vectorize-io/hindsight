"""Integration tests for the off-peak consolidation window gate (#2762).

Covers the defer semantics end-to-end: a consolidation task claimed while the
configured daily window is closed is held as ``pending`` with ``next_retry_at``
set to the next window-open instant (no retry_count bump, no error), the
reconcile sweep does not re-submit such a bank, and the extension hook now
feeds into the same deferral path.
"""

import json
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, time, timedelta

import pytest
import pytest_asyncio

from hindsight_api.config import _get_raw_config, get_config
from hindsight_api.engine.maintenance import MaintenanceLoop
from hindsight_api.engine.memory_engine import MemoryEngine, _next_consolidation_window_open
from hindsight_api.engine.task_backend import WorkerTaskBackend
from hindsight_api.worker import WorkerPoller
from hindsight_api.worker.poller import ClaimedTask

# Shared pg0 database state — run serialised with the other worker tests.
pytestmark = pytest.mark.xdist_group("worker_tests")


@pytest_asyncio.fixture
async def backend(pg0_db_url):
    """Create a DatabaseBackend for the poller under test."""
    from hindsight_api.engine.db import create_database_backend
    from hindsight_api.pg0 import resolve_database_url

    resolved_url = await resolve_database_url(pg0_db_url)
    b = create_database_backend("postgresql")
    await b.initialize(resolved_url, min_size=2, max_size=10, command_timeout=30)
    yield b
    await b.shutdown()


@pytest_asyncio.fixture
async def pool(backend):
    """Expose the raw asyncpg pool from the backend for direct DB access."""
    yield backend.get_pool()


@pytest_asyncio.fixture
async def clean_operations(pool):
    """Clean up async_operations rows before and after tests."""
    await pool.execute("DELETE FROM async_operations WHERE status = 'pending'")
    yield
    await pool.execute("DELETE FROM async_operations WHERE bank_id LIKE 'test-window-%' OR bank_id LIKE 'recon-%'")


async def _ensure_bank(pool, bank_id: str) -> None:
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


async def _make_bank(memory: MemoryEngine, request_context, suffix: str) -> str:
    bank_id = f"recon-{suffix}-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    return bank_id


async def _insert_fact(conn, bank_id: str) -> None:
    await conn.execute(
        "INSERT INTO memory_units (id, bank_id, text, fact_type, created_at) "
        "VALUES ($1, $2, 'a fact', 'experience', now())",
        uuid.uuid4(),
        bank_id,
    )


def _closed_window_times() -> tuple[time, time]:
    """A 1-hour daily window that is always closed right now (opens in ~2h)."""
    now = datetime.now(UTC)
    return time((now.hour + 2) % 24, 0), time((now.hour + 3) % 24, 0)


@contextmanager
def _temporary_window(start: time | None, end: time | None, tz: str = "UTC"):
    """Swap the (static, global) window config for the duration of the test."""
    config = _get_raw_config()
    prev = (
        config.consolidation_window_start,
        config.consolidation_window_end,
        config.consolidation_window_tz,
    )
    config.consolidation_window_start = start
    config.consolidation_window_end = end
    config.consolidation_window_tz = tz
    try:
        yield
    finally:
        config.consolidation_window_start, config.consolidation_window_end, config.consolidation_window_tz = prev


class TestConsolidationWindowDefer:
    """Window-closed semantics on the worker execution path."""

    @pytest.mark.asyncio
    async def test_window_closed_defers_without_retry_bump(self, backend, pool, clean_operations, memory):
        """A consolidation executed while the window is closed is held as pending
        until the next window open — no retry_count bump, no error_message."""
        # The shared `memory` fixture uses SyncTaskBackend (inline execution,
        # no queue to defer to). This test exercises the worker execution path,
        # so swap in a worker backend — deferral only makes sense when a poller
        # can re-claim the task once next_retry_at elapses.
        memory._task_backend = WorkerTaskBackend()
        start_t, end_t = _closed_window_times()
        with _temporary_window(start_t, end_t, "UTC"):
            bank_id = f"test-window-def-{uuid.uuid4().hex[:8]}"
            await _ensure_bank(pool, bank_id)
            op_id = uuid.uuid4()
            payload = json.dumps(
                {
                    "type": "consolidate",
                    "operation_type": "consolidation",
                    "operation_id": str(op_id),
                    "bank_id": bank_id,
                }
            )
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status,
                                              task_payload, worker_id, claimed_at, retry_count)
                VALUES ($1, $2, 'consolidation', 'processing', $3::jsonb, 'test-window', now(), 0)
                """,
                op_id,
                bank_id,
                payload,
            )

            poller = WorkerPoller(
                backend=backend,
                worker_id="test-window",
                executor=memory._handle_consolidation,
            )
            await poller.execute_task(ClaimedTask(operation_id=str(op_id), task_dict=json.loads(payload), schema=None))
            completed = await poller.wait_for_active_tasks(timeout=10.0)
            assert completed, "deferred task did not settle within timeout"

            row = await pool.fetchrow(
                "SELECT status, worker_id, claimed_at, retry_count, error_message, next_retry_at "
                "FROM async_operations WHERE operation_id = $1",
                op_id,
            )
            assert row["status"] == "pending"
            assert row["worker_id"] is None
            assert row["claimed_at"] is None
            assert row["retry_count"] == 0, "defer must NOT increment retry_count"
            assert row["error_message"] is None, "defer must NOT write error_message"
            expected_open = _next_consolidation_window_open()
            assert row["next_retry_at"] is not None
            # Boundary precision is one minute; allow caller/DB slack.
            assert abs((row["next_retry_at"] - expected_open).total_seconds()) < 60

    @pytest.mark.asyncio
    async def test_deferred_task_ignored_by_reconcile(self, memory, request_context, monkeypatch):
        """A deferred (pending + future next_retry_at) consolidation op keeps
        banks_needing_consolidation() from re-submitting the same bank."""
        bank_id = await _make_bank(memory, request_context, "deferred")
        async with memory._pool.acquire() as conn:
            await _insert_fact(conn, bank_id)
            await conn.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type,
                                              status, task_payload, next_retry_at)
                VALUES ($1, $2, 'consolidation', 'pending', '{}'::jsonb, $3)
                """,
                uuid.uuid4(),
                bank_id,
                datetime.now(UTC) + timedelta(hours=2),
            )

        submitted: list[str] = []

        async def _record(*, bank_id, request_context, observation_scopes=None):
            submitted.append(bank_id)
            return {"operation_id": str(uuid.uuid4())}

        monkeypatch.setattr(memory, "submit_async_consolidation", _record)

        await MaintenanceLoop(memory)._run_reconcile()

        assert bank_id not in submitted

    @pytest.mark.asyncio
    async def test_window_unset_runs_consolidation_normally(self, memory, request_context):
        """With no window configured, a consolidation task completes normally
        (the default, backwards-compatible behaviour)."""
        bank_id = await _make_bank(memory, request_context, "open")
        async with memory._pool.acquire() as conn:
            await _insert_fact(conn, bank_id)

        result = await memory._handle_consolidation({"operation_id": str(uuid.uuid4()), "bank_id": bank_id})
        assert isinstance(result, dict)
        assert result.get("skipped") is not True
        # The mock-LLM run reports processed memories >= 1 (the fact we seeded).
        assert result.get("memories_processed", 0) >= 1


class TestConsolidationExtensionRejection:
    """The validate_consolidate hook now feeds the deferral path."""

    @pytest.mark.asyncio
    async def test_rejected_consolidation_defers_with_backoff(self, backend, pool, clean_operations, memory):
        """An extension rejection holds the task with the retry backoff — it is
        not treated as a failure (no retry_count bump) and not run."""

        class RejectingValidator:
            async def validate_consolidate(self, ctx):
                from hindsight_api.extensions.operation_validator import ValidationResult

                return ValidationResult(allowed=False, reason="maintenance window")

        memory._operation_validator = RejectingValidator()
        # Worker-mode engine: an extension rejection defers (see the fixture
        # note in test_window_closed_defers_without_retry_bump).
        memory._task_backend = WorkerTaskBackend()
        try:
            bank_id = f"test-window-rej-{uuid.uuid4().hex[:8]}"
            await _ensure_bank(pool, bank_id)
            op_id = uuid.uuid4()
            payload = json.dumps(
                {
                    "type": "consolidate",
                    "operation_type": "consolidation",
                    "operation_id": str(op_id),
                    "bank_id": bank_id,
                }
            )
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status,
                                              task_payload, worker_id, claimed_at, retry_count)
                VALUES ($1, $2, 'consolidation', 'processing', $3::jsonb, 'test-window', now(), 0)
                """,
                op_id,
                bank_id,
                payload,
            )

            poller = WorkerPoller(
                backend=backend,
                worker_id="test-window",
                executor=memory._handle_consolidation,
            )
            await poller.execute_task(ClaimedTask(operation_id=str(op_id), task_dict=json.loads(payload), schema=None))
            completed = await poller.wait_for_active_tasks(timeout=10.0)
            assert completed

            row = await pool.fetchrow(
                "SELECT status, retry_count, error_message, next_retry_at "
                "FROM async_operations WHERE operation_id = $1",
                op_id,
            )
            assert row["status"] == "pending"
            assert row["retry_count"] == 0
            assert row["error_message"] is None
            expected = datetime.now(UTC) + timedelta(seconds=get_config().worker_task_retry_backoff_seconds)
            assert row["next_retry_at"] is not None
            assert abs((row["next_retry_at"] - expected).total_seconds()) < 30
        finally:
            memory._operation_validator = None

    @pytest.mark.asyncio
    async def test_next_window_open_reflects_config_tz(self):
        """The helper honours the configured timezone when the window is set."""
        with _temporary_window(time(22, 0), time(6, 0), "Asia/Shanghai"):
            # 08:00 UTC == 16:00 Shanghai — inside 22:00–06:00? No: still outside.
            # 2026-08-03 12:00 UTC == 20:00 Shanghai — outside 22:00–06:00 (opens at 22:00 = 14:00 UTC).
            nxt = _next_consolidation_window_open(datetime(2026, 8, 3, 12, 0, tzinfo=UTC))
            assert nxt == datetime(2026, 8, 3, 14, 0, tzinfo=UTC)
            # 22:00 Shanghai == 14:00 UTC is inside till 06:00 (22:00–06:00 cross-midnight).
            assert _next_consolidation_window_open(datetime(2026, 8, 3, 16, 0, tzinfo=UTC)) is None


class TestConsolidationSyncBackend:
    """Sync-backend semantics: the fixture's SyncTaskBackend executes tasks
    inline with no worker, so there is no queue to defer to. The window is
    informational (log and run now) and an extension rejection skips — neither
    raises DeferOperation into the HTTP caller (a 500 there) nor strands a
    pending op that nothing will ever re-claim (which would also block
    banks_needing_consolidation() from re-submitting the bank)."""

    @pytest.mark.asyncio
    async def test_sync_backend_window_closed_runs_consolidation(self, memory, request_context):
        """A closed window on the sync backend logs a warning and runs
        consolidation anyway — never raises DeferOperation."""
        bank_id = await _make_bank(memory, request_context, "syncwin")
        async with memory._pool.acquire() as conn:
            await _insert_fact(conn, bank_id)

        start_t, end_t = _closed_window_times()
        with _temporary_window(start_t, end_t, "UTC"):
            result = await memory._handle_consolidation({"operation_id": str(uuid.uuid4()), "bank_id": bank_id})

        assert isinstance(result, dict)
        assert result.get("skipped") is not True
        # The mock-LLM run reports processed memories >= 1 (the fact we seeded).
        assert result.get("memories_processed", 0) >= 1

    @pytest.mark.asyncio
    async def test_sync_backend_extension_rejection_skips(self, memory, request_context):
        """An extension rejection on the sync backend completes as skipped
        (respecting the policy) instead of deferring into a queue that does not
        exist."""

        class RejectingValidator:
            async def validate_consolidate(self, ctx):
                from hindsight_api.extensions.operation_validator import ValidationResult

                return ValidationResult(allowed=False, reason="maintenance window")

        bank_id = await _make_bank(memory, request_context, "syncskip")
        memory._operation_validator = RejectingValidator()
        try:
            result = await memory._handle_consolidation({"operation_id": str(uuid.uuid4()), "bank_id": bank_id})
        finally:
            memory._operation_validator = None

        assert isinstance(result, dict)
        assert result.get("skipped") is True
        assert "maintenance window" in result.get("reason", "")
