"""
Tests for the worker releasing operations it still owns when a task stops
running without reaching its terminal-marking code.

Three paths previously stranded rows in status='processing' forever:
- asyncio.CancelledError escaping _execute_task_inner (BaseException, so no
  `except Exception` catches it),
- _mark_failed itself raising (it is a DB write),
- shutdown_graceful cancelling in-flight tasks and returning without any
  DB reconciliation.

See issue #3228.
"""

import asyncio
import contextlib
import json
import uuid

import pytest
import pytest_asyncio


async def _ensure_bank(pool, bank_id: str) -> None:
    """Upsert a minimal bank row so FK on async_operations passes."""
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


# Use loadgroup to ensure these tests run in the same worker
# since they share database state
pytestmark = pytest.mark.xdist_group("worker_tests")


@pytest_asyncio.fixture
async def backend(pg0_db_url):
    """Create a DatabaseBackend for worker tests."""
    from hindsight_api.engine.db import create_database_backend
    from hindsight_api.pg0 import resolve_database_url

    resolved_url = await resolve_database_url(pg0_db_url)

    b = create_database_backend("postgresql")
    await b.initialize(resolved_url, min_size=2, max_size=10, command_timeout=30)
    yield b
    await b.shutdown()


@pytest_asyncio.fixture
async def pool(backend):
    """Expose the raw asyncpg pool from the backend for direct DB access in tests."""
    yield backend.get_pool()


@pytest_asyncio.fixture
async def clean_operations(pool):
    """Clean up async_operations table before and after tests."""
    await pool.execute("DELETE FROM async_operations WHERE status = 'pending'")
    yield
    await pool.execute("DELETE FROM async_operations WHERE bank_id LIKE 'test-worker-%'")


async def _insert_pending(pool, bank_id: str) -> uuid.UUID:
    """Insert one claimable pending operation, returning its id."""
    op_id = uuid.uuid4()
    payload = json.dumps({"type": "test_task", "bank_id": bank_id, "operation_id": str(op_id)})
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
        VALUES ($1, $2, 'test', 'pending', $3::jsonb)
        """,
        op_id,
        bank_id,
        payload,
    )
    return op_id


async def _fetch_row(pool, op_id):
    return await pool.fetchrow(
        "SELECT status, worker_id, claimed_at, retry_count FROM async_operations WHERE operation_id = $1",
        op_id,
    )


async def _wait_for_status(pool, op_id, status: str, timeout: float = 2.0):
    """Poll until the row reaches `status` or timeout; return the final row."""
    deadline = asyncio.get_event_loop().time() + timeout
    row = await _fetch_row(pool, op_id)
    while row["status"] != status and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
        row = await _fetch_row(pool, op_id)
    return row


class TestTaskRelease:
    @pytest.mark.asyncio
    async def test_cancelled_task_returns_operation_to_pending(self, pool, backend, clean_operations):
        """A cancelled in-flight task must not leave its row 'processing'."""
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        op_id = await _insert_pending(pool, bank_id)

        started = asyncio.Event()
        block = asyncio.Event()

        async def blocking_executor(task_dict):
            started.set()
            await block.wait()

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-release-worker",
            executor=blocking_executor,
        )

        claimed = await poller.claim_batch()
        ours = [t for t in claimed if t.operation_id == str(op_id)]
        assert len(ours) == 1, "test operation should be claimed"

        row = await _fetch_row(pool, op_id)
        assert row["status"] == "processing"
        assert row["worker_id"] == "test-release-worker"

        await poller.execute_task(ours[0])
        await asyncio.wait_for(started.wait(), timeout=5)

        bg_task = poller._active_tasks[str(op_id)].bg_task
        bg_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await bg_task

        row = await _wait_for_status(pool, op_id, "pending")
        assert row["status"] == "pending", "cancelled task must release its row"
        assert row["worker_id"] is None
        assert row["claimed_at"] is None
        assert row["retry_count"] == 1, "an interrupted run counts against the retry budget"

    @pytest.mark.asyncio
    async def test_failed_terminal_write_returns_operation_to_pending(self, pool, backend, clean_operations):
        """If _mark_failed itself raises, the row must be released, not stranded."""
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        op_id = await _insert_pending(pool, bank_id)

        async def failing_executor(task_dict):
            raise RuntimeError("executor blew up")

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-release-worker",
            executor=failing_executor,
        )

        async def broken_mark_failed(operation_id, error_message, schema):
            raise RuntimeError("pool exhausted")

        poller._mark_failed = broken_mark_failed

        claimed = await poller.claim_batch()
        ours = [t for t in claimed if t.operation_id == str(op_id)]
        assert len(ours) == 1

        await poller.execute_task(ours[0])

        row = await _wait_for_status(pool, op_id, "pending")
        assert row["status"] == "pending", "a failed terminal write must not strand the row"
        assert row["worker_id"] is None
        assert row["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_shutdown_graceful_releases_inflight_operations(self, pool, backend, clean_operations):
        """shutdown_graceful must not strand rows it cancelled."""
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        op_id = await _insert_pending(pool, bank_id)

        started = asyncio.Event()
        block = asyncio.Event()

        async def blocking_executor(task_dict):
            started.set()
            await block.wait()

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-release-worker",
            executor=blocking_executor,
        )

        claimed = await poller.claim_batch()
        ours = [t for t in claimed if t.operation_id == str(op_id)]
        assert len(ours) == 1

        await poller.execute_task(ours[0])
        await asyncio.wait_for(started.wait(), timeout=5)

        await poller.shutdown_graceful(timeout=0.1)

        row = await _wait_for_status(pool, op_id, "pending")
        assert row["status"] == "pending", "shutdown must return in-flight rows to pending"
        assert row["worker_id"] is None
