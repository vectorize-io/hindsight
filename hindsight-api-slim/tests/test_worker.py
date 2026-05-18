"""
Tests for the distributed worker system.

Tests cover:
- BrokerTaskBackend task submission and storage
- WorkerPoller task claiming with FOR UPDATE SKIP LOCKED
- Concurrent workers claiming different tasks (no duplicates)
- Task completion and failure handling
- Retry mechanism
- Worker decommissioning
"""

import asyncio
import json
import uuid

import pytest
import pytest_asyncio

from hindsight_api.engine.task_backend import BrokerTaskBackend, SyncTaskBackend, WorkerTaskBackend


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
    """Clean up async_operations table before and after tests.

    We must clean ALL pending operations (not just test-worker-* prefixed ones)
    because WorkerPoller.claim_batch scans the entire schema for pending tasks.
    Stale operations left by other tests (e.g. consolidation) cause spurious
    failures when the poller picks them up unexpectedly.
    """
    await pool.execute("DELETE FROM async_operations WHERE status = 'pending'")
    yield
    await pool.execute(
        "DELETE FROM async_operations WHERE bank_id LIKE 'test-worker-%' OR bank_id LIKE 'test_worker_%'"
    )


def test_all_operation_types_have_slot_reservation_config():
    """Every operation_type used in memory_engine must be listed in
    WORKER_SLOT_RESERVATION_TYPES so it can be reserved via env var.

    If this test fails, a new operation_type was added to memory_engine.py
    without a corresponding entry in config.WORKER_SLOT_RESERVATION_TYPES.
    Add the new type there (single line) and it will automatically get an
    env var, config field, and validation.
    """
    import ast
    import pathlib

    from hindsight_api.config import WORKER_SLOT_RESERVATION_TYPES

    # Parse memory_engine.py and extract all operation_type="..." string values
    engine_path = pathlib.Path(__file__).parent.parent / "hindsight_api" / "engine" / "memory_engine.py"
    tree = ast.parse(engine_path.read_text())

    operation_types_in_code: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "operation_type" and isinstance(node.value, ast.Constant):
            operation_types_in_code.add(node.value.value)

    missing = operation_types_in_code - set(WORKER_SLOT_RESERVATION_TYPES.keys())
    assert not missing, (
        f"Operation types {missing} are used in memory_engine.py but missing from "
        f"config.WORKER_SLOT_RESERVATION_TYPES. Add them there so they can be "
        f"reserved via env var."
    )


class TestBrokerTaskBackend:
    """Tests for BrokerTaskBackend task storage."""

    @pytest.mark.asyncio
    async def test_submit_task_updates_existing_operation(self, pool, clean_operations):
        """Test that submit_task updates task_payload for existing operations."""
        # Create an operation record first
        operation_id = uuid.uuid4()
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status)
            VALUES ($1, $2, 'test_operation', 'pending')
            """,
            operation_id,
            bank_id,
        )

        # Submit task with same operation_id
        backend = BrokerTaskBackend(pool_getter=lambda: pool)
        await backend.initialize()

        task_dict = {
            "operation_id": str(operation_id),
            "type": "test_task",
            "bank_id": bank_id,
            "data": {"key": "value"},
        }
        await backend.submit_task(task_dict)

        # Verify task_payload was stored
        row = await pool.fetchrow(
            "SELECT task_payload, status FROM async_operations WHERE operation_id = $1",
            operation_id,
        )
        assert row is not None
        assert row["status"] == "pending"
        payload = json.loads(row["task_payload"])
        assert payload["type"] == "test_task"
        assert payload["data"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_submit_task_creates_new_operation(self, pool, clean_operations):
        """Test that submit_task creates new operation when no operation_id provided."""
        backend = BrokerTaskBackend(pool_getter=lambda: pool)
        await backend.initialize()

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        task_dict = {
            "type": "access_count_update",
            "bank_id": bank_id,
            "node_ids": ["node1", "node2"],
        }
        await backend.submit_task(task_dict)

        # Verify new operation was created
        row = await pool.fetchrow(
            "SELECT operation_type, status, task_payload FROM async_operations WHERE bank_id = $1",
            bank_id,
        )
        assert row is not None
        assert row["operation_type"] == "access_count_update"
        assert row["status"] == "pending"
        payload = json.loads(row["task_payload"])
        assert payload["node_ids"] == ["node1", "node2"]

    @pytest.mark.asyncio
    async def test_submit_task_preserves_existing_payload(self, pool, clean_operations):
        """Callers now INSERT task_payload atomically, then call submit_task as a
        no-op for the BrokerTaskBackend path. submit_task must not overwrite a
        payload that is already set, otherwise a stale updated_at bump on a
        possibly-already-processing row reintroduces noise the fix aimed to remove.
        """
        operation_id = uuid.uuid4()
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        original_payload = {"type": "test_task", "bank_id": bank_id, "version": "inserted"}
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'test_operation', 'pending', $3::jsonb)
            """,
            operation_id,
            bank_id,
            json.dumps(original_payload),
        )

        row_before = await pool.fetchrow(
            "SELECT updated_at FROM async_operations WHERE operation_id = $1",
            operation_id,
        )

        backend = BrokerTaskBackend(pool_getter=lambda: pool)
        await backend.initialize()

        await backend.submit_task(
            {
                "operation_id": str(operation_id),
                "type": "test_task",
                "bank_id": bank_id,
                "version": "resubmitted",
            }
        )

        row_after = await pool.fetchrow(
            "SELECT task_payload, updated_at FROM async_operations WHERE operation_id = $1",
            operation_id,
        )
        payload = json.loads(row_after["task_payload"])
        assert payload["version"] == "inserted", "submit_task must not overwrite an existing payload"
        assert row_after["updated_at"] == row_before["updated_at"], (
            "submit_task must not bump updated_at when payload was already set"
        )


class TestWorkerPoller:
    """Tests for WorkerPoller task claiming and execution."""

    @pytest.mark.asyncio
    async def test_claim_batch_claims_pending_tasks(self, pool, backend, clean_operations):
        """Test that claim_batch claims pending tasks with task_payload."""
        from hindsight_api.worker import WorkerPoller

        # Create some pending tasks
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        for i in range(3):
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "index": i, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
                VALUES ($1, $2, 'test', 'pending', $3::jsonb)
                """,
                op_id,
                bank_id,
                payload,
            )

        # Create poller and claim tasks
        executed_tasks = []

        async def mock_executor(task_dict):
            executed_tasks.append(task_dict)

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=mock_executor,
        )

        claimed = await poller.claim_batch()
        # Filter to our bank — parallel tests may contribute other claims.
        my_claims = [c for c in claimed if c.task_dict.get("bank_id") == bank_id]
        assert len(my_claims) == 3, f"Expected 3 claims for our bank, got {len(my_claims)}"

        # ClaimedTask objects have operation_id, task_dict, schema attributes
        for task in my_claims:
            assert task.operation_id is not None
            assert task.task_dict is not None

        # Verify tasks are marked as processing with worker_id
        rows = await pool.fetch(
            "SELECT status, worker_id FROM async_operations WHERE bank_id = $1",
            bank_id,
        )
        for row in rows:
            assert row["status"] == "processing"
            assert row["worker_id"] == "test-worker-1"

    @pytest.mark.asyncio
    async def test_claim_batch_respects_max_slots(self, pool, backend, clean_operations):
        """Test that claim_batch respects the max_slots limit."""
        from hindsight_api.worker import WorkerPoller

        # Create 10 pending tasks
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        for i in range(10):
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "index": i, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
                VALUES ($1, $2, 'test', 'pending', $3::jsonb)
                """,
                op_id,
                bank_id,
                payload,
            )

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=lambda x: None,
            max_slots=3,  # Limit to 3 concurrent tasks
            slot_reservations={},  # No reservations; all 3 slots available as shared pool
        )

        claimed = await poller.claim_batch()
        assert len(claimed) == 3

    @pytest.mark.asyncio
    async def test_execute_task_executor_marks_completed(self, pool, backend, clean_operations):
        """Test that executor's status marking is preserved by the poller.

        The executor (MemoryEngine.execute_task) handles marking operations as completed/failed.
        The poller should NOT override those status updates.
        """
        from hindsight_api.worker import WorkerPoller
        from hindsight_api.worker.poller import ClaimedTask

        # Create a pending task
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        op_id = uuid.uuid4()
        payload = json.dumps({"type": "test_task", "operation_id": str(op_id), "bank_id": bank_id})
        await _ensure_bank(pool, bank_id)
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id)
            VALUES ($1, $2, 'test', 'processing', $3::jsonb, 'test-worker-1')
            """,
            op_id,
            bank_id,
            payload,
        )

        executed = []

        async def mock_executor(task_dict):
            """Executor that marks its own status as completed (like MemoryEngine.execute_task)."""
            executed.append(task_dict)
            await pool.execute(
                """
                UPDATE async_operations
                SET status = 'completed', completed_at = now(), updated_at = now()
                WHERE operation_id = $1
                """,
                op_id,
            )

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=mock_executor,
        )

        # Execute the task (fire-and-forget)
        task_dict = json.loads(payload)
        claimed_task = ClaimedTask(operation_id=str(op_id), task_dict=task_dict, schema=None)
        await poller.execute_task(claimed_task)

        # Wait for background task to complete
        completed = await poller.wait_for_active_tasks(timeout=5.0)
        assert completed, "Task did not complete within timeout"
        assert len(executed) == 1

        # Verify task is marked as completed (by executor, not overridden by poller)
        row = await pool.fetchrow(
            "SELECT status, completed_at FROM async_operations WHERE operation_id = $1",
            op_id,
        )
        assert row["status"] == "completed"
        assert row["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_executor_exception_triggers_retry(self, pool, backend, clean_operations):
        """Test that exceptions from the executor trigger _retry_or_fail (not a crash).

        When the executor re-raises an exception (as MemoryEngine.execute_task does for
        retryable task failures), the poller calls _retry_or_fail, which resets the task
        back to 'pending' and increments retry_count so it can be reclaimed.

        This is the fix for the consolidation deadlock: previously submit_task was called
        with only a task_payload update, leaving status='processing' forever.
        """
        from hindsight_api.worker import WorkerPoller
        from hindsight_api.worker.poller import ClaimedTask

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        op_id = uuid.uuid4()
        payload = json.dumps({"type": "consolidation", "operation_id": str(op_id), "bank_id": bank_id})
        await _ensure_bank(pool, bank_id)
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id, claimed_at)
            VALUES ($1, $2, 'consolidation', 'processing', $3::jsonb, 'test-worker-1', now())
            """,
            op_id,
            bank_id,
            payload,
        )

        from datetime import datetime, timezone

        from hindsight_api.worker.exceptions import RetryTaskAt

        async def failing_executor(task_dict):
            raise RetryTaskAt(retry_at=datetime.now(timezone.utc), message="TimeoutError during recall")

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=failing_executor,
        )

        task_dict = json.loads(payload)
        claimed_task = ClaimedTask(operation_id=str(op_id), task_dict=task_dict, schema=None)
        await poller.execute_task(claimed_task)

        completed = await poller.wait_for_active_tasks(timeout=5.0)
        assert completed, "Task did not complete within timeout"

        # Task must be reset to 'pending' with worker_id/claimed_at cleared — not left as
        # 'processing', which would cause a permanent deadlock via the NOT EXISTS guard.
        row = await pool.fetchrow(
            "SELECT status, worker_id, claimed_at, retry_count FROM async_operations WHERE operation_id = $1",
            op_id,
        )
        assert row["status"] == "pending", (
            f"REGRESSION: Task status is '{row['status']}' instead of 'pending'. "
            "A task stuck in 'processing' after a retry causes a consolidation deadlock."
        )
        assert row["worker_id"] is None, "worker_id must be cleared on retry"
        assert row["claimed_at"] is None, "claimed_at must be cleared on retry"
        assert row["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_executor_exception_marks_failed_immediately(self, pool, backend, clean_operations):
        """Test that a plain exception (not RetryTaskAt) permanently marks a task as 'failed'.

        With the task-owned retry model, plain exceptions are non-retryable — the poller
        marks them as 'failed' immediately. Tasks that want to be retried must raise RetryTaskAt.
        """
        from hindsight_api.worker import WorkerPoller
        from hindsight_api.worker.poller import ClaimedTask

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        op_id = uuid.uuid4()
        payload = json.dumps({"type": "consolidation", "operation_id": str(op_id), "bank_id": bank_id})
        await _ensure_bank(pool, bank_id)
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id, claimed_at, retry_count)
            VALUES ($1, $2, 'consolidation', 'processing', $3::jsonb, 'test-worker-1', now(), 0)
            """,
            op_id,
            bank_id,
            payload,
        )

        async def failing_executor(task_dict):
            raise ValueError("Non-retryable error")

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=failing_executor,
        )

        task_dict = json.loads(payload)
        claimed_task = ClaimedTask(operation_id=str(op_id), task_dict=task_dict, schema=None)
        await poller.execute_task(claimed_task)

        completed = await poller.wait_for_active_tasks(timeout=5.0)
        assert completed, "Task did not complete within timeout"

        row = await pool.fetchrow(
            "SELECT status, error_message, retry_count FROM async_operations WHERE operation_id = $1",
            op_id,
        )
        assert row["status"] == "failed", f"Expected 'failed' for plain exception, got '{row['status']}'"
        assert row["error_message"] is not None
        assert row["retry_count"] == 0  # not incremented; plain exception = immediate fail

    @pytest.mark.asyncio
    async def test_executor_failed_status_not_overridden(self, pool, backend, clean_operations):
        """REGRESSION TEST: Verify poller does NOT overwrite executor's 'failed' status to 'completed'.

        This test covers the non-retryable failure path (e.g., file_convert_retain):
        1. Executor catches an internal error, marks the operation as 'failed' in the DB
        2. Executor returns normally (does NOT re-raise) — so no exception reaches the poller
        3. The poller must NOT overwrite the 'failed' status to 'completed'

        Retryable failures re-raise instead (see test_executor_exception_triggers_retry).
        """
        from hindsight_api.worker import WorkerPoller
        from hindsight_api.worker.poller import ClaimedTask

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        op_id = uuid.uuid4()
        payload = json.dumps({"type": "test_task", "operation_id": str(op_id), "bank_id": bank_id})
        await _ensure_bank(pool, bank_id)
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id)
            VALUES ($1, $2, 'test', 'processing', $3::jsonb, 'test-worker-1')
            """,
            op_id,
            bank_id,
            payload,
        )

        async def executor_that_marks_failed(task_dict):
            """Simulates MemoryEngine.execute_task behavior on internal error.

            The executor catches the error, marks the operation as 'failed',
            and returns normally (does NOT re-raise the exception).
            """
            # Simulate internal failure handling (like MemoryEngine._mark_operation_failed)
            await pool.execute(
                """
                UPDATE async_operations
                SET status = 'failed', error_message = $2, completed_at = now(), updated_at = now()
                WHERE operation_id = $1
                """,
                op_id,
                "Simulated conversion error: file format not supported",
            )
            # Returns normally - this is the key: executor does NOT re-raise

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=executor_that_marks_failed,
        )

        task_dict = json.loads(payload)
        claimed_task = ClaimedTask(operation_id=str(op_id), task_dict=task_dict, schema=None)
        await poller.execute_task(claimed_task)

        completed = await poller.wait_for_active_tasks(timeout=5.0)
        assert completed, "Task did not complete within timeout"

        # THE KEY ASSERTION: Status must be 'failed', NOT 'completed'
        row = await pool.fetchrow(
            "SELECT status, error_message FROM async_operations WHERE operation_id = $1",
            op_id,
        )
        assert row["status"] == "failed", (
            f"REGRESSION: Poller overwrote executor's 'failed' status to '{row['status']}'. "
            "The poller must not override status set by the executor."
        )
        assert "Simulated conversion error" in row["error_message"]

    @pytest.mark.asyncio
    async def test_executor_defer_requeues_without_bumping_retry_count(self, pool, backend, clean_operations):
        """DeferOperation requeues the task without counting as a retry.

        Unlike RetryTaskAt (failure-driven), DeferOperation is intentional
        backpressure: the row goes back to 'pending' with next_retry_at set,
        but retry_count is unchanged and error_message stays NULL.
        """
        from datetime import datetime, timedelta, timezone

        from hindsight_api.worker import WorkerPoller
        from hindsight_api.worker.exceptions import DeferOperation
        from hindsight_api.worker.poller import ClaimedTask

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        op_id = uuid.uuid4()
        payload = json.dumps({"type": "retain", "operation_id": str(op_id), "bank_id": bank_id})
        await _ensure_bank(pool, bank_id)
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id, claimed_at, retry_count)
            VALUES ($1, $2, 'retain', 'processing', $3::jsonb, 'test-worker-1', now(), 0)
            """,
            op_id,
            bank_id,
            payload,
        )

        defer_until = datetime.now(timezone.utc) + timedelta(minutes=5)

        async def deferring_executor(task_dict):
            raise DeferOperation(exec_date=defer_until, reason="upstream quota window not yet open")

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=deferring_executor,
        )

        task_dict = json.loads(payload)
        claimed_task = ClaimedTask(operation_id=str(op_id), task_dict=task_dict, schema=None)
        await poller.execute_task(claimed_task)

        completed = await poller.wait_for_active_tasks(timeout=5.0)
        assert completed, "Task did not complete within timeout"

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
        assert row["next_retry_at"] is not None
        # exec_date should round-trip; allow 1s slack for db precision
        assert abs((row["next_retry_at"] - defer_until).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_deferred_task_not_picked_up_until_exec_date(self, pool, backend, clean_operations):
        """A deferred task is invisible to claim_batch until next_retry_at <= NOW()."""
        from datetime import datetime, timedelta, timezone

        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        op_id = uuid.uuid4()
        payload = json.dumps({"type": "retain", "operation_id": str(op_id), "bank_id": bank_id})
        await _ensure_bank(pool, bank_id)
        future = datetime.now(timezone.utc) + timedelta(minutes=5)
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, next_retry_at)
            VALUES ($1, $2, 'retain', 'pending', $3::jsonb, $4)
            """,
            op_id,
            bank_id,
            payload,
            future,
        )

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=lambda x: None,
        )

        claimed = await poller.claim_batch()
        assert all(c.operation_id != str(op_id) for c in claimed), "deferred task must not be claimed before exec_date"

        # Move next_retry_at into the past — task becomes claimable.
        await pool.execute(
            "UPDATE async_operations SET next_retry_at = now() - interval '1 minute' WHERE operation_id = $1",
            op_id,
        )
        claimed = await poller.claim_batch()
        assert any(c.operation_id == str(op_id) for c in claimed), "task must be claimed once next_retry_at has passed"

    @pytest.mark.asyncio
    async def test_defer_operation_exported_from_extensions(self):
        """DeferOperation must be importable from hindsight_api.extensions for extension authors."""
        from hindsight_api.extensions import DeferOperation as DeferFromExtensions
        from hindsight_api.worker.exceptions import DeferOperation as DeferFromWorker

        assert DeferFromExtensions is DeferFromWorker

    @pytest.mark.asyncio
    async def test_extension_validate_retain_defer_propagates_to_poller(self, pool, backend, clean_operations):
        """An OperationValidatorExtension that raises DeferOperation in validate_retain
        causes the worker to requeue the task at the requested exec_date.

        Mimics the MemoryEngine flow: the executor calls validate_retain before doing
        any real work, the exception bubbles up to the poller, which defers the row.
        """
        from datetime import datetime, timedelta, timezone

        from hindsight_api.extensions import (
            DeferOperation,
            OperationValidatorExtension,
            RecallContext,
            ReflectContext,
            RetainContext,
            ValidationResult,
        )
        from hindsight_api.worker import WorkerPoller
        from hindsight_api.worker.poller import ClaimedTask

        defer_until = datetime.now(timezone.utc) + timedelta(minutes=10)

        class DeferringValidator(OperationValidatorExtension):
            def __init__(self):
                super().__init__({})

            async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
                raise DeferOperation(exec_date=defer_until, reason="quota window closed")

            async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
                return ValidationResult.accept()

            async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
                return ValidationResult.accept()

        validator = DeferringValidator()

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        op_id = uuid.uuid4()
        payload = json.dumps({"type": "retain", "operation_id": str(op_id), "bank_id": bank_id})
        await _ensure_bank(pool, bank_id)
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id, claimed_at, retry_count)
            VALUES ($1, $2, 'retain', 'processing', $3::jsonb, 'test-worker-1', now(), 0)
            """,
            op_id,
            bank_id,
            payload,
        )

        async def executor_calling_validator(task_dict):
            ctx = RetainContext(
                bank_id=task_dict["bank_id"],
                contents=[],
                request_context=None,  # type: ignore[arg-type]
            )
            await validator.validate_retain(ctx)

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=executor_calling_validator,
        )

        task_dict = json.loads(payload)
        claimed_task = ClaimedTask(operation_id=str(op_id), task_dict=task_dict, schema=None)
        await poller.execute_task(claimed_task)

        completed = await poller.wait_for_active_tasks(timeout=5.0)
        assert completed, "Task did not complete within timeout"

        row = await pool.fetchrow(
            "SELECT status, worker_id, claimed_at, retry_count, error_message, next_retry_at "
            "FROM async_operations WHERE operation_id = $1",
            op_id,
        )
        assert row["status"] == "pending"
        assert row["worker_id"] is None
        assert row["claimed_at"] is None
        assert row["retry_count"] == 0
        assert row["error_message"] is None
        assert abs((row["next_retry_at"] - defer_until).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_memory_engine_execute_task_passes_through_defer_operation(self, memory):
        """Regression test: MemoryEngine.execute_task must pass DeferOperation
        through to the worker poller unchanged (same as RetryTaskAt).

        Prior to this fix, the generic ``except Exception`` in execute_task
        converted any exception — including DeferOperation — into a
        ``RetryTaskAt(60s)``, which bumps retry_count and writes an
        error_message. That silently broke the "defer is not a failure"
        semantics added in #1105: a task scheduled hours out (e.g. by a
        backpressure-aware extension waiting for a quota window to open)
        would instead come back in 60 seconds with retry_count bumped.

        Verify by injecting a validator that raises DeferOperation during
        validate_retain and confirming the outer exception type that
        escapes execute_task is DeferOperation, not RetryTaskAt.
        """
        from datetime import datetime, timedelta, timezone

        from hindsight_api.extensions import (
            DeferOperation,
            OperationValidatorExtension,
            RecallContext,
            ReflectContext,
            RetainContext,
            ValidationResult,
        )
        from hindsight_api.worker.exceptions import DeferOperation as DeferOpFromExceptions
        from hindsight_api.worker.exceptions import RetryTaskAt

        defer_until = datetime.now(timezone.utc) + timedelta(hours=6)

        class QuotaDeferringValidator(OperationValidatorExtension):
            def __init__(self):
                super().__init__({})

            async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
                raise DeferOperation(exec_date=defer_until, reason="quota_window_closed")

            async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
                return ValidationResult.accept()

            async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
                return ValidationResult.accept()

        # Attach the validator to the MemoryEngine so it's invoked from
        # inside retain_batch_async → _validate_operation.
        memory._operation_validator = QuotaDeferringValidator()

        # Set up a bank the task handler can address.
        bank_id = f"test-defer-passthrough-{uuid.uuid4().hex[:8]}"
        operation_id = uuid.uuid4()
        # execute_task short-circuits if the async_operations row is missing
        # (treats it as cancelled), so insert a pending row first.
        async with memory._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO banks (bank_id) VALUES ($1) ON CONFLICT DO NOTHING",
                bank_id,
            )
            await conn.execute(
                """
                INSERT INTO async_operations
                    (operation_id, bank_id, operation_type, status, task_payload)
                VALUES ($1, $2, 'retain', 'pending', $3::jsonb)
                """,
                operation_id,
                bank_id,
                json.dumps({"type": "batch_retain", "bank_id": bank_id}),
            )

        task_dict = {
            "type": "batch_retain",
            "bank_id": bank_id,
            "contents": [{"content": "x"}],
            "operation_id": str(operation_id),
            "_tenant_id": "default",
        }

        # Contract: DeferOperation must escape execute_task intact.
        # Must NOT be converted to RetryTaskAt. Must NOT be swallowed.
        with pytest.raises(DeferOpFromExceptions) as exc_info:
            await memory.execute_task(task_dict)

        assert exc_info.value.reason == "quota_window_closed"
        assert abs((exc_info.value.exec_date - defer_until).total_seconds()) < 1

        # Defensive: confirm it wasn't a RetryTaskAt masquerading as Defer.
        assert not isinstance(exc_info.value, RetryTaskAt)

    @pytest.mark.asyncio
    async def test_claim_batch_skips_consolidation_when_same_bank_processing(self, pool, backend, clean_operations):
        """Test that pending consolidation is skipped if same bank has one processing."""
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        # Create a processing consolidation for bank
        processing_op_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id)
            VALUES ($1, $2, 'consolidation', 'processing', $3::jsonb, 'other-worker')
            """,
            processing_op_id,
            bank_id,
            json.dumps({"type": "consolidation", "bank_id": bank_id}),
        )

        # Create a pending consolidation for same bank (should be skipped)
        pending_op_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'consolidation', 'pending', $3::jsonb)
            """,
            pending_op_id,
            bank_id,
            json.dumps({"type": "consolidation", "bank_id": bank_id}),
        )

        # Create a pending consolidation for different bank (should be claimed)
        other_bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, other_bank_id)
        other_op_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'consolidation', 'pending', $3::jsonb)
            """,
            other_op_id,
            other_bank_id,
            json.dumps({"type": "consolidation", "bank_id": other_bank_id}),
        )

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=lambda x: None,
        )

        claimed = await poller.claim_batch()

        # Filter to banks this test created — other test files running in
        # parallel may have pending async_operations that the poller would
        # legitimately claim; we only care about our own rows here.
        test_banks = {bank_id, other_bank_id}
        my_claims = [c for c in claimed if c.task_dict.get("bank_id") in test_banks]

        # Should only claim the consolidation for the other bank
        assert len(my_claims) == 1, f"Expected 1 claim for our banks, got {len(my_claims)}: {my_claims}"
        assert my_claims[0].operation_id == str(other_op_id)
        assert my_claims[0].task_dict["bank_id"] == other_bank_id

        # Verify the pending consolidation for first bank is still pending
        row = await pool.fetchrow(
            "SELECT status, worker_id FROM async_operations WHERE operation_id = $1",
            pending_op_id,
        )
        assert row["status"] == "pending"
        assert row["worker_id"] is None

    @pytest.mark.asyncio
    async def test_claim_batch_allows_non_consolidation_when_consolidation_processing(self, pool, backend, clean_operations):
        """Test that non-consolidation tasks are still claimed even if consolidation is processing."""
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        # Create a processing consolidation for bank
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id)
            VALUES ($1, $2, 'consolidation', 'processing', $3::jsonb, 'other-worker')
            """,
            uuid.uuid4(),
            bank_id,
            json.dumps({"type": "consolidation", "bank_id": bank_id}),
        )

        # Create a pending retain task for same bank (should be claimed)
        retain_op_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'retain', 'pending', $3::jsonb)
            """,
            retain_op_id,
            bank_id,
            json.dumps({"type": "batch_retain", "bank_id": bank_id}),
        )

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=lambda x: None,
        )

        claimed = await poller.claim_batch()

        # Should claim the retain task (non-consolidation tasks are unaffected)
        assert len(claimed) == 1
        assert claimed[0].operation_id == str(retain_op_id)


class TestWorkerRecovery:
    """Tests for worker task recovery on startup."""

    @pytest.mark.asyncio
    async def test_recover_own_tasks_resets_processing_to_pending(self, pool, backend, clean_operations):
        """Test that recover_own_tasks resets processing tasks back to pending."""
        from hindsight_api.worker import WorkerPoller

        # Create tasks that were being processed by this worker (simulating a crash)
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        worker_id = "crashed-worker"
        task_ids = []

        for i in range(3):
            op_id = uuid.uuid4()
            task_ids.append(op_id)
            payload = json.dumps({"type": "test_task", "index": i, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id, claimed_at)
                VALUES ($1, $2, 'test', 'processing', $3::jsonb, $4, now())
                """,
                op_id,
                bank_id,
                payload,
                worker_id,
            )

        # Create poller with same worker_id and call recover
        poller = WorkerPoller(
            backend=backend,
            worker_id=worker_id,
            executor=lambda x: None,
        )

        recovered_count = await poller.recover_own_tasks()
        assert recovered_count == 3

        # Verify all tasks are back to pending with no worker assigned
        rows = await pool.fetch(
            "SELECT status, worker_id, claimed_at FROM async_operations WHERE bank_id = $1",
            bank_id,
        )
        for row in rows:
            assert row["status"] == "pending"
            assert row["worker_id"] is None
            assert row["claimed_at"] is None

    @pytest.mark.asyncio
    async def test_recover_own_tasks_does_not_affect_other_workers(self, pool, backend, clean_operations):
        """Test that recover_own_tasks only affects tasks from the same worker_id."""
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        # Create tasks for worker-1 (the one that will recover)
        for i in range(2):
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "index": i, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id)
                VALUES ($1, $2, 'test', 'processing', $3::jsonb, 'worker-1')
                """,
                op_id,
                bank_id,
                payload,
            )

        # Create tasks for worker-2 (should not be affected)
        for i in range(2):
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "index": i + 10, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id)
                VALUES ($1, $2, 'test', 'processing', $3::jsonb, 'worker-2')
                """,
                op_id,
                bank_id,
                payload,
            )

        # Worker-1 recovers its tasks
        poller = WorkerPoller(
            backend=backend,
            worker_id="worker-1",
            executor=lambda x: None,
        )

        recovered_count = await poller.recover_own_tasks()
        assert recovered_count == 2

        # Verify worker-1 tasks are released
        worker1_rows = await pool.fetch(
            "SELECT status, worker_id FROM async_operations WHERE bank_id = $1 AND worker_id IS NULL",
            bank_id,
        )
        assert len(worker1_rows) == 2

        # Verify worker-2 tasks are unaffected
        worker2_rows = await pool.fetch(
            "SELECT status, worker_id FROM async_operations WHERE bank_id = $1 AND worker_id = 'worker-2'",
            bank_id,
        )
        assert len(worker2_rows) == 2
        for row in worker2_rows:
            assert row["status"] == "processing"

    @pytest.mark.asyncio
    async def test_recover_own_tasks_returns_zero_when_no_stale_tasks(self, pool, backend, clean_operations):
        """Test that recover_own_tasks returns 0 when there are no stale tasks."""
        from hindsight_api.worker import WorkerPoller

        poller = WorkerPoller(
            backend=backend,
            worker_id="fresh-worker",
            executor=lambda x: None,
        )

        recovered_count = await poller.recover_own_tasks()
        assert recovered_count == 0


class TestConcurrentWorkers:
    """Tests for concurrent worker task claiming (FOR UPDATE SKIP LOCKED)."""

    @pytest.mark.asyncio
    async def test_concurrent_workers_claim_different_tasks(self, pool, backend, clean_operations):
        """Test that multiple workers claim different tasks (no duplicates)."""
        from hindsight_api.worker import WorkerPoller

        # Create 10 pending tasks
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        task_ids = []
        for i in range(10):
            op_id = uuid.uuid4()
            task_ids.append(op_id)
            payload = json.dumps({"type": "test_task", "index": i, "bank_id": bank_id, "operation_id": str(op_id)})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
                VALUES ($1, $2, 'test', 'pending', $3::jsonb)
                """,
                op_id,
                bank_id,
                payload,
            )

        # Create 3 workers that will claim tasks concurrently. Filter each
        # worker's claims down to our bank — parallel test files may contribute
        # other pending rows that these workers legitimately pick up.
        workers_claimed: dict[str, list[str]] = {"worker-1": [], "worker-2": [], "worker-3": []}

        async def claim_for_worker(worker_id: str):
            poller = WorkerPoller(
                backend=backend,
                worker_id=worker_id,
                executor=lambda x: None,
            )
            claimed = await poller.claim_batch()
            workers_claimed[worker_id] = [
                task.operation_id for task in claimed if task.task_dict.get("bank_id") == bank_id
            ]

        # Run all workers concurrently
        await asyncio.gather(
            claim_for_worker("worker-1"),
            claim_for_worker("worker-2"),
            claim_for_worker("worker-3"),
        )

        # Verify no duplicates - each task claimed by exactly one worker
        all_claimed = workers_claimed["worker-1"] + workers_claimed["worker-2"] + workers_claimed["worker-3"]
        assert len(all_claimed) == len(set(all_claimed)), "Duplicate task claimed by multiple workers!"

        # Verify total claimed equals available tasks (10) for our bank
        assert len(all_claimed) == 10, f"Expected 10 tasks claimed for our bank, got {len(all_claimed)}"

        # Verify each task is assigned to exactly one worker in DB
        rows = await pool.fetch(
            "SELECT operation_id, worker_id FROM async_operations WHERE bank_id = $1",
            bank_id,
        )
        worker_assignments = {str(row["operation_id"]): row["worker_id"] for row in rows}

        # With FOR UPDATE SKIP LOCKED, it's a race condition which workers get tasks.
        # The important invariant is no duplicates and all tasks claimed, which we verified above.
        # Just verify that at least 1 worker got tasks and all tasks have a worker assigned.
        assert len(set(worker_assignments.values())) >= 1, "At least one worker should have claimed tasks"
        assert all(w is not None for w in worker_assignments.values()), "All tasks should have a worker assigned"

    @pytest.mark.asyncio
    async def test_workers_do_not_claim_already_processing_tasks(self, pool, backend, clean_operations):
        """Test that workers skip tasks already being processed by another worker."""
        from hindsight_api.worker import WorkerPoller

        # Create tasks - some pending, some already processing
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        # Create 3 pending tasks
        for i in range(3):
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "index": i, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
                VALUES ($1, $2, 'test', 'pending', $3::jsonb)
                """,
                op_id,
                bank_id,
                payload,
            )

        # Create 2 already-processing tasks owned by another worker
        for i in range(2):
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "index": i + 10, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id)
                VALUES ($1, $2, 'test', 'processing', $3::jsonb, 'other-worker')
                """,
                op_id,
                bank_id,
                payload,
            )

        # New worker should only claim the 3 pending tasks
        poller = WorkerPoller(
            backend=backend,
            worker_id="new-worker",
            executor=lambda x: None,
        )

        claimed = await poller.claim_batch()
        # Filter to this test's bank — parallel tests may contribute other claims.
        my_claims = [c for c in claimed if c.task_dict.get("bank_id") == bank_id]
        assert len(my_claims) == 3, f"Worker should only claim the 3 pending tasks for our bank, got {len(my_claims)}"

        # Verify other worker's tasks are still owned by them
        row = await pool.fetchrow(
            "SELECT COUNT(*) as count FROM async_operations WHERE bank_id = $1 AND worker_id = 'other-worker'",
            bank_id,
        )
        assert row["count"] == 2


class TestWorkerDecommission:
    """Tests for worker decommissioning functionality."""

    @pytest.mark.asyncio
    async def test_decommission_releases_worker_tasks(self, pool, clean_operations):
        """Test that decommissioning a worker releases all its processing tasks."""
        # Create tasks being processed by a worker
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        worker_id = "worker-to-decommission"

        for i in range(5):
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "index": i, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id, claimed_at)
                VALUES ($1, $2, 'test', 'processing', $3::jsonb, $4, now())
                """,
                op_id,
                bank_id,
                payload,
                worker_id,
            )

        # Run decommission
        result = await pool.fetch(
            """
            UPDATE async_operations
            SET status = 'pending', worker_id = NULL, claimed_at = NULL, updated_at = now()
            WHERE worker_id = $1 AND status = 'processing'
            RETURNING operation_id
            """,
            worker_id,
        )

        assert len(result) == 5

        # Verify all tasks are back to pending
        rows = await pool.fetch(
            "SELECT status, worker_id, claimed_at FROM async_operations WHERE bank_id = $1",
            bank_id,
        )
        for row in rows:
            assert row["status"] == "pending"
            assert row["worker_id"] is None
            assert row["claimed_at"] is None

    @pytest.mark.asyncio
    async def test_decommission_does_not_affect_other_workers(self, pool, clean_operations):
        """Test that decommissioning one worker doesn't affect another worker's tasks."""
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        # Create tasks for worker-1
        for i in range(3):
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "index": i, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id)
                VALUES ($1, $2, 'test', 'processing', $3::jsonb, 'worker-1')
                """,
                op_id,
                bank_id,
                payload,
            )

        # Create tasks for worker-2
        for i in range(3):
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "index": i + 10, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id)
                VALUES ($1, $2, 'test', 'processing', $3::jsonb, 'worker-2')
                """,
                op_id,
                bank_id,
                payload,
            )

        # Decommission worker-1 only
        await pool.execute(
            """
            UPDATE async_operations
            SET status = 'pending', worker_id = NULL, claimed_at = NULL
            WHERE worker_id = 'worker-1' AND status = 'processing'
            """
        )

        # Verify worker-1 tasks are released
        worker1_rows = await pool.fetch(
            "SELECT status, worker_id FROM async_operations WHERE bank_id = $1 AND worker_id IS NULL",
            bank_id,
        )
        assert len(worker1_rows) == 3

        # Verify worker-2 tasks are unaffected
        worker2_rows = await pool.fetch(
            "SELECT status, worker_id FROM async_operations WHERE bank_id = $1 AND worker_id = 'worker-2'",
            bank_id,
        )
        assert len(worker2_rows) == 3
        for row in worker2_rows:
            assert row["status"] == "processing"


class TestSyncTaskBackend:
    """Tests for SyncTaskBackend (used in tests and embedded mode)."""

    @pytest.mark.asyncio
    async def test_sync_backend_executes_immediately(self):
        """Test that SyncTaskBackend executes tasks immediately."""
        executed = []

        async def mock_executor(task_dict):
            executed.append(task_dict)

        backend = SyncTaskBackend()
        backend.set_executor(mock_executor)
        await backend.initialize()

        task_dict = {"type": "test", "data": "value"}
        await backend.submit_task(task_dict)

        assert len(executed) == 1
        assert executed[0] == task_dict

    @pytest.mark.asyncio
    async def test_sync_backend_propagates_errors(self):
        """Test that SyncTaskBackend propagates executor errors instead of swallowing them."""

        async def failing_executor(task_dict):
            raise ValueError("Test error")

        backend = SyncTaskBackend()
        backend.set_executor(failing_executor)
        await backend.initialize()

        # Should raise so callers can handle or surface the failure
        with pytest.raises(ValueError, match="Test error"):
            await backend.submit_task({"type": "test"})


class TestWorkerTaskBackend:
    """Tests for WorkerTaskBackend (used by worker processes).

    WorkerTaskBackend.submit_task is a no-op: child tasks spawned during
    execution (e.g. consolidation triggered by retain) are left as pending
    rows in async_operations for the poller to pick up on the next cycle,
    instead of being executed inline which would block the parent task.
    """

    @pytest.mark.asyncio
    async def test_worker_backend_does_not_execute_inline(self):
        """WorkerTaskBackend.submit_task must NOT call the executor."""
        executed = []

        async def mock_executor(task_dict):
            executed.append(task_dict)

        backend = WorkerTaskBackend()
        backend.set_executor(mock_executor)
        await backend.initialize()

        await backend.submit_task({"type": "consolidation", "bank_id": "b1"})

        assert len(executed) == 0, (
            "WorkerTaskBackend must not execute tasks inline — child tasks should be picked up by the poller"
        )

    @pytest.mark.asyncio
    async def test_sync_backend_blocks_parent_on_child_task(self):
        """Demonstrate the blocking problem: with SyncTaskBackend, a parent
        task that calls submit_task for a child is blocked until the child
        completes, because submit_task executes inline.
        """
        execution_order: list[str] = []

        async def mock_executor(task_dict):
            task_type = task_dict.get("type", "unknown")
            execution_order.append(f"{task_type}:start")
            if task_type == "parent":
                # Parent spawns a child via submit_task
                await backend.submit_task({"type": "child", "bank_id": "b1"})
                # This line only runs AFTER the child finishes (blocking)
            execution_order.append(f"{task_type}:end")

        backend = SyncTaskBackend()
        backend.set_executor(mock_executor)
        await backend.initialize()

        await backend.submit_task({"type": "parent", "bank_id": "b1"})

        # With SyncTaskBackend the child runs inline, nested inside the parent
        assert execution_order == [
            "parent:start",
            "child:start",
            "child:end",
            "parent:end",
        ], f"SyncTaskBackend should execute child inline (blocking parent). Got: {execution_order}"

    @pytest.mark.asyncio
    async def test_worker_backend_does_not_block_parent_on_child_task(self):
        """With WorkerTaskBackend, a parent task that calls submit_task for a
        child task is NOT blocked — submit_task is a no-op, so the parent
        completes immediately and the child row stays pending for the poller.
        """
        execution_order: list[str] = []

        async def mock_executor(task_dict):
            task_type = task_dict.get("type", "unknown")
            execution_order.append(f"{task_type}:start")
            if task_type == "parent":
                # Parent spawns a child via submit_task — should be a no-op
                await backend.submit_task({"type": "child", "bank_id": "b1"})
            execution_order.append(f"{task_type}:end")

        backend = WorkerTaskBackend()
        backend.set_executor(mock_executor)
        await backend.initialize()

        # Simulate the poller calling the executor directly for the parent task
        await mock_executor({"type": "parent", "bank_id": "b1"})

        # The child should NOT have been executed — only the parent runs
        assert execution_order == [
            "parent:start",
            "parent:end",
        ], f"WorkerTaskBackend must not execute child tasks inline. Got: {execution_order}"

    @pytest.mark.asyncio
    async def test_worker_backend_child_task_stays_pending_in_db(self, pool, backend, clean_operations):
        """End-to-end: when a worker-executed task spawns a child operation,
        the child row stays as 'pending' in async_operations (not executed inline).
        A subsequent poll cycle can then claim and execute it independently.
        """
        from hindsight_api.worker import WorkerPoller
        from hindsight_api.worker.poller import ClaimedTask

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        # Pre-create the parent operation (as the poller would claim it)
        parent_op_id = uuid.uuid4()
        parent_payload = {
            "type": "batch_retain",
            "operation_id": str(parent_op_id),
            "bank_id": bank_id,
        }
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id, claimed_at)
            VALUES ($1, $2, 'retain', 'processing', $3::jsonb, 'test-worker-1', now())
            """,
            parent_op_id,
            bank_id,
            json.dumps(parent_payload),
        )

        # Pre-create the child operation row (as _submit_async_operation would)
        child_op_id = uuid.uuid4()
        child_payload = {
            "type": "consolidation",
            "operation_id": str(child_op_id),
            "bank_id": bank_id,
        }
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'consolidation', 'pending', $3::jsonb)
            """,
            child_op_id,
            bank_id,
            json.dumps(child_payload),
        )

        # The executor simulates retain: marks parent completed, then calls
        # submit_task for the child (which WorkerTaskBackend should ignore).
        task_backend = WorkerTaskBackend()

        async def executor(task_dict):
            # Mark parent as completed
            await pool.execute(
                "UPDATE async_operations SET status = 'completed', completed_at = now() WHERE operation_id = $1",
                parent_op_id,
            )
            # Trigger child task via submit_task (should be no-op for WorkerTaskBackend)
            await task_backend.submit_task(child_payload)

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=executor,
        )

        # Execute the parent task
        claimed_task = ClaimedTask(
            operation_id=str(parent_op_id),
            task_dict=parent_payload,
            schema=None,
        )
        await poller.execute_task(claimed_task)
        completed = await poller.wait_for_active_tasks(timeout=5.0)
        assert completed, "Parent task did not complete within timeout"

        # Parent should be completed
        parent_row = await pool.fetchrow(
            "SELECT status FROM async_operations WHERE operation_id = $1",
            parent_op_id,
        )
        assert parent_row["status"] == "completed"

        # Child should still be pending (NOT executed inline)
        child_row = await pool.fetchrow(
            "SELECT status, worker_id FROM async_operations WHERE operation_id = $1",
            child_op_id,
        )
        assert child_row["status"] == "pending", (
            f"Child task should remain 'pending' for the next poll cycle, "
            f"but got '{child_row['status']}'. This means it was executed inline, "
            f"blocking the parent task."
        )
        assert child_row["worker_id"] is None

        # Now verify the poller can claim the child on the next cycle
        claimed = await poller.claim_batch()
        child_claimed = [c for c in claimed if c.operation_id == str(child_op_id)]
        assert len(child_claimed) == 1, "Child task should be claimable on next poll"


class TestDynamicTenantDiscovery:
    """Tests for dynamic tenant discovery via TenantExtension."""

    @pytest.mark.asyncio
    async def test_poller_discovers_tenants_dynamically(self, pool, backend, clean_operations):
        """Test that poller calls list_tenants() on each poll cycle."""
        from hindsight_api.extensions.tenant import Tenant, TenantExtension
        from hindsight_api.worker import WorkerPoller

        # Create a mock tenant extension that tracks calls
        class MockTenantExtension(TenantExtension):
            def __init__(self):
                self.list_tenants_calls = 0
                self.tenants_to_return: list[Tenant] = [Tenant(schema="public")]

            async def authenticate(self, context):
                raise NotImplementedError("Not used in this test")

            async def list_tenants(self) -> list[Tenant]:
                self.list_tenants_calls += 1
                return self.tenants_to_return

        mock_extension = MockTenantExtension()

        # Create pending tasks in public schema
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        for i in range(2):
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "index": i, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
                VALUES ($1, $2, 'test', 'pending', $3::jsonb)
                """,
                op_id,
                bank_id,
                payload,
            )

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=lambda x: None,
            tenant_extension=mock_extension,
        )

        # First claim_batch should call list_tenants
        claimed1 = await poller.claim_batch()
        assert mock_extension.list_tenants_calls == 1
        assert len(claimed1) == 2

        # Add more tasks
        for i in range(2):
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "index": i + 10, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
                VALUES ($1, $2, 'test', 'pending', $3::jsonb)
                """,
                op_id,
                bank_id,
                payload,
            )

        # Second claim_batch should call list_tenants again
        claimed2 = await poller.claim_batch()
        assert mock_extension.list_tenants_calls == 2
        assert len(claimed2) == 2

    @pytest.mark.asyncio
    async def test_poller_picks_up_new_tenants_without_restart(self, pool, backend, clean_operations):
        """Test that new tenants are discovered on subsequent poll cycles."""
        from hindsight_api.extensions.tenant import Tenant, TenantExtension
        from hindsight_api.worker import WorkerPoller

        class DynamicTenantExtension(TenantExtension):
            def __init__(self):
                # Start with just public
                self.tenants: list[Tenant] = [Tenant(schema="public")]
                self.list_tenants_calls = 0

            async def authenticate(self, context):
                raise NotImplementedError("Not used in this test")

            async def list_tenants(self) -> list[Tenant]:
                self.list_tenants_calls += 1
                return self.tenants

        dynamic_extension = DynamicTenantExtension()

        # Create a task in public schema
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        op_id = uuid.uuid4()
        payload = json.dumps({"type": "test_task", "bank_id": bank_id})
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'test', 'pending', $3::jsonb)
            """,
            op_id,
            bank_id,
            payload,
        )

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=lambda x: None,
            tenant_extension=dynamic_extension,
        )

        # First poll - only public schema
        claimed1 = await poller.claim_batch()
        assert len(claimed1) == 1
        assert claimed1[0].schema is None  # public is represented as None
        assert dynamic_extension.list_tenants_calls == 1

        # Simulate tenant list changing (but we won't add a non-existent schema)
        # In real world, the schema would be created before list_tenants returns it
        # Here we just verify that list_tenants is called again

        # Add another task to public
        op_id2 = uuid.uuid4()
        payload2 = json.dumps({"type": "test_task", "bank_id": bank_id})
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'test', 'pending', $3::jsonb)
            """,
            op_id2,
            bank_id,
            payload2,
        )

        # Second poll - list_tenants should be called again
        claimed2 = await poller.claim_batch()
        assert len(claimed2) == 1
        assert dynamic_extension.list_tenants_calls == 2  # Called again on second poll

        # Third poll with no tasks - still calls list_tenants
        claimed3 = await poller.claim_batch()
        assert len(claimed3) == 0
        assert dynamic_extension.list_tenants_calls == 3  # Called again even with no tasks

    @pytest.mark.asyncio
    async def test_poller_without_tenant_extension_uses_public(self, pool, backend, clean_operations):
        """Test that poller uses public schema when no tenant extension is configured."""
        from hindsight_api.worker import WorkerPoller

        # Create pending tasks
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        for i in range(3):
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "index": i, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
                VALUES ($1, $2, 'test', 'pending', $3::jsonb)
                """,
                op_id,
                bank_id,
                payload,
            )

        # No tenant_extension provided
        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-1",
            executor=lambda x: None,
        )

        claimed = await poller.claim_batch()
        # Filter to our bank — parallel tests may contribute other claims.
        my_claims = [c for c in claimed if c.task_dict.get("bank_id") == bank_id]
        assert len(my_claims) == 3, f"Expected 3 claims for our bank, got {len(my_claims)}"

        # All tasks (ours or not) should have schema=None since no tenant extension is set.
        for task in claimed:
            assert task.schema is None

    @pytest.mark.asyncio
    async def test_poller_with_custom_schema(self, pool, backend):
        """Test that poller uses custom schema when schema parameter is provided."""
        from hindsight_api.worker import WorkerPoller

        # Create a custom schema for testing
        test_schema = "test_custom_schema"

        try:
            # Create schema and copy table structure
            await pool.execute(f'CREATE SCHEMA IF NOT EXISTS "{test_schema}"')
            await pool.execute(
                f"""
                CREATE TABLE "{test_schema}".async_operations (
                    LIKE public.async_operations INCLUDING ALL
                )
                """
            )

            # Create pending tasks in the custom schema
            bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
            task_ids = []
            for i in range(3):
                op_id = uuid.uuid4()
                task_ids.append(str(op_id))
                payload = json.dumps({"type": "test_task", "index": i, "bank_id": bank_id})
                await pool.execute(
                    f"""
                    INSERT INTO "{test_schema}".async_operations (operation_id, bank_id, operation_type, status, task_payload)
                    VALUES ($1, $2, 'test', 'pending', $3::jsonb)
                    """,
                    op_id,
                    bank_id,
                    payload,
                )

            # Create poller with custom schema
            poller = WorkerPoller(
                backend=backend,
                worker_id="test-worker-custom-schema",
                executor=lambda x: None,
                schema=test_schema,
            )

            # Claim tasks
            claimed = await poller.claim_batch()
            assert len(claimed) == 3, f"Expected 3 tasks, got {len(claimed)}"

            # All tasks should have schema=test_schema
            claimed_ids = []
            for task in claimed:
                assert task.schema == test_schema, f"Expected schema '{test_schema}', got '{task.schema}'"
                claimed_ids.append(task.operation_id)

            # Verify claimed tasks match what we inserted
            assert set(claimed_ids) == set(task_ids)

            # Verify tasks are marked as processing in the custom schema
            rows = await pool.fetch(
                f"""
                SELECT operation_id, status, worker_id
                FROM "{test_schema}".async_operations
                WHERE operation_id = ANY($1)
                """,
                [uuid.UUID(tid) for tid in task_ids],
            )
            assert len(rows) == 3
            for row in rows:
                assert row["status"] == "processing"
                assert row["worker_id"] == "test-worker-custom-schema"

        finally:
            # Clean up: drop the custom schema
            await pool.execute(f'DROP SCHEMA IF EXISTS "{test_schema}" CASCADE')


async def test_worker_fire_and_forget_nonblocking(pool, backend, clean_operations):
    """
    Test that worker continues polling while tasks run (fire-and-forget pattern).

    This test verifies the FIX: With the old blocking behavior, the worker would
    wait for all tasks in a batch to complete before claiming more. This test
    would FAIL with the old code because tasks 3-4 wouldn't be claimed until
    tasks 1-2 complete. With fire-and-forget, tasks 3-4 are claimed immediately.
    """
    from hindsight_api.worker.poller import WorkerPoller

    task_started = {}  # operation_id -> Event (set when task starts)
    task_canfinish = {}  # operation_id -> Event (wait before finishing)
    # Set before the executor is defined so the closure captures it.
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"

    async def blocking_executor(task_dict: dict):
        # Ignore tasks leaked in from parallel test files — they'd otherwise
        # show up in task_started and make our count-based gating flaky.
        if task_dict.get("bank_id") != bank_id:
            return
        op_id = task_dict["operation_id"]
        # Signal that this task has started
        started = asyncio.Event()
        task_started[op_id] = started
        started.set()

        # Block until we're told to finish
        finish = asyncio.Event()
        task_canfinish[op_id] = finish
        await finish.wait()

    poller = WorkerPoller(
        backend=backend,
        worker_id="test-worker",
        executor=blocking_executor,
        poll_interval_ms=50,  # Fast polling
        max_slots=10,
        slot_reservations={"consolidation": 2},
    )

    await _ensure_bank(pool, bank_id)

    # Submit initial 2 tasks
    task_ids = []
    for i in range(2):
        op_id = uuid.uuid4()
        task_ids.append(str(op_id))
        payload = json.dumps(
            {"type": "test", "operation_type": "retain", "operation_id": str(op_id), "bank_id": bank_id}
        )
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'retain', 'pending', $3::jsonb)
            """,
            op_id,
            bank_id,
            payload,
        )

    poll_task = asyncio.create_task(poller.run())

    try:
        # Wait for first 2 tasks to start executing (but not finish)
        for i in range(100):  # Try for up to 1 second
            if len(task_started) >= 2:
                break
            await asyncio.sleep(0.01)
        assert len(task_started) == 2, f"Expected 2 tasks started, got {len(task_started)}"

        # Verify at least our 2 tasks are in_flight (leaked non-our-bank tasks
        # return immediately from the executor but can transiently be counted).
        async with poller._in_flight_lock:
            assert poller._in_flight_count >= 2

        # NOW submit 2 more tasks WHILE the first 2 are still running
        for i in range(2):
            op_id = uuid.uuid4()
            task_ids.append(str(op_id))
            payload = json.dumps(
                {"type": "test", "operation_type": "retain", "operation_id": str(op_id), "bank_id": bank_id}
            )
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
                VALUES ($1, $2, 'retain', 'pending', $3::jsonb)
                """,
                op_id,
                bank_id,
                payload,
            )

        # KEY ASSERTION: Worker should claim tasks 3-4 WITHOUT waiting for 1-2 to finish
        # This would FAIL with the old blocking behavior
        for i in range(100):  # Try for up to 1 second
            if len(task_started) >= 4:
                break
            await asyncio.sleep(0.01)

        assert len(task_started) == 4, (
            f"Fire-and-forget FAILED: Expected 4 tasks started, got {len(task_started)}. "
            "This means the worker blocked waiting for the first batch to complete."
        )

        # Verify at least all 4 of ours are in-flight (see comment above about
        # transient leaked tasks).
        async with poller._in_flight_lock:
            assert poller._in_flight_count >= 4

        # Clean up: allow all tasks to finish
        for event in task_canfinish.values():
            event.set()

    finally:
        # Ensure cleanup
        for event in task_canfinish.values():
            event.set()
        await poller.shutdown_graceful(timeout=2.0)
        try:
            await asyncio.wait_for(poll_task, timeout=1.0)
        except asyncio.CancelledError:
            pass


async def test_worker_slot_limits_enforced(pool, backend, clean_operations):
    """Test that worker respects max_slots and won't exceed the limit."""
    from hindsight_api.worker.poller import WorkerPoller

    tasks_started = set()
    task_events = {}
    # Use a closure-captured bank_id (set below) so the executor can ignore
    # tasks leaked in by parallel test files.
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"

    async def controlled_executor(task_dict: dict):
        # Other test files running in parallel may have pending async_operations
        # that this poller legitimately claims — don't count them toward our
        # 3-slot limit assertions, but still service them so we don't starve
        # them. For this test, we simply complete them immediately and focus
        # our gating on tasks for our own bank.
        op_id = task_dict["operation_id"]
        if task_dict.get("bank_id") != bank_id:
            return
        tasks_started.add(op_id)
        event = asyncio.Event()
        task_events[op_id] = event
        await event.wait()

    poller = WorkerPoller(
        backend=backend,
        worker_id="test-worker",
        executor=controlled_executor,
        poll_interval_ms=50,
        max_slots=3,  # Only allow 3 concurrent tasks
        slot_reservations={},  # No reservations; all 3 slots available as shared pool
    )

    # Submit 10 tasks
    await _ensure_bank(pool, bank_id)
    for i in range(10):
        op_id = uuid.uuid4()
        payload = json.dumps(
            {"type": "test", "operation_type": "retain", "operation_id": str(op_id), "bank_id": bank_id}
        )
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'retain', 'pending', $3::jsonb)
            """,
            op_id,
            bank_id,
            payload,
        )

    poll_task = asyncio.create_task(poller.run())

    try:
        # Wait for slots to fill
        for i in range(100):
            if len(tasks_started) >= 3:
                break
            await asyncio.sleep(0.01)

        # Should have claimed exactly 3 tasks (slot limit)
        assert len(tasks_started) == 3

        # Wait to ensure no additional tasks are claimed
        for i in range(30):
            await asyncio.sleep(0.01)
        assert len(tasks_started) == 3, "Worker exceeded slot limit!"

        # Release tasks one by one and verify remaining are claimed
        completed = 0
        while completed < 10 and len(tasks_started) < 10:
            # Release the next batch
            events_to_release = list(task_events.values())[completed : completed + 3]
            for event in events_to_release:
                event.set()
            completed += len(events_to_release)

            # Wait for new tasks to be claimed
            for i in range(100):
                if len(tasks_started) >= min(completed + 3, 10):
                    break
                await asyncio.sleep(0.01)

        assert len(tasks_started) == 10

    finally:
        for event in task_events.values():
            event.set()
        await poller.shutdown_graceful(timeout=2.0)
        try:
            await asyncio.wait_for(poll_task, timeout=1.0)
        except asyncio.CancelledError:
            pass


async def test_consolidation_slots_reserved_when_retain_saturates(pool, backend, clean_operations):
    """Regression: consolidation must not be starved when retain saturates the queue.

    With ``max_slots=5`` and ``slot_reservations={"consolidation": 2}``, retain tasks
    may use at most 3 concurrent slots (the shared pool), leaving 2 slots reserved for
    consolidation. Without the reservation (issue #1006), a continuous stream of retain
    tasks would fill every slot and consolidation would never run.
    """
    from hindsight_api.worker.poller import WorkerPoller

    started: dict[str, str] = {}  # op_id -> op_type
    finish_events: dict[str, asyncio.Event] = {}
    # Set before the executor is defined so the closure captures it.
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"

    async def blocking_executor(task_dict: dict):
        # Ignore tasks leaked in from parallel test files — they'd otherwise
        # inflate the started/in_flight counters we assert on.
        if task_dict.get("bank_id") != bank_id:
            return
        op_id = task_dict["operation_id"]
        started[op_id] = task_dict.get("operation_type", "unknown")
        event = asyncio.Event()
        finish_events[op_id] = event
        await event.wait()

    poller = WorkerPoller(
        backend=backend,
        worker_id="test-worker-consolidation-reservation",
        executor=blocking_executor,
        poll_interval_ms=50,
        max_slots=5,
        slot_reservations={"consolidation": 2},
    )

    await _ensure_bank(pool, bank_id)

    # Submit 10 retain tasks first — these should be claimed up to the
    # shared pool cap (max_slots - sum(reservations) = 3).
    for _ in range(10):
        op_id = uuid.uuid4()
        payload = json.dumps(
            {"type": "test", "operation_type": "retain", "operation_id": str(op_id), "bank_id": bank_id}
        )
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'retain', 'pending', $3::jsonb)
            """,
            op_id,
            bank_id,
            payload,
        )

    # Submit 1 consolidation task. Note the payload deliberately omits operation_type
    # to verify the poller injects it from the DB column.
    consolidation_op_id = uuid.uuid4()
    consolidation_payload = json.dumps({"type": "test", "operation_id": str(consolidation_op_id), "bank_id": bank_id})
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
        VALUES ($1, $2, 'consolidation', 'pending', $3::jsonb)
        """,
        consolidation_op_id,
        bank_id,
        consolidation_payload,
    )

    poll_task = asyncio.create_task(poller.run())

    try:
        # Wait for the worker to fill its slots: 3 retain + 1 consolidation = 4 active.
        for _ in range(200):
            if len(started) >= 4:
                break
            await asyncio.sleep(0.01)

        retain_started = [op for op, t in started.items() if t == "retain"]
        consolidation_started = [op for op, t in started.items() if t == "consolidation"]

        assert len(retain_started) == 3, (
            f"Retain should be capped at shared pool size (max_slots - sum(reservations)) = 3, got {len(retain_started)}"
        )
        assert len(consolidation_started) == 1, (
            f"Consolidation should claim its reserved slot even while retain saturates, "
            f"got {len(consolidation_started)}"
        )
        assert str(consolidation_op_id) in consolidation_started

        # In-flight tracking must record the consolidation task under the right key,
        # otherwise the consolidation pool accounting drifts on subsequent claims.
        # Use >= because parallel test files may transiently contribute a
        # consolidation claim that our executor returns from immediately.
        async with poller._in_flight_lock:
            assert poller._in_flight_by_type.get("consolidation", 0) >= 1

    finally:
        for event in finish_events.values():
            event.set()
        await poller.shutdown_graceful(timeout=2.0)
        try:
            await asyncio.wait_for(poll_task, timeout=1.0)
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_per_operation_slot_reservations(pool, backend, clean_operations):
    """Test that per-operation slot reservations guarantee capacity for each type.

    With ``max_slots=8`` and ``slot_reservations={"consolidation": 2, "retain": 3}``,
    retain gets 3 reserved, consolidation gets 2 reserved, and there are 3 shared slots.
    When retain saturates the queue, it should claim 3 (reserved) + 3 (shared) = 6 max,
    but consolidation's 2 reserved slots must remain available.
    """
    from hindsight_api.worker.poller import WorkerPoller

    started: dict[str, str] = {}  # op_id -> op_type
    finish_events: dict[str, asyncio.Event] = {}
    # Set before the executor is defined so the closure captures it.
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
    consolidation_bank_1 = f"test-worker-{uuid.uuid4().hex[:8]}"
    consolidation_bank_2 = f"test-worker-{uuid.uuid4().hex[:8]}"
    our_banks = {bank_id, consolidation_bank_1, consolidation_bank_2}

    async def blocking_executor(task_dict: dict):
        # Ignore tasks leaked in from parallel test files.
        if task_dict.get("bank_id") not in our_banks:
            return
        op_id = task_dict["operation_id"]
        started[op_id] = task_dict.get("operation_type", "unknown")
        event = asyncio.Event()
        finish_events[op_id] = event
        await event.wait()

    poller = WorkerPoller(
        backend=backend,
        worker_id="test-worker-per-op-slots",
        executor=blocking_executor,
        poll_interval_ms=50,
        max_slots=8,
        slot_reservations={"consolidation": 2, "retain": 3},
    )

    await _ensure_bank(pool, bank_id)

    # Submit 10 retain tasks — should fill 3 reserved + 3 shared = 6
    for _ in range(10):
        op_id = uuid.uuid4()
        payload = json.dumps(
            {"type": "test", "operation_type": "retain", "operation_id": str(op_id), "bank_id": bank_id}
        )
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'retain', 'pending', $3::jsonb)
            """,
            op_id,
            bank_id,
            payload,
        )

    # Submit 2 consolidation tasks (different banks for bank-serialization)
    await _ensure_bank(pool, consolidation_bank_1)
    await _ensure_bank(pool, consolidation_bank_2)

    for c_bank in [consolidation_bank_1, consolidation_bank_2]:
        c_op_id = uuid.uuid4()
        c_payload = json.dumps({"type": "test", "operation_id": str(c_op_id), "bank_id": c_bank})
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'consolidation', 'pending', $3::jsonb)
            """,
            c_op_id,
            c_bank,
            c_payload,
        )

    poll_task = asyncio.create_task(poller.run())

    try:
        # Wait for the worker to fill its slots: 6 retain + 2 consolidation = 8 active
        for _ in range(200):
            if len(started) >= 8:
                break
            await asyncio.sleep(0.01)

        retain_started = [op for op, t in started.items() if t == "retain"]
        consolidation_started = [op for op, t in started.items() if t == "consolidation"]

        assert len(retain_started) == 6, f"Retain should use 3 reserved + 3 shared = 6 slots, got {len(retain_started)}"
        assert len(consolidation_started) == 2, (
            f"Consolidation should use its 2 reserved slots, got {len(consolidation_started)}"
        )

        # Verify in-flight tracking — use >= to tolerate transient leaked
        # tasks from parallel test files (our executor returns from them
        # immediately, but the counter may see them briefly).
        async with poller._in_flight_lock:
            assert poller._in_flight_by_type.get("retain", 0) >= 6
            assert poller._in_flight_by_type.get("consolidation", 0) >= 2

    finally:
        for event in finish_events.values():
            event.set()
        await poller.shutdown_graceful(timeout=2.0)
        try:
            await asyncio.wait_for(poll_task, timeout=1.0)
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_shared_pool_usable_by_reserved_types(pool, backend, clean_operations):
    """Test that operation types with reservations can also use shared pool slots.

    With ``max_slots=5`` and ``slot_reservations={"retain": 2}``, shared pool = 3.
    If 5 retain tasks are submitted, retain should use 2 reserved + 3 shared = 5.
    """
    from hindsight_api.worker.poller import WorkerPoller

    started: dict[str, str] = {}
    finish_events: dict[str, asyncio.Event] = {}
    # Set before the executor is defined so the closure captures it.
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"

    async def blocking_executor(task_dict: dict):
        # Ignore tasks leaked in from parallel test files.
        if task_dict.get("bank_id") != bank_id:
            return
        op_id = task_dict["operation_id"]
        started[op_id] = task_dict.get("operation_type", "unknown")
        event = asyncio.Event()
        finish_events[op_id] = event
        await event.wait()

    poller = WorkerPoller(
        backend=backend,
        worker_id="test-worker-shared-overflow",
        executor=blocking_executor,
        poll_interval_ms=50,
        max_slots=5,
        slot_reservations={"retain": 2},
    )

    await _ensure_bank(pool, bank_id)

    # Submit 10 retain tasks
    for _ in range(10):
        op_id = uuid.uuid4()
        payload = json.dumps(
            {"type": "test", "operation_type": "retain", "operation_id": str(op_id), "bank_id": bank_id}
        )
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'retain', 'pending', $3::jsonb)
            """,
            op_id,
            bank_id,
            payload,
        )

    poll_task = asyncio.create_task(poller.run())

    try:
        # Wait for the worker to fill all 5 slots with retain
        for _ in range(200):
            if len(started) >= 5:
                break
            await asyncio.sleep(0.01)

        retain_started = [op for op, t in started.items() if t == "retain"]
        assert len(retain_started) == 5, f"Retain should use 2 reserved + 3 shared = 5 slots, got {len(retain_started)}"

        # Should not exceed max_slots (for our bank's tasks). Leaked tasks
        # from other test files return from our executor immediately, so they
        # don't appear in `started`.
        await asyncio.sleep(0.1)
        assert len(started) == 5, f"Should not exceed max_slots=5 for our tasks, got {len(started)}"

    finally:
        for event in finish_events.values():
            event.set()
        await poller.shutdown_graceful(timeout=2.0)
        try:
            await asyncio.wait_for(poll_task, timeout=1.0)
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_pending_breakdown_explains_unclaimable_rows(pool, backend, clean_operations, caplog):
    """Pending rows that the claim query filters out must be visible in logs.

    Background: production incident where a 'pending' retain sat in the queue for
    hours while workers had free slots. With only the global pending count in
    [WORKER_STATS] there's no way to tell whether the rows are claimable-but-not-
    being-claimed (real bug) vs filtered out by the claim WHERE clause (data
    state). This test verifies [PENDING_BREAKDOWN] surfaces each filter bucket
    so operators can diagnose without DB access.
    """
    import logging

    from hindsight_api.worker.poller import WorkerPoller

    poller = WorkerPoller(
        backend=backend,
        worker_id="test-worker-pending-breakdown",
        executor=lambda _t: asyncio.sleep(0),
        poll_interval_ms=50,
        max_slots=5,
    )

    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(pool, bank_id)

    # Mix of pending rows that the claim query treats differently:
    #   * payload_null  - batch_retain parent (orphan candidate)
    #   * retry_blocked - failed once, scheduled an hour out
    #   * assigned      - worker_id stamped (e.g. left over from a prior crash
    #                     that re-queued without clearing worker_id)
    #   * claimable     - normal retain ready to go
    #   * consolidation - normal consolidation, also claimable
    rows = [
        ("batch_retain", None, None, None),  # payload_null
        ("retain", json.dumps({"type": "test"}), "future", None),  # retry_blocked
        ("retain", json.dumps({"type": "test"}), None, "ghost-worker"),  # assigned
        ("retain", json.dumps({"type": "test"}), None, None),  # claimable
        ("consolidation", json.dumps({"type": "test"}), None, None),  # claimable
    ]
    for op_type, payload, retry_marker, worker_id in rows:
        op_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO async_operations
                (operation_id, bank_id, operation_type, status, task_payload,
                 next_retry_at, worker_id)
            VALUES ($1, $2, $3, 'pending', $4::jsonb,
                    CASE WHEN $5::text = 'future' THEN now() + interval '1 hour' ELSE NULL END,
                    $6)
            """,
            op_id,
            bank_id,
            op_type,
            payload,
            retry_marker,
            worker_id,
        )

    # Trigger one stats emit. _last_progress_log starts at 0, so the first call
    # always logs.
    with caplog.at_level(logging.INFO, logger="hindsight_api.worker.poller"):
        await poller._log_progress_if_due()

    breakdown_lines = [r.message for r in caplog.records if r.message.startswith("[PENDING_BREAKDOWN]")]
    assert len(breakdown_lines) == 1, f"Expected exactly one breakdown line, got: {breakdown_lines}"

    # The breakdown is global (not bank-scoped), so other rows in the table may
    # contribute. Parse the per-op_type buckets from the line and assert that
    # our additions appear (>= 1 for each bucket we populated).
    line = breakdown_lines[0]
    buckets: dict[str, dict[str, int]] = {}
    for section in line.removeprefix("[PENDING_BREAKDOWN]").split("|"):
        section = section.strip()
        if ":" not in section:
            continue
        op_type, fields = section.split(":", 1)
        kv = {}
        for token in fields.strip().split():
            k, _, v = token.partition("=")
            kv[k] = int(v)
        buckets[op_type.strip()] = kv

    assert buckets["batch_retain"]["payload_null"] >= 1
    assert buckets["retain"]["retry_blocked"] >= 1
    assert buckets["retain"]["assigned"] >= 1
    assert buckets["retain"]["claimable"] >= 1
    assert buckets["consolidation"]["claimable"] >= 1


class TestSummariseChildErrorMessages:
    """Pure unit tests for the _summarise_child_error_messages helper.

    The helper picks a representative error message for a parent whose
    children failed. The integration tests above exercise the full path
    through _mark_failed; these tests focus on the choice itself.
    """

    def _sib(self, status: str, error_message: str | None = None) -> dict:
        return {"status": status, "error_message": error_message}

    def test_all_failed_with_same_message_inherits_that_message(self):
        from hindsight_api.worker.poller import _summarise_child_error_messages

        siblings = [
            self._sib("failed", "boom"),
            self._sib("failed", "boom"),
            self._sib("failed", "boom"),
        ]
        assert _summarise_child_error_messages(siblings) == "boom"

    def test_mixed_failed_messages_picks_most_common(self):
        from hindsight_api.worker.poller import _summarise_child_error_messages

        siblings = [
            self._sib("failed", "common cause"),
            self._sib("failed", "common cause"),
            self._sib("failed", "rare cause"),
        ]
        assert _summarise_child_error_messages(siblings) == "common cause"

    def test_completed_siblings_ignored(self):
        from hindsight_api.worker.poller import _summarise_child_error_messages

        siblings = [
            self._sib("completed", None),
            self._sib("completed", None),
            self._sib("failed", "the one real failure"),
        ]
        assert _summarise_child_error_messages(siblings) == "the one real failure"

    def test_no_failed_siblings_falls_back_to_generic(self):
        from hindsight_api.worker.poller import _summarise_child_error_messages

        siblings = [
            self._sib("completed", None),
            self._sib("completed", None),
        ]
        assert _summarise_child_error_messages(siblings) == "One or more sub-batches failed"

    def test_failed_siblings_with_no_error_message_falls_back_to_generic(self):
        from hindsight_api.worker.poller import _summarise_child_error_messages

        siblings = [
            self._sib("failed", None),
            self._sib("failed", ""),
        ]
        assert _summarise_child_error_messages(siblings) == "One or more sub-batches failed"

    def test_whitespace_only_messages_treated_as_empty(self):
        from hindsight_api.worker.poller import _summarise_child_error_messages

        siblings = [
            self._sib("failed", "   "),
            self._sib("failed", "actual error"),
        ]
        assert _summarise_child_error_messages(siblings) == "actual error"


class TestMarkFailedParentPropagation:
    """Tests for _mark_failed parent propagation in WorkerPoller.

    When a child retain operation fails via an unhandled exception, the memory
    engine's transaction is rolled back entirely — including any call to
    _maybe_update_parent_operation inside the engine. The poller's fallback
    _mark_failed must detect this and finalise the parent batch_retain itself.
    """

    async def _insert_op(
        self,
        pool,
        *,
        op_id: "uuid.UUID",
        bank_id: str,
        operation_type: str,
        status: str,
        result_metadata: dict | None = None,
    ) -> None:
        meta_json = json.dumps(result_metadata if result_metadata is not None else {})
        await pool.execute(
            """
            INSERT INTO async_operations
                (operation_id, bank_id, operation_type, status, result_metadata)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            """,
            op_id,
            bank_id,
            operation_type,
            status,
            meta_json,
        )

    @pytest.mark.asyncio
    async def test_mark_failed_finalises_parent_when_last_sibling_fails(self, pool, backend, clean_operations):
        """When the last pending child fails, parent batch_retain is marked failed."""
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        parent_id = uuid.uuid4()
        child1_id = uuid.uuid4()
        child2_id = uuid.uuid4()

        # Parent batch_retain still pending
        await self._insert_op(pool, op_id=parent_id, bank_id=bank_id, operation_type="batch_retain", status="pending")

        # child1 already completed
        await self._insert_op(
            pool,
            op_id=child1_id,
            bank_id=bank_id,
            operation_type="retain",
            status="completed",
            result_metadata={"parent_operation_id": str(parent_id)},
        )

        # child2 still processing — this is the one that will fail
        await self._insert_op(
            pool,
            op_id=child2_id,
            bank_id=bank_id,
            operation_type="retain",
            status="processing",
            result_metadata={"parent_operation_id": str(parent_id)},
        )

        poller = WorkerPoller(backend=backend, worker_id="test-worker-1", executor=lambda x: None)
        await poller._mark_failed(str(child2_id), "DB constraint violation", schema=None)

        # child2 must be failed
        child2_row = await pool.fetchrow(
            "SELECT status, error_message FROM async_operations WHERE operation_id = $1", child2_id
        )
        assert child2_row["status"] == "failed"
        assert "DB constraint violation" in child2_row["error_message"]

        # parent must now be failed (all siblings done, at least one failed)
        parent_row = await pool.fetchrow(
            "SELECT status, error_message FROM async_operations WHERE operation_id = $1",
            parent_id,
        )
        assert parent_row["status"] == "failed", (
            f"Parent should be 'failed' when last sibling fails, got '{parent_row['status']}'"
        )
        # Parent error_message must propagate the child's actual error reason,
        # not the legacy generic "One or more sub-batches failed". Without this,
        # downstream filters that classify failures by error_message lose all
        # signal once a batch has children.
        assert "DB constraint violation" in (parent_row["error_message"] or ""), (
            f"Parent error_message should inherit child's reason, "
            f"got: {parent_row['error_message']!r}"
        )

    @pytest.mark.asyncio
    async def test_mark_failed_finalises_parent_when_last_sibling_is_sole_child(self, pool, backend, clean_operations):
        """When the only child fails, parent batch_retain becomes failed."""
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        parent_id = uuid.uuid4()
        child_id = uuid.uuid4()

        await self._insert_op(pool, op_id=parent_id, bank_id=bank_id, operation_type="batch_retain", status="pending")
        await self._insert_op(
            pool,
            op_id=child_id,
            bank_id=bank_id,
            operation_type="retain",
            status="processing",
            result_metadata={"parent_operation_id": str(parent_id)},
        )

        poller = WorkerPoller(backend=backend, worker_id="test-worker-1", executor=lambda x: None)
        await poller._mark_failed(str(child_id), "unexpected error", schema=None)

        parent_row = await pool.fetchrow("SELECT status FROM async_operations WHERE operation_id = $1", parent_id)
        assert parent_row["status"] == "failed"

    @pytest.mark.asyncio
    async def test_mark_failed_does_not_finalise_parent_when_siblings_still_pending(self, pool, backend, clean_operations):
        """Parent is NOT updated while other siblings are still processing/pending."""
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        parent_id = uuid.uuid4()
        child1_id = uuid.uuid4()
        child2_id = uuid.uuid4()

        await self._insert_op(pool, op_id=parent_id, bank_id=bank_id, operation_type="batch_retain", status="pending")

        # child1 is the one failing
        await self._insert_op(
            pool,
            op_id=child1_id,
            bank_id=bank_id,
            operation_type="retain",
            status="processing",
            result_metadata={"parent_operation_id": str(parent_id)},
        )
        # child2 is still pending — not done yet
        await self._insert_op(
            pool,
            op_id=child2_id,
            bank_id=bank_id,
            operation_type="retain",
            status="pending",
            result_metadata={"parent_operation_id": str(parent_id)},
        )

        poller = WorkerPoller(backend=backend, worker_id="test-worker-1", executor=lambda x: None)
        await poller._mark_failed(str(child1_id), "early failure", schema=None)

        # child1 is failed
        child1_row = await pool.fetchrow("SELECT status FROM async_operations WHERE operation_id = $1", child1_id)
        assert child1_row["status"] == "failed"

        # parent must still be pending (child2 not done)
        parent_row = await pool.fetchrow("SELECT status FROM async_operations WHERE operation_id = $1", parent_id)
        assert parent_row["status"] == "pending", (
            f"Parent should remain 'pending' while siblings are outstanding, got '{parent_row['status']}'"
        )

    @pytest.mark.asyncio
    async def test_mark_failed_no_parent_is_safe(self, pool, backend, clean_operations):
        """Operations without a parent (no result_metadata parent_operation_id) fail cleanly."""
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        op_id = uuid.uuid4()
        await self._insert_op(pool, op_id=op_id, bank_id=bank_id, operation_type="retain", status="processing")

        poller = WorkerPoller(backend=backend, worker_id="test-worker-1", executor=lambda x: None)
        # Must not raise
        await poller._mark_failed(str(op_id), "standalone failure", schema=None)

        row = await pool.fetchrow("SELECT status FROM async_operations WHERE operation_id = $1", op_id)
        assert row["status"] == "failed"

    @pytest.mark.asyncio
    async def test_unhandled_exception_via_execute_task_propagates_to_parent(self, pool, backend, clean_operations):
        """End-to-end: executor raises a plain exception, poller calls _mark_failed,
        which then resolves the parent batch_retain to failed."""
        from hindsight_api.worker import WorkerPoller
        from hindsight_api.worker.poller import ClaimedTask

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        parent_id = uuid.uuid4()
        child_id = uuid.uuid4()

        await self._insert_op(pool, op_id=parent_id, bank_id=bank_id, operation_type="batch_retain", status="pending")
        await self._insert_op(
            pool,
            op_id=child_id,
            bank_id=bank_id,
            operation_type="retain",
            status="processing",
            result_metadata={"parent_operation_id": str(parent_id)},
        )

        async def crashing_executor(task_dict):
            raise RuntimeError("Simulated DB constraint violation — transaction rolled back")

        poller = WorkerPoller(backend=backend, worker_id="test-worker-1", executor=crashing_executor)

        task_dict = {"type": "retain", "operation_id": str(child_id), "bank_id": bank_id}
        claimed_task = ClaimedTask(operation_id=str(child_id), task_dict=task_dict, schema=None)
        await poller.execute_task(claimed_task)

        completed = await poller.wait_for_active_tasks(timeout=5.0)
        assert completed, "Task did not complete within timeout"

        child_row = await pool.fetchrow("SELECT status FROM async_operations WHERE operation_id = $1", child_id)
        assert child_row["status"] == "failed"

        parent_row = await pool.fetchrow("SELECT status FROM async_operations WHERE operation_id = $1", parent_id)
        assert parent_row["status"] == "failed", (
            f"Parent batch_retain should be 'failed' after child fails via unhandled exception, "
            f"got '{parent_row['status']}'"
        )


class TestClaimBatchRotation:
    """Tests for round-robin schema rotation in claim_batch.

    These use a mocked _claim_batch_for_schema so the tests are hermetic
    and exercise rotation logic without needing multiple real tenant schemas.
    """

    def _make_poller_with_fake_work(self, pool, backend, pending_per_schema, max_slots=1):
        """Build a poller whose schemas and per-schema claims are scripted.

        ``pending_per_schema`` maps schema name -> current pending count.
        The fake claim handler decrements the count and returns a ClaimedTask
        if the schema still has work, else returns an empty list.
        """
        from hindsight_api.extensions.tenant import Tenant, TenantExtension
        from hindsight_api.worker import WorkerPoller
        from hindsight_api.worker.poller import ClaimedTask

        schemas = list(pending_per_schema.keys())

        class StaticTenantExtension(TenantExtension):
            def __init__(self):
                super().__init__(config={})

            async def authenticate(self, context):
                raise NotImplementedError

            async def list_tenants(self) -> list[Tenant]:
                return [Tenant(schema=s) for s in schemas]

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-rotation",
            executor=lambda x: None,
            tenant_extension=StaticTenantExtension(),
            max_slots=max_slots,
            # No per-type reservations — all slots available as shared pool.
            # Keeps the fair-rotation behavior easy to assert.
            slot_reservations={},
        )

        serviced: list[str] = []

        async def fake_claim(schema, reserved_limits, shared_limit):
            # Tests only exercise non-reserved ("test") tasks, so we only
            # consult the shared_limit.
            remaining = pending_per_schema.get(schema, 0)
            if remaining <= 0 or shared_limit <= 0:
                return []
            take = min(remaining, shared_limit)
            pending_per_schema[schema] = remaining - take
            out = []
            for _ in range(take):
                serviced.append(schema)
                out.append(
                    ClaimedTask(
                        operation_id=str(uuid.uuid4()),
                        task_dict={"operation_type": "test", "bank_id": schema or "default"},
                        schema=schema,
                    )
                )
            return out

        poller._claim_batch_for_schema = fake_claim  # type: ignore[method-assign]

        async def fake_scan(scan_schemas):
            return {s for s in scan_schemas if pending_per_schema.get(s, 0) > 0}

        poller._scan_active_schemas = fake_scan  # type: ignore[method-assign]
        return poller, serviced

    @pytest.mark.asyncio
    async def test_rotation_advances_past_serviced_schema(self, pool, backend):
        """After claiming from schema at offset N, next poll starts at N+1.

        This is the 'crucial detail' that separates working rotation from
        broken rotation: advancing +1 from the previous offset would cause
        the first schema with work to always win.
        """
        # Only schema "b" has work; "a" and "c" are idle.
        pending = {"a": 0, "b": 5, "c": 0}
        poller, serviced = self._make_poller_with_fake_work(pool, backend, pending, max_slots=1)

        await poller.claim_batch()
        # Found work at index 1 ("b"), so next offset should be 2 ("c").
        assert poller._next_schema_idx == 2
        assert serviced == ["b"]

    @pytest.mark.asyncio
    async def test_rotation_advances_by_one_when_no_work(self, pool, backend):
        """Empty sweep advances offset by 1 so we don't keep re-hitting the same head."""
        pending = {"a": 0, "b": 0, "c": 0}
        poller, serviced = self._make_poller_with_fake_work(pool, backend, pending, max_slots=1)

        poller._next_schema_idx = 0
        await poller.claim_batch()
        assert poller._next_schema_idx == 1
        assert serviced == []

        await poller.claim_batch()
        assert poller._next_schema_idx == 2
        assert serviced == []

    @pytest.mark.asyncio
    async def test_small_tenant_not_starved_by_busy_tenant(self, pool, backend):
        """Small tenant with 1 pending task gets serviced within bounded polls
        even when another tenant has a huge backlog. Prevents the regression
        observed in prod where one tenant's 1000+ retains monopolized workers.
        """
        pending = {"friday-main": 1000, "tenant-b": 1}
        poller, serviced = self._make_poller_with_fake_work(pool, backend, pending, max_slots=1)

        # MAX_SLOTS=1 means one claim per poll. Over ~2 polls the rotation
        # must reach tenant-b, regardless of which started first.
        for _ in range(5):
            await poller.claim_batch()
            if "tenant-b" in serviced:
                break

        assert "tenant-b" in serviced, f"tenant-b was starved; serviced={serviced[:20]}"

    @pytest.mark.asyncio
    async def test_max_slots_greater_than_one_spreads_across_tenants(self, pool, backend):
        """With MAX_SLOTS>1 the first pass caps at 1 claim per schema so
        a single poll services multiple tenants rather than draining one.
        """
        pending = {"a": 10, "b": 10, "c": 10, "d": 10, "e": 10}
        poller, serviced = self._make_poller_with_fake_work(pool, backend, pending, max_slots=3)

        await poller.claim_batch()
        # First pass gives 1 claim each to 3 different schemas — not 3 from the same one.
        assert len(serviced) == 3
        assert len(set(serviced)) == 3, f"Expected 3 different tenants, got {serviced}"

    @pytest.mark.asyncio
    async def test_max_slots_greater_than_one_backfills_when_only_one_tenant_has_work(self, pool, backend):
        """Second pass fills remaining slots when only one tenant has work,
        so fairness doesn't sacrifice throughput in the single-tenant case.
        """
        pending = {"a": 0, "b": 10, "c": 0}
        poller, serviced = self._make_poller_with_fake_work(pool, backend, pending, max_slots=3)

        await poller.claim_batch()
        # Pass 1: 1 from "b" (only one with work). Pass 2: 2 more from "b".
        assert serviced == ["b", "b", "b"]

    @pytest.mark.asyncio
    async def test_scan_finds_schemas_with_pending_work(self, pool, backend, clean_operations):
        """_scan_active_schemas identifies schemas with pending rows
        and ignores empty schemas. Uses real DB, no mocks.
        """
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-scan-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-scan",
            executor=lambda x: None,
        )

        # Insert a pending row and verify the scan finds its schema.
        # Note: we can't assert the "no pending rows" case here because
        # parallel test files may add their own pending rows to the public
        # schema during this test.
        op_id = uuid.uuid4()
        await pool.execute(
            """INSERT INTO async_operations
               (operation_id, bank_id, operation_type, status, task_payload)
               VALUES ($1, $2, 'test', 'pending', $3::jsonb)""",
            op_id,
            bank_id,
            json.dumps({"type": "test", "bank_id": bank_id}),
        )

        try:
            result = await poller._scan_active_schemas([None])
            assert None in result, "Scan missed schema with pending work"
        finally:
            await pool.execute("DELETE FROM async_operations WHERE operation_id = $1", op_id)

    @pytest.mark.asyncio
    async def test_scan_uses_optional_routine_when_installed(self, pool, backend, clean_operations):
        """When ``public.schemas_with_pending_work()`` exists in pg_proc,
        ``_scan_active_schemas`` invokes it instead of running per-schema
        EXISTS queries from Python. Confirms the OptionalRoutines probe
        path is wired correctly end-to-end.

        The routine body installed here is a minimal stand-in that
        satisfies the contract documented on
        ``optional_routines.SCHEMAS_WITH_PENDING_WORK`` — Hindsight does
        not own the canonical implementation.
        """
        from hindsight_api.engine.db.postgresql import PostgresConnection
        from hindsight_api.worker import WorkerPoller

        # Minimal contract-satisfying implementation: returns the empty
        # set. Enough to prove the poller follows the server-side path.
        await pool.execute(
            "CREATE OR REPLACE FUNCTION public.schemas_with_pending_work() "
            "RETURNS SETOF text AS $$ BEGIN RETURN; END $$ LANGUAGE plpgsql STABLE"
        )
        try:
            poller = WorkerPoller(
                backend=backend,
                worker_id="test-routine",
                executor=lambda x: None,
            )

            captured_fetch: list[str] = []
            captured_fetchval: list[str] = []
            original_fetch = PostgresConnection.fetch
            original_fetchval = PostgresConnection.fetchval

            async def spy_fetch(self, query, *args, timeout=None):
                captured_fetch.append(query)
                return await original_fetch(self, query, *args, timeout=timeout)

            async def spy_fetchval(self, query, *args, column=0, timeout=None):
                captured_fetchval.append(query)
                return await original_fetchval(self, query, *args, column=column, timeout=timeout)

            PostgresConnection.fetch = spy_fetch  # type: ignore[method-assign]
            PostgresConnection.fetchval = spy_fetchval  # type: ignore[method-assign]
            try:
                await poller._scan_active_schemas([None])
            finally:
                PostgresConnection.fetch = original_fetch  # type: ignore[method-assign]
                PostgresConnection.fetchval = original_fetchval  # type: ignore[method-assign]

            # Routine was probed (one pg_proc lookup) and then invoked.
            assert any("pg_proc" in q for q in captured_fetchval), (
                f"Expected a pg_proc existence probe; fetchval queries: {captured_fetchval}"
            )
            assert any("schemas_with_pending_work" in q for q in captured_fetch), (
                f"Expected schemas_with_pending_work() to be invoked; fetch queries: {captured_fetch}"
            )
            # Fallback per-schema EXISTS path must NOT have run.
            assert not any("async_operations" in q and "EXISTS" in q for q in captured_fetchval), (
                f"Fallback EXISTS path should be skipped when routine is installed; fetchval queries: {captured_fetchval}"
            )
            # Probe result is cached so the next scan skips the pg_proc lookup.
            assert poller._optional_routines._cache.get("schemas_with_pending_work") is True
        finally:
            await pool.execute("DROP FUNCTION IF EXISTS public.schemas_with_pending_work()")

    @pytest.mark.asyncio
    async def test_claim_batch_only_queries_active_schemas(self, pool, backend, clean_operations):
        """claim_batch uses _scan_active_schemas to pre-filter, then
        only calls _claim_batch_for_schema on schemas the scan found.
        Verifies the scan→claim pipeline end-to-end with real DB rows.
        """
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-pipeline-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        await pool.execute(
            """INSERT INTO async_operations
               (operation_id, bank_id, operation_type, status, task_payload)
               VALUES ($1, $2, 'test', 'pending', $3::jsonb)""",
            uuid.uuid4(),
            bank_id,
            json.dumps({"type": "test", "bank_id": bank_id}),
        )

        schemas_claimed: list[str | None] = []
        poller = WorkerPoller(
            backend=backend,
            worker_id="test-pipeline",
            executor=lambda x: None,
        )
        original_claim = poller._claim_batch_for_schema

        async def tracking_claim(schema, nc_limit, cons_limit):
            schemas_claimed.append(schema)
            return await original_claim(schema, nc_limit, cons_limit)

        poller._claim_batch_for_schema = tracking_claim  # type: ignore[method-assign]

        claimed = await poller.claim_batch()

        # Filter to our bank — parallel test files may contribute other claims in public schema.
        my_claims = [c for c in claimed if c.task_dict.get("bank_id") == bank_id]
        assert len(my_claims) == 1, f"Expected 1 claim for our bank, got {len(my_claims)}"
        assert len(schemas_claimed) <= 3, (
            f"Claim called on {len(schemas_claimed)} schemas — should only visit schemas the scan identified as active"
        )


class TestDecommissionAllWorkers:
    """Tests for decommission-workers (all workers) functionality."""

    @pytest.mark.asyncio
    async def test_decommission_all_releases_all_processing_tasks(self, pool, clean_operations):
        """Test that decommissioning all workers releases tasks from every worker."""
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        # Create tasks for multiple workers
        for worker in ["worker-a", "worker-b", "worker-c"]:
            for i in range(2):
                op_id = uuid.uuid4()
                payload = json.dumps({"type": "test_task", "index": i, "bank_id": bank_id})
                await pool.execute(
                    """
                    INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id, claimed_at)
                    VALUES ($1, $2, 'test', 'processing', $3::jsonb, $4, now())
                    """,
                    op_id,
                    bank_id,
                    payload,
                    worker,
                )

        # Decommission all
        result = await pool.fetch(
            """
            UPDATE async_operations
            SET status = 'pending', worker_id = NULL, claimed_at = NULL, updated_at = now()
            WHERE status = 'processing' AND bank_id = $1
            RETURNING operation_id, worker_id, operation_type
            """,
            bank_id,
        )

        assert len(result) == 6

        # All should be pending now
        rows = await pool.fetch(
            "SELECT status, worker_id, claimed_at FROM async_operations WHERE bank_id = $1",
            bank_id,
        )
        for row in rows:
            assert row["status"] == "pending"
            assert row["worker_id"] is None
            assert row["claimed_at"] is None

    @pytest.mark.asyncio
    async def test_decommission_all_does_not_affect_pending_or_completed(self, pool, clean_operations):
        """Test that decommissioning all workers only touches 'processing' tasks."""
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        # Create a pending task
        pending_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'test', 'pending', '{"type":"test","bank_id":"x"}'::jsonb)
            """,
            pending_id,
            bank_id,
        )

        # Create a completed task
        completed_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, completed_at)
            VALUES ($1, $2, 'test', 'completed', '{"type":"test","bank_id":"x"}'::jsonb, now())
            """,
            completed_id,
            bank_id,
        )

        # Create a processing task
        processing_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id, claimed_at)
            VALUES ($1, $2, 'test', 'processing', '{"type":"test","bank_id":"x"}'::jsonb, 'dead-worker', now())
            """,
            processing_id,
            bank_id,
        )

        # Decommission all
        result = await pool.fetch(
            """
            UPDATE async_operations
            SET status = 'pending', worker_id = NULL, claimed_at = NULL, updated_at = now()
            WHERE status = 'processing' AND bank_id = $1
            RETURNING operation_id
            """,
            bank_id,
        )

        assert len(result) == 1
        assert result[0]["operation_id"] == processing_id

        # Pending task unchanged
        pending_row = await pool.fetchrow("SELECT status FROM async_operations WHERE operation_id = $1", pending_id)
        assert pending_row["status"] == "pending"

        # Completed task unchanged
        completed_row = await pool.fetchrow("SELECT status FROM async_operations WHERE operation_id = $1", completed_id)
        assert completed_row["status"] == "completed"

    @pytest.mark.asyncio
    async def test_decommission_all_returns_empty_when_no_processing(self, pool, clean_operations):
        """Test decommissioning when there are no processing tasks."""
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        # Only pending tasks
        for i in range(3):
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "index": i, "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
                VALUES ($1, $2, 'test', 'pending', $3::jsonb)
                """,
                op_id,
                bank_id,
                payload,
            )

        result = await pool.fetch(
            """
            UPDATE async_operations
            SET status = 'pending', worker_id = NULL, claimed_at = NULL, updated_at = now()
            WHERE status = 'processing' AND bank_id = $1
            RETURNING operation_id
            """,
            bank_id,
        )

        assert len(result) == 0


class TestWorkerStatus:
    """Tests for worker-status functionality."""

    @pytest.mark.asyncio
    async def test_worker_status_shows_processing_tasks(self, pool, clean_operations):
        """Test that worker status returns all processing tasks with their details."""
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        # Create processing tasks for two workers
        for worker, op_type in [("worker-x", "retain"), ("worker-x", "consolidation"), ("worker-y", "retain")]:
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "bank_id": bank_id})
            await pool.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id, claimed_at)
                VALUES ($1, $2, $3, 'processing', $4::jsonb, $5, now())
                """,
                op_id,
                bank_id,
                op_type,
                payload,
                worker,
            )

        rows = await pool.fetch(
            """
            SELECT worker_id, operation_id, operation_type, bank_id,
                   claimed_at, updated_at,
                   now() - claimed_at AS running_for,
                   now() - updated_at AS last_update_ago
            FROM async_operations
            WHERE status = 'processing' AND bank_id = $1
            ORDER BY worker_id, claimed_at
            """,
            bank_id,
        )

        assert len(rows) == 3

        # Verify all expected columns are present
        for row in rows:
            assert row["worker_id"] in ("worker-x", "worker-y")
            assert row["operation_type"] in ("retain", "consolidation")
            assert row["bank_id"] == bank_id
            assert row["claimed_at"] is not None
            assert row["updated_at"] is not None
            assert row["running_for"] is not None
            assert row["last_update_ago"] is not None

        # Verify grouping: worker-x has 2, worker-y has 1
        worker_x_rows = [r for r in rows if r["worker_id"] == "worker-x"]
        worker_y_rows = [r for r in rows if r["worker_id"] == "worker-y"]
        assert len(worker_x_rows) == 2
        assert len(worker_y_rows) == 1

    @pytest.mark.asyncio
    async def test_worker_status_excludes_non_processing(self, pool, clean_operations):
        """Test that worker status only shows processing tasks, not pending/completed/failed."""
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        # Create tasks in various statuses
        for status in ["pending", "processing", "completed", "failed"]:
            op_id = uuid.uuid4()
            payload = json.dumps({"type": "test_task", "bank_id": bank_id})
            worker = "status-worker" if status == "processing" else None
            claimed = "now()" if status == "processing" else "NULL"
            await pool.execute(
                f"""
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id, claimed_at)
                VALUES ($1, $2, 'test', $3, $4::jsonb, $5, {claimed})
                """,
                op_id,
                bank_id,
                status,
                payload,
                worker,
            )

        rows = await pool.fetch(
            """
            SELECT worker_id, operation_type, bank_id
            FROM async_operations
            WHERE status = 'processing' AND bank_id = $1
            """,
            bank_id,
        )

        assert len(rows) == 1
        assert rows[0]["worker_id"] == "status-worker"

    @pytest.mark.asyncio
    async def test_worker_status_empty_when_no_processing(self, pool, clean_operations):
        """Test that worker status returns empty when no tasks are processing."""
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)

        # Only pending tasks
        op_id = uuid.uuid4()
        payload = json.dumps({"type": "test_task", "bank_id": bank_id})
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'test', 'pending', $3::jsonb)
            """,
            op_id,
            bank_id,
            payload,
        )

        rows = await pool.fetch(
            """
            SELECT worker_id FROM async_operations
            WHERE status = 'processing' AND bank_id = $1
            """,
            bank_id,
        )

        assert len(rows) == 0
