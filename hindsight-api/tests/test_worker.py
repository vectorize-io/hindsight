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

from hindsight_api.engine.task_backend import BrokerTaskBackend, SyncTaskBackend


# Use loadgroup to ensure these tests run in the same worker
# since they share database state
pytestmark = pytest.mark.xdist_group("worker_tests")


@pytest_asyncio.fixture
async def pool(pg0_db_url):
    """Create a dedicated connection pool for worker tests."""
    import asyncpg

    from hindsight_api.pg0 import resolve_database_url

    # Resolve pg0:// URL to postgresql:// URL if needed
    resolved_url = await resolve_database_url(pg0_db_url)

    pool = await asyncpg.create_pool(
        resolved_url,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def clean_operations(pool):
    """Clean up async_operations table before and after tests."""
    # Clean before test
    await pool.execute("DELETE FROM async_operations WHERE bank_id LIKE 'test-worker-%'")
    yield
    # Clean after test
    await pool.execute("DELETE FROM async_operations WHERE bank_id LIKE 'test-worker-%'")


class TestBrokerTaskBackend:
    """Tests for BrokerTaskBackend task storage."""

    @pytest.mark.asyncio
    async def test_submit_task_updates_existing_operation(self, pool, clean_operations):
        """Test that submit_task updates task_payload for existing operations."""
        # Create an operation record first
        operation_id = uuid.uuid4()
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"

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


class TestWorkerPoller:
    """Tests for WorkerPoller task claiming and execution."""

    @pytest.mark.asyncio
    async def test_claim_batch_claims_pending_tasks(self, pool, clean_operations):
        """Test that claim_batch claims pending tasks with task_payload."""
        from hindsight_api.worker import WorkerPoller

        # Create some pending tasks
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
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
            pool=pool,
            worker_id="test-worker-1",
            executor=mock_executor,
        )

        claimed = await poller.claim_batch()
        assert len(claimed) == 3

        # ClaimedTask objects have operation_id, task_dict, schema attributes
        for task in claimed:
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
    async def test_claim_batch_respects_max_slots(self, pool, clean_operations):
        """Test that claim_batch respects the max_slots limit."""
        from hindsight_api.worker import WorkerPoller

        # Create 10 pending tasks
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
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
            pool=pool,
            worker_id="test-worker-1",
            executor=lambda x: None,
            max_slots=3,  # Limit to 3 concurrent tasks
        )

        claimed = await poller.claim_batch()
        assert len(claimed) == 3

    @pytest.mark.asyncio
    async def test_execute_task_marks_completed(self, pool, clean_operations):
        """Test that successful task execution marks task as completed."""
        from hindsight_api.worker import WorkerPoller
        from hindsight_api.worker.poller import ClaimedTask

        # Create a pending task
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        op_id = uuid.uuid4()
        payload = json.dumps({"type": "test_task", "operation_id": str(op_id), "bank_id": bank_id})
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
            executed.append(task_dict)

        poller = WorkerPoller(
            pool=pool,
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

        # Verify task is marked as completed
        row = await pool.fetchrow(
            "SELECT status, completed_at FROM async_operations WHERE operation_id = $1",
            op_id,
        )
        assert row["status"] == "completed"
        assert row["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_execute_task_retries_on_failure(self, pool, clean_operations):
        """Test that failed task execution triggers retry mechanism."""
        from hindsight_api.worker import WorkerPoller
        from hindsight_api.worker.poller import ClaimedTask

        # Create a pending task with retry_count=0
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        op_id = uuid.uuid4()
        payload = json.dumps({"type": "test_task", "operation_id": str(op_id), "bank_id": bank_id})
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id, retry_count)
            VALUES ($1, $2, 'test', 'processing', $3::jsonb, 'test-worker-1', 0)
            """,
            op_id,
            bank_id,
            payload,
        )

        async def failing_executor(task_dict):
            raise ValueError("Simulated failure")

        poller = WorkerPoller(
            pool=pool,
            worker_id="test-worker-1",
            executor=failing_executor,
            max_retries=3,
        )

        # Execute (should fail and retry) - fire-and-forget
        task_dict = json.loads(payload)
        claimed_task = ClaimedTask(operation_id=str(op_id), task_dict=task_dict, schema=None)
        await poller.execute_task(claimed_task)

        # Wait for background task to complete
        completed = await poller.wait_for_active_tasks(timeout=5.0)
        assert completed, "Task did not complete within timeout"

        # Verify task is back to pending with incremented retry_count
        row = await pool.fetchrow(
            "SELECT status, retry_count, worker_id FROM async_operations WHERE operation_id = $1",
            op_id,
        )
        assert row["status"] == "pending"
        assert row["retry_count"] == 1
        assert row["worker_id"] is None  # Worker ID cleared for retry

    @pytest.mark.asyncio
    async def test_execute_task_fails_after_max_retries(self, pool, clean_operations):
        """Test that task is marked failed after exceeding max retries."""
        from hindsight_api.worker import WorkerPoller
        from hindsight_api.worker.poller import ClaimedTask

        # Create a task that has already used all retries
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
        op_id = uuid.uuid4()
        payload = json.dumps({"type": "test_task", "operation_id": str(op_id), "bank_id": bank_id})
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id, retry_count)
            VALUES ($1, $2, 'test', 'processing', $3::jsonb, 'test-worker-1', 3)
            """,
            op_id,
            bank_id,
            payload,
        )

        async def failing_executor(task_dict):
            raise ValueError("Simulated failure")

        poller = WorkerPoller(
            pool=pool,
            worker_id="test-worker-1",
            executor=failing_executor,
            max_retries=3,
        )

        # Execute (should fail permanently) - fire-and-forget
        task_dict = json.loads(payload)
        claimed_task = ClaimedTask(operation_id=str(op_id), task_dict=task_dict, schema=None)
        await poller.execute_task(claimed_task)

        # Wait for background task to complete
        completed = await poller.wait_for_active_tasks(timeout=5.0)
        assert completed, "Task did not complete within timeout"

        # Verify task is marked as failed
        row = await pool.fetchrow(
            "SELECT status, error_message FROM async_operations WHERE operation_id = $1",
            op_id,
        )
        assert row["status"] == "failed"
        assert "Max retries" in row["error_message"]

    @pytest.mark.asyncio
    async def test_claim_batch_skips_consolidation_when_same_bank_processing(self, pool, clean_operations):
        """Test that pending consolidation is skipped if same bank has one processing."""
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"

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
            pool=pool,
            worker_id="test-worker-1",
            executor=lambda x: None,
        )

        claimed = await poller.claim_batch()

        # Should only claim the consolidation for the other bank
        assert len(claimed) == 1
        assert claimed[0].operation_id == str(other_op_id)
        assert claimed[0].task_dict["bank_id"] == other_bank_id

        # Verify the pending consolidation for first bank is still pending
        row = await pool.fetchrow(
            "SELECT status, worker_id FROM async_operations WHERE operation_id = $1",
            pending_op_id,
        )
        assert row["status"] == "pending"
        assert row["worker_id"] is None

    @pytest.mark.asyncio
    async def test_claim_batch_allows_non_consolidation_when_consolidation_processing(self, pool, clean_operations):
        """Test that non-consolidation tasks are still claimed even if consolidation is processing."""
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"

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
            pool=pool,
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
    async def test_recover_own_tasks_resets_processing_to_pending(self, pool, clean_operations):
        """Test that recover_own_tasks resets processing tasks back to pending."""
        from hindsight_api.worker import WorkerPoller

        # Create tasks that were being processed by this worker (simulating a crash)
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
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
            pool=pool,
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
    async def test_recover_own_tasks_does_not_affect_other_workers(self, pool, clean_operations):
        """Test that recover_own_tasks only affects tasks from the same worker_id."""
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"

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
            pool=pool,
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
    async def test_recover_own_tasks_returns_zero_when_no_stale_tasks(self, pool, clean_operations):
        """Test that recover_own_tasks returns 0 when there are no stale tasks."""
        from hindsight_api.worker import WorkerPoller

        poller = WorkerPoller(
            pool=pool,
            worker_id="fresh-worker",
            executor=lambda x: None,
        )

        recovered_count = await poller.recover_own_tasks()
        assert recovered_count == 0


class TestConcurrentWorkers:
    """Tests for concurrent worker task claiming (FOR UPDATE SKIP LOCKED)."""

    @pytest.mark.asyncio
    async def test_concurrent_workers_claim_different_tasks(self, pool, clean_operations):
        """Test that multiple workers claim different tasks (no duplicates)."""
        from hindsight_api.worker import WorkerPoller

        # Create 10 pending tasks
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
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

        # Create 3 workers that will claim tasks concurrently
        workers_claimed: dict[str, list[str]] = {"worker-1": [], "worker-2": [], "worker-3": []}

        async def claim_for_worker(worker_id: str):
            poller = WorkerPoller(
                pool=pool,
                worker_id=worker_id,
                executor=lambda x: None,
            )
            claimed = await poller.claim_batch()
            workers_claimed[worker_id] = [task.operation_id for task in claimed]

        # Run all workers concurrently
        await asyncio.gather(
            claim_for_worker("worker-1"),
            claim_for_worker("worker-2"),
            claim_for_worker("worker-3"),
        )

        # Verify no duplicates - each task claimed by exactly one worker
        all_claimed = workers_claimed["worker-1"] + workers_claimed["worker-2"] + workers_claimed["worker-3"]
        assert len(all_claimed) == len(set(all_claimed)), "Duplicate task claimed by multiple workers!"

        # Verify total claimed equals available tasks (10)
        assert len(all_claimed) == 10, f"Expected 10 tasks claimed, got {len(all_claimed)}"

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
    async def test_workers_do_not_claim_already_processing_tasks(self, pool, clean_operations):
        """Test that workers skip tasks already being processed by another worker."""
        from hindsight_api.worker import WorkerPoller

        # Create tasks - some pending, some already processing
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"

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
            pool=pool,
            worker_id="new-worker",
            executor=lambda x: None,
        )

        claimed = await poller.claim_batch()
        assert len(claimed) == 3, "Worker should only claim pending tasks"

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
    async def test_sync_backend_handles_errors(self):
        """Test that SyncTaskBackend handles executor errors gracefully."""

        async def failing_executor(task_dict):
            raise ValueError("Test error")

        backend = SyncTaskBackend()
        backend.set_executor(failing_executor)
        await backend.initialize()

        # Should not raise, error is logged
        await backend.submit_task({"type": "test"})


class TestDynamicTenantDiscovery:
    """Tests for dynamic tenant discovery via TenantExtension."""

    @pytest.mark.asyncio
    async def test_poller_discovers_tenants_dynamically(self, pool, clean_operations):
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
            pool=pool,
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
    async def test_poller_picks_up_new_tenants_without_restart(self, pool, clean_operations):
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
            pool=pool,
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
    async def test_poller_without_tenant_extension_uses_public(self, pool, clean_operations):
        """Test that poller uses public schema when no tenant extension is configured."""
        from hindsight_api.worker import WorkerPoller

        # Create pending tasks
        bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
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
            pool=pool,
            worker_id="test-worker-1",
            executor=lambda x: None,
        )

        claimed = await poller.claim_batch()
        assert len(claimed) == 3

        # All tasks should have schema=None (public)
        for task in claimed:
            assert task.schema is None

    @pytest.mark.asyncio
    async def test_poller_with_custom_schema(self, pool):
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
                pool=pool,
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


async def test_worker_fire_and_forget_nonblocking(pool, clean_operations):
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

    async def blocking_executor(task_dict: dict):
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
        pool=pool,
        worker_id="test-worker",
        executor=blocking_executor,
        poll_interval_ms=50,  # Fast polling
        max_slots=10,
        consolidation_max_slots=2,
    )

    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"

    # Submit initial 2 tasks
    task_ids = []
    for i in range(2):
        op_id = uuid.uuid4()
        task_ids.append(str(op_id))
        payload = json.dumps({"type": "test", "operation_type": "retain", "operation_id": str(op_id), "bank_id": bank_id})
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

        # Verify tasks are in_flight
        async with poller._in_flight_lock:
            assert poller._in_flight_count == 2

        # NOW submit 2 more tasks WHILE the first 2 are still running
        for i in range(2):
            op_id = uuid.uuid4()
            task_ids.append(str(op_id))
            payload = json.dumps({"type": "test", "operation_type": "retain", "operation_id": str(op_id), "bank_id": bank_id})
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

        # Verify all 4 tasks are in-flight
        async with poller._in_flight_lock:
            assert poller._in_flight_count == 4

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


async def test_worker_slot_limits_enforced(pool, clean_operations):
    """Test that worker respects max_slots and won't exceed the limit."""
    from hindsight_api.worker.poller import WorkerPoller

    tasks_started = set()
    task_events = {}

    async def controlled_executor(task_dict: dict):
        op_id = task_dict["operation_id"]
        tasks_started.add(op_id)
        event = asyncio.Event()
        task_events[op_id] = event
        await event.wait()

    poller = WorkerPoller(
        pool=pool,
        worker_id="test-worker",
        executor=controlled_executor,
        poll_interval_ms=50,
        max_slots=3,  # Only allow 3 concurrent tasks
        consolidation_max_slots=1,
    )

    # Submit 10 tasks
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
    for i in range(10):
        op_id = uuid.uuid4()
        payload = json.dumps({"type": "test", "operation_type": "retain", "operation_id": str(op_id), "bank_id": bank_id})
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
            events_to_release = list(task_events.values())[completed:completed+3]
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
