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
            batch_size=10,
        )

        claimed = await poller.claim_batch()
        assert len(claimed) == 3

        # Verify tasks are marked as processing with worker_id
        rows = await pool.fetch(
            "SELECT status, worker_id FROM async_operations WHERE bank_id = $1",
            bank_id,
        )
        for row in rows:
            assert row["status"] == "processing"
            assert row["worker_id"] == "test-worker-1"

    @pytest.mark.asyncio
    async def test_claim_batch_respects_batch_size(self, pool, clean_operations):
        """Test that claim_batch respects the batch_size limit."""
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

        # Claim with batch_size=3
        poller = WorkerPoller(
            pool=pool,
            worker_id="test-worker-1",
            executor=lambda x: None,
            batch_size=3,
        )

        claimed = await poller.claim_batch()
        assert len(claimed) == 3

    @pytest.mark.asyncio
    async def test_execute_task_marks_completed(self, pool, clean_operations):
        """Test that successful task execution marks task as completed."""
        from hindsight_api.worker import WorkerPoller

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

        # Execute the task
        task_dict = json.loads(payload)
        await poller.execute_task(str(op_id), task_dict)

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

        # Execute (should fail and retry)
        task_dict = json.loads(payload)
        await poller.execute_task(str(op_id), task_dict)

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

        # Execute (should fail permanently)
        task_dict = json.loads(payload)
        await poller.execute_task(str(op_id), task_dict)

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
            batch_size=10,
        )

        claimed = await poller.claim_batch()

        # Should only claim the consolidation for the other bank
        assert len(claimed) == 1
        claimed_op_id, claimed_payload = claimed[0]
        assert claimed_op_id == str(other_op_id)
        assert claimed_payload["bank_id"] == other_bank_id

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
            batch_size=10,
        )

        claimed = await poller.claim_batch()

        # Should claim the retain task (non-consolidation tasks are unaffected)
        assert len(claimed) == 1
        claimed_op_id, _ = claimed[0]
        assert claimed_op_id == str(retain_op_id)


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
                batch_size=5,  # Each worker tries to claim 5
            )
            claimed = await poller.claim_batch()
            workers_claimed[worker_id] = [op_id for op_id, _ in claimed]

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
            batch_size=10,
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
