"""Worker quarantine tests for unclaimable async operations."""

import json
import uuid
from typing import Any, cast

import pytest
import pytest_asyncio

pytestmark = pytest.mark.xdist_group("worker_tests")


async def _ensure_bank(pool, bank_id: str) -> None:
    """Upsert a minimal bank row so FK on async_operations passes."""
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


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
    """Remove rows that can interfere with worker claim/quarantine tests."""
    await pool.execute("DELETE FROM async_operations WHERE bank_id LIKE 'test-quarantine-%'")
    yield
    await pool.execute("DELETE FROM async_operations WHERE bank_id LIKE 'test-quarantine-%'")


class _NonPostgresBackend:
    backend_type = "oracle"

    def acquire(self):  # pragma: no cover - guard should return before DB access
        raise AssertionError("non-PostgreSQL quarantine path should not acquire a connection")


class TestWorkerPollerQuarantine:
    @pytest.mark.asyncio
    async def test_quarantine_guard_skips_non_postgresql_backend(self):
        """The PostgreSQL JSONB quarantine query must not run on other backends."""
        from hindsight_api.worker import WorkerPoller

        async def noop_executor(_task_dict):
            return None

        poller = WorkerPoller(
            backend=cast(Any, _NonPostgresBackend()),
            worker_id="test-worker-quarantine",
            executor=noop_executor,
            slot_reservations={},
        )

        assert await poller._quarantine_unclaimable_pending_operations([None, "tenant_a"]) == 0

    @pytest.mark.asyncio
    async def test_claim_batch_quarantines_null_payload_batch_parent_without_children(
        self, pool, backend, clean_operations
    ):
        """A parent with no executable payload and no children cannot ever be claimed.

        This is a poison queue shape observed after crash recovery: a pending
        batch_retain parent row with task_payload=NULL but no child rows. The
        poller should move it out of the normal pending lane instead of letting
        queue health report ordinary backlog forever.
        """
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-quarantine-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        parent_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status, task_payload)
            VALUES ($1, $2, 'batch_retain', $3::jsonb, 'pending', NULL)
            """,
            parent_id,
            bank_id,
            json.dumps({"items_count": 5, "num_sub_batches": 1, "is_parent": True}),
        )

        async def executor(_task_dict):  # pragma: no cover - must not run
            raise AssertionError("unclaimable null-payload parent should not execute")

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-quarantine",
            executor=executor,
            slot_reservations={},
        )

        claimed = await poller.claim_batch()
        assert claimed == []

        row = await pool.fetchrow(
            "SELECT status, error_message, task_payload, result_metadata, completed_at FROM async_operations WHERE operation_id = $1",
            parent_id,
        )
        assert row["status"] == "failed"
        assert row["task_payload"] is None
        assert row["completed_at"] is None
        assert "unclaimable" in row["error_message"]
        assert "task_payload is NULL" in row["error_message"]
        metadata = json.loads(row["result_metadata"]) if isinstance(row["result_metadata"], str) else row["result_metadata"]
        assert metadata["quarantined"] is True
        assert metadata["quarantine_reason"] == "task_payload_null"

    @pytest.mark.asyncio
    async def test_claim_batch_keeps_null_payload_batch_parent_with_children_pending(
        self, pool, backend, clean_operations
    ):
        """Valid batch parents are aggregators and may have task_payload=NULL.

        A parent with children is not executable itself, but it is not poison:
        child completion/failure should reconcile the parent. The quarantine
        guard must not fail these legitimate aggregate rows.
        """
        from hindsight_api.worker import WorkerPoller

        bank_id = f"test-quarantine-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_id)
        parent_id = uuid.uuid4()
        child_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status, task_payload)
            VALUES ($1, $2, 'batch_retain', $3::jsonb, 'pending', NULL)
            """,
            parent_id,
            bank_id,
            json.dumps({"items_count": 1, "num_sub_batches": 1, "is_parent": True}),
        )
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status, task_payload)
            VALUES ($1, $2, 'retain', $3::jsonb, 'pending', $4::jsonb)
            """,
            child_id,
            bank_id,
            json.dumps({"parent_operation_id": str(parent_id), "sub_batch_index": 1, "total_sub_batches": 1}),
            json.dumps({"type": "batch_retain", "bank_id": bank_id, "contents": [{"content": "hello"}]}),
        )

        async def noop_executor(_task_dict):
            return None

        poller = WorkerPoller(
            backend=backend,
            worker_id="test-worker-quarantine",
            executor=noop_executor,
            max_slots=1,
            slot_reservations={},
        )

        claimed = await poller.claim_batch()
        assert len(claimed) == 1
        assert claimed[0].operation_id == str(child_id)

        parent = await pool.fetchrow(
            "SELECT status, error_message FROM async_operations WHERE operation_id = $1",
            parent_id,
        )
        assert parent["status"] == "pending"
        assert parent["error_message"] is None
