"""
Tests for per-bank serialisation of graph_maintenance at claim time.

Two concurrent graph_maintenance runs against the same bank produce no extra
work — the second convoys on the row locks the first holds — so claim_tasks
must (a) skip banks with a run already 'processing' and (b) claim at most one
per bank within a single batch, since submit-time dedupe_by_bank only inspects
'pending' rows and a bank can accumulate many pending rows while one is in
flight.

See issue #3230.
"""

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


async def _insert_op(pool, bank_id: str, op_type: str, status: str, worker_id: str | None = None) -> uuid.UUID:
    """Insert one operation row with a claimable payload."""
    op_id = uuid.uuid4()
    payload = json.dumps({"type": op_type, "bank_id": bank_id, "operation_id": str(op_id)})
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, worker_id)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6)
        """,
        op_id,
        bank_id,
        op_type,
        status,
        payload,
        worker_id,
    )
    return op_id


def _make_poller(backend):
    from hindsight_api.worker import WorkerPoller

    return WorkerPoller(
        backend=backend,
        worker_id="test-gm-claim-worker",
        executor=lambda x: None,
    )


async def _status_of(pool, op_id) -> str:
    return await pool.fetchval("SELECT status FROM async_operations WHERE operation_id = $1", op_id)


class TestGraphMaintenanceClaimSerialization:
    @pytest.mark.asyncio
    async def test_second_run_not_claimed_while_bank_busy(self, pool, backend, clean_operations):
        """A pending graph_maintenance for a bank with one already processing stays pending."""
        bank_a = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_a)
        await _insert_op(pool, bank_a, "graph_maintenance", "processing", worker_id="other-worker")
        pending_id = await _insert_op(pool, bank_a, "graph_maintenance", "pending")

        claimed = await _make_poller(backend).claim_batch()

        claimed_ids = {t.operation_id for t in claimed}
        assert str(pending_id) not in claimed_ids, "must not claim while the bank has a run in flight"
        assert await _status_of(pool, pending_id) == "pending"

    @pytest.mark.asyncio
    async def test_single_batch_claims_at_most_one_per_bank(self, pool, backend, clean_operations):
        """One claim batch takes at most one graph_maintenance per bank."""
        bank_a = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_a)
        op_ids = [await _insert_op(pool, bank_a, "graph_maintenance", "pending") for _ in range(5)]

        claimed = await _make_poller(backend).claim_batch()

        ours = [t for t in claimed if t.operation_id in {str(i) for i in op_ids}]
        assert len(ours) == 1, f"expected exactly one same-bank claim, got {len(ours)}"

    @pytest.mark.asyncio
    async def test_idle_bank_still_claimed_while_another_is_busy(self, pool, backend, clean_operations):
        """Serialisation is per bank: a busy bank A must not block idle bank B."""
        bank_a = f"test-worker-{uuid.uuid4().hex[:8]}"
        bank_b = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_a)
        await _ensure_bank(pool, bank_b)
        await _insert_op(pool, bank_a, "graph_maintenance", "processing", worker_id="other-worker")
        b_pending = await _insert_op(pool, bank_b, "graph_maintenance", "pending")

        claimed = await _make_poller(backend).claim_batch()

        claimed_ids = {t.operation_id for t in claimed}
        assert str(b_pending) in claimed_ids, "an idle bank must still be claimable"

    @pytest.mark.asyncio
    async def test_other_operation_types_unaffected(self, pool, backend, clean_operations):
        """The guard must not starve other operation types on the same bank."""
        bank_a = f"test-worker-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(pool, bank_a)
        await _insert_op(pool, bank_a, "graph_maintenance", "processing", worker_id="other-worker")
        retain_id = await _insert_op(pool, bank_a, "retain", "pending")

        claimed = await _make_poller(backend).claim_batch()

        claimed_ids = {t.operation_id for t in claimed}
        assert str(retain_id) in claimed_ids, "retain on the same bank must still be claimed"
