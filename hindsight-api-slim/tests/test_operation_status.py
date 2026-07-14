"""
Tests that async operation statuses (pending, processing, completed, failed, cancelled)
are correctly exposed through list and get API endpoints.

Regression tests:
- Previously the API collapsed 'processing' into 'pending', hiding the real status.
- Cancel used to delete the operation row; now it sets status to 'cancelled'.
- Retry now accepts both 'failed' and 'cancelled' operations.
"""

import asyncio
import uuid
from datetime import datetime

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_bank_id():
    return f"op_status_test_{datetime.now().timestamp()}"


async def _ensure_bank(pool, bank_id: str) -> None:
    """Create a bank row if it doesn't already exist."""
    await pool.execute(
        """
        INSERT INTO banks (bank_id) VALUES ($1)
        ON CONFLICT (bank_id) DO NOTHING
        """,
        bank_id,
    )


async def _insert_operation(pool, bank_id: str, status: str) -> str:
    """Insert a test operation with the given status and return its ID."""
    op_id = uuid.uuid4()
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
        VALUES ($1, $2, 'retain', $3, '{"test": true}'::jsonb)
        """,
        op_id,
        bank_id,
        status,
    )
    return str(op_id)


@pytest.mark.asyncio
async def test_list_operations_returns_processing_status(api_client, memory, test_bank_id):
    """GET /operations should return 'processing' status, not collapse it to 'pending'."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    pending_id = await _insert_operation(pool, test_bank_id, "pending")
    processing_id = await _insert_operation(pool, test_bank_id, "processing")

    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/operations")
    assert response.status_code == 200
    ops = response.json()["operations"]

    statuses_by_id = {op["id"]: op["status"] for op in ops}
    assert statuses_by_id[pending_id] == "pending"
    assert statuses_by_id[processing_id] == "processing"


@pytest.mark.asyncio
async def test_list_operations_filter_by_processing(api_client, memory, test_bank_id):
    """Filtering by status=processing should only return processing operations."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    await _insert_operation(pool, test_bank_id, "pending")
    processing_id = await _insert_operation(pool, test_bank_id, "processing")

    response = await api_client.get(
        f"/v1/default/banks/{test_bank_id}/operations",
        params={"status": "processing"},
    )
    assert response.status_code == 200
    ops = response.json()["operations"]

    assert len(ops) == 1
    assert ops[0]["id"] == processing_id
    assert ops[0]["status"] == "processing"


@pytest.mark.asyncio
async def test_list_operations_filter_by_pending_excludes_processing(api_client, memory, test_bank_id):
    """Filtering by status=pending should NOT include processing operations."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    pending_id = await _insert_operation(pool, test_bank_id, "pending")
    await _insert_operation(pool, test_bank_id, "processing")

    response = await api_client.get(
        f"/v1/default/banks/{test_bank_id}/operations",
        params={"status": "pending"},
    )
    assert response.status_code == 200
    ops = response.json()["operations"]

    assert len(ops) == 1
    assert ops[0]["id"] == pending_id
    assert ops[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_get_operation_returns_processing_status(api_client, memory, test_bank_id):
    """GET /operations/{id} should return 'processing' status."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    processing_id = await _insert_operation(pool, test_bank_id, "processing")

    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/operations/{processing_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    assert data["operation_id"] == processing_id


@pytest.mark.asyncio
async def test_all_statuses_returned_correctly(api_client, memory, test_bank_id):
    """All four DB statuses should be returned as-is through both list and get endpoints."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    ids = {}
    for status in ("pending", "processing", "completed", "failed", "cancelled"):
        ids[status] = await _insert_operation(pool, test_bank_id, status)

    # Verify list endpoint
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/operations")
    assert response.status_code == 200
    ops = response.json()["operations"]
    statuses_by_id = {op["id"]: op["status"] for op in ops}

    for status, op_id in ids.items():
        assert statuses_by_id[op_id] == status, f"List: expected {status} for {op_id}, got {statuses_by_id[op_id]}"

    # Verify get endpoint for each
    for status, op_id in ids.items():
        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/operations/{op_id}")
        assert response.status_code == 200
        assert response.json()["status"] == status, f"Get: expected {status} for {op_id}"


@pytest.mark.asyncio
async def test_cancel_sets_cancelled_status(api_client, memory, test_bank_id):
    """DELETE /operations/{id} should set status to 'cancelled', not delete the row."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    op_id = await _insert_operation(pool, test_bank_id, "pending")

    # Cancel the operation
    response = await api_client.delete(f"/v1/default/banks/{test_bank_id}/operations/{op_id}")
    assert response.status_code == 200
    assert response.json()["success"] is True

    # Verify the operation still exists with 'cancelled' status
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/operations/{op_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"

    # Verify it shows up in list with cancelled filter
    response = await api_client.get(
        f"/v1/default/banks/{test_bank_id}/operations",
        params={"status": "cancelled"},
    )
    assert response.status_code == 200
    ops = response.json()["operations"]
    assert len(ops) == 1
    assert ops[0]["id"] == op_id


@pytest.mark.asyncio
async def test_retry_cancelled_operation(api_client, memory, test_bank_id):
    """POST /operations/{id}/retry should accept cancelled operations."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    op_id = await _insert_operation(pool, test_bank_id, "cancelled")

    # Retry the cancelled operation
    response = await api_client.post(f"/v1/default/banks/{test_bank_id}/operations/{op_id}/retry")
    assert response.status_code == 200
    assert response.json()["success"] is True

    # Verify the operation is now pending
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/operations/{op_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "pending"


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["failed", "cancelled"])
async def test_fresh_terminal_operation_keeps_payload_and_remains_retryable(
    api_client, memory, test_bank_id, terminal_status
):
    """Retry depends on the original task payload remaining present throughout retention."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)
    op_id = await _insert_operation(pool, test_bank_id, terminal_status)

    raw_before = await pool.fetchrow(
        "SELECT status, task_payload FROM async_operations WHERE operation_id = $1",
        uuid.UUID(op_id),
    )
    assert raw_before["status"] == terminal_status
    assert raw_before["task_payload"] is not None

    response = await api_client.post(f"/v1/default/banks/{test_bank_id}/operations/{op_id}/retry")
    assert response.status_code == 200

    raw_after = await pool.fetchrow(
        "SELECT status, task_payload FROM async_operations WHERE operation_id = $1",
        uuid.UUID(op_id),
    )
    assert raw_after["status"] == "pending"
    assert raw_after["task_payload"] is not None


@pytest.mark.asyncio
async def test_retry_does_not_acknowledge_operation_deleted_by_cleanup(api_client, memory, test_bank_id):
    """A pruning winner must turn a concurrent retry into 404, never false 200."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)
    op_id = await _insert_operation(pool, test_bank_id, "failed")
    op_uuid = uuid.UUID(op_id)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.fetchrow(
                "SELECT operation_id FROM async_operations WHERE operation_id = $1 FOR UPDATE",
                op_uuid,
            )
            retry_task = asyncio.create_task(
                api_client.post(f"/v1/default/banks/{test_bank_id}/operations/{op_id}/retry")
            )
            await asyncio.sleep(0)
            await conn.execute("DELETE FROM async_operations WHERE operation_id = $1", op_uuid)

    response = await retry_task
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_retry_rejects_non_retriable_statuses(api_client, memory, test_bank_id):
    """POST /operations/{id}/retry should reject pending, processing, and completed operations."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    for status in ("pending", "processing", "completed"):
        op_id = await _insert_operation(pool, test_bank_id, status)
        response = await api_client.post(f"/v1/default/banks/{test_bank_id}/operations/{op_id}/retry")
        assert response.status_code == 409, f"Expected 409 for {status}, got {response.status_code}"


@pytest.mark.asyncio
async def test_cancel_rejects_non_pending_operations(api_client, memory, test_bank_id):
    """DELETE /operations/{id} should only cancel pending operations."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    for status in ("processing", "completed", "failed"):
        op_id = await _insert_operation(pool, test_bank_id, status)
        response = await api_client.delete(f"/v1/default/banks/{test_bank_id}/operations/{op_id}")
        assert response.status_code == 409, f"Expected 409 for {status}, got {response.status_code}"
