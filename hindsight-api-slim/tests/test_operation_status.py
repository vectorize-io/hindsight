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


async def _insert_operation(
    pool, bank_id: str, status: str, operation_type: str = "retain", result_metadata: str = "{}"
) -> str:
    """Insert a test operation with the given status and return its ID.

    ``result_metadata`` is the column a batch parent is marked in (``is_parent``), which is
    what exclude_parents filters on.
    """
    op_id = uuid.uuid4()
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, status, operation_type, task_payload, result_metadata)
        VALUES ($1, $2, $3, $4, '{"test": true}'::jsonb, $5::jsonb)
        """,
        op_id,
        bank_id,
        status,
        operation_type,
        result_metadata,
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
async def test_list_operations_active_only_totals_every_non_terminal_row(api_client, memory, test_bank_id):
    """`total` counts every non-terminal row, not the one-row page it returns.

    A leaked terminal row would keep a client waiting on work that already finished.
    """
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    for status in ("pending", "pending", "pending", "processing", "processing"):
        await _insert_operation(pool, test_bank_id, status)
    for status in ("completed", "completed", "failed", "cancelled"):
        await _insert_operation(pool, test_bank_id, status)

    response = await api_client.get(
        f"/v1/default/banks/{test_bank_id}/operations",
        params={"active_only": "true", "limit": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 5
    assert len(body["operations"]) == 1
    assert body["operations"][0]["status"] in ("pending", "processing")


@pytest.mark.asyncio
async def test_list_operations_active_only_conjoins_with_the_other_filters(api_client, memory, test_bank_id):
    """active_only narrows the WHERE clause the other filters share; it never replaces one.

    Winning over status, type or exclude_parents would report a backlog nobody asked about.
    """
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    await _insert_operation(pool, test_bank_id, "pending", "batch_retain", '{"is_parent": true}')
    pending_retain = await _insert_operation(pool, test_bank_id, "pending")
    processing_retain = await _insert_operation(pool, test_bank_id, "processing")
    await _insert_operation(pool, test_bank_id, "completed")
    pending_consolidation = await _insert_operation(pool, test_bank_id, "pending", "consolidation")
    await _insert_operation(pool, test_bank_id, "failed", "consolidation")

    url = f"/v1/default/banks/{test_bank_id}/operations"
    narrowed = await api_client.get(
        url,
        params={"active_only": "true", "status": "pending", "type": "retain", "exclude_parents": "true"},
    )
    assert narrowed.status_code == 200
    body = narrowed.json()
    assert body["total"] == 1
    assert [op["id"] for op in body["operations"]] == [pending_retain]

    active_leaves = await api_client.get(url, params={"active_only": "true", "exclude_parents": "true"})
    body = active_leaves.json()
    assert body["total"] == 3
    assert {op["id"] for op in body["operations"]} == {pending_retain, processing_retain, pending_consolidation}


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
async def test_retry_rejects_batch_retain_parent(api_client, memory, test_bank_id):
    """A failed batch_retain parent with no retryable work must not be revived into a re-stranded state.

    The parent is a payload-less status aggregator, so retrying it only makes sense
    when it still has failed/cancelled children to re-run (see the retry tests in
    test_async_batch_retain.py). This parent has no children at all, so there is
    nothing to re-queue — reviving it would strand it 'pending' forever (issue #2985).
    The 409 should point the caller at resubmit + delete instead.
    """
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    op_id = uuid.uuid4()
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
        VALUES ($1, $2, 'batch_retain', 'failed', NULL)
        """,
        op_id,
        test_bank_id,
    )

    response = await api_client.post(f"/v1/default/banks/{test_bank_id}/operations/{op_id}/retry")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "batch_retain parent" in detail
    assert "delete" in detail.lower()

    # Guard must not have mutated the row.
    row = await pool.fetchrow(
        "SELECT status FROM async_operations WHERE operation_id = $1",
        op_id,
    )
    assert row["status"] == "failed"

    # But it remains deletable as the sanctioned cleanup path.
    response = await api_client.delete(f"/v1/default/banks/{test_bank_id}/operations/{op_id}/delete")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_cancel_rejects_terminal_operations(api_client, memory, test_bank_id):
    """DELETE /operations/{id} should refuse operations that already finished."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    for status in ("completed", "failed", "cancelled"):
        op_id = await _insert_operation(pool, test_bank_id, status)
        response = await api_client.delete(f"/v1/default/banks/{test_bank_id}/operations/{op_id}")
        assert response.status_code == 409, f"Expected 409 for {status}, got {response.status_code}"


@pytest.mark.asyncio
async def test_cancel_processing_operation(api_client, memory, test_bank_id):
    """DELETE /operations/{id} should cancel an in-flight operation (issue #4131).

    This is the only way to clear a row stranded in 'processing' by a worker that
    was killed before it could write a terminal status.
    """
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    op_id = await _insert_operation(pool, test_bank_id, "processing")

    response = await api_client.delete(f"/v1/default/banks/{test_bank_id}/operations/{op_id}")
    assert response.status_code == 200
    assert response.json()["success"] is True

    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/operations/{op_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"

    # And the stranded work can be re-queued once it is unwedged.
    response = await api_client.post(f"/v1/default/banks/{test_bank_id}/operations/{op_id}/retry")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_cancelling_last_child_terminalises_parent(api_client, memory, test_bank_id):
    """A cancelled child must not strand its batch_retain parent in 'processing' (issue #4131).

    'cancelled' is a done state for the sibling rollup, and the cancel itself performs the
    rollup — otherwise nothing else ever writes the parent's terminal status and the batch
    sits in 'processing' forever, which is the exact wedge this endpoint clears.
    """
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    parent_id = uuid.uuid4()
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status, result_metadata)
        VALUES ($1, $2, 'batch_retain', 'processing', '{"is_parent": true}'::jsonb)
        """,
        parent_id,
        test_bank_id,
    )

    child_ids = []
    for status in ("completed", "processing"):
        child_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, result_metadata)
            VALUES ($1, $2, 'retain', $3, $4::jsonb)
            """,
            child_id,
            test_bank_id,
            status,
            f'{{"parent_operation_id": "{parent_id}"}}',
        )
        child_ids.append(child_id)

    # Cancel the one child still in flight — it is the last outstanding sibling.
    response = await api_client.delete(f"/v1/default/banks/{test_bank_id}/operations/{child_ids[1]}")
    assert response.status_code == 200

    parent_status = await pool.fetchval("SELECT status FROM async_operations WHERE operation_id = $1", parent_id)
    assert parent_status == "cancelled", "parent must not be left in 'processing'"


@pytest.mark.asyncio
async def test_delete_removes_terminal_operation(api_client, memory, test_bank_id):
    """DELETE /operations/{id}/delete should remove a failed operation's row entirely."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    op_id = await _insert_operation(pool, test_bank_id, "failed")

    response = await api_client.delete(f"/v1/default/banks/{test_bank_id}/operations/{op_id}/delete")
    assert response.status_code == 200
    assert response.json()["success"] is True

    # get_operation_status returns a not_found dict (not HTTP 404)
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/operations/{op_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "not_found"

    response = await api_client.get(
        f"/v1/default/banks/{test_bank_id}/operations",
        params={"status": "failed"},
    )
    assert response.status_code == 200
    assert response.json()["operations"] == []


@pytest.mark.asyncio
async def test_delete_rejects_non_terminal_statuses(api_client, memory, test_bank_id):
    """DELETE /operations/{id}/delete should reject pending and processing operations."""
    pool = memory._pool
    await _ensure_bank(pool, test_bank_id)

    for status in ("pending", "processing"):
        op_id = await _insert_operation(pool, test_bank_id, status)
        response = await api_client.delete(f"/v1/default/banks/{test_bank_id}/operations/{op_id}/delete")
        assert response.status_code == 409, f"Expected 409 for {status}, got {response.status_code}"


@pytest.mark.asyncio
async def test_delete_wrong_bank_returns_404(api_client, memory, test_bank_id):
    """DELETE under a different valid bank must 404 and leave the row intact."""
    pool = memory._pool
    bank_a = test_bank_id
    bank_b = f"{test_bank_id}_other"
    await _ensure_bank(pool, bank_a)
    await _ensure_bank(pool, bank_b)

    op_id = await _insert_operation(pool, bank_a, "failed")

    response = await api_client.delete(f"/v1/default/banks/{bank_b}/operations/{op_id}/delete")
    assert response.status_code == 404

    response = await api_client.get(f"/v1/default/banks/{bank_a}/operations/{op_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
