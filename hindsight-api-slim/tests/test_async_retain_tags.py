"""Unit tests for async retain tag propagation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.models import RequestContext


@pytest.mark.asyncio
async def test_submit_async_retain_includes_document_tags_in_task_payload():
    """submit_async_retain should include document_tags in queued task payload.

    submit_async_batch_retain inserts the parent + all children inline inside
    a single transaction (not via _submit_async_operation), then notifies the
    task backend after commit. The test verifies document_tags propagates
    through to the task backend's submit_task call, which is the post-commit
    notification that drives SyncTaskBackend in tests and is a no-op for
    BrokerTaskBackend / WorkerTaskBackend in production.
    """
    import json

    engine = MemoryEngine.__new__(MemoryEngine)
    engine._initialized = True
    engine._authenticate_tenant = AsyncMock()
    engine._operation_validator = None
    # Children are now inserted inline (no _submit_async_operation hop), and
    # submit_task fires post-commit. Mock both so the inline path runs cleanly
    # without the test needing real DB or task backend.
    engine._task_backend = AsyncMock()
    engine._task_backend.submit_task = AsyncMock()

    # Mock the pool and connection for parent + child INSERTs in one transaction.
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.transaction = MagicMock()
    mock_conn.transaction.return_value.__aenter__ = AsyncMock()
    # __aexit__ must return falsy or `async with` will swallow exceptions —
    # AsyncMock's default return is a truthy MagicMock.
    mock_conn.transaction.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_pool = AsyncMock()
    mock_pool.acquire = AsyncMock(return_value=mock_conn)
    mock_pool.release = AsyncMock()

    engine._get_pool = AsyncMock(return_value=mock_pool)
    # _backend used by bank_utils (patched below) and _get_backend for acquire_with_retry
    engine._backend = mock_pool
    engine._get_backend = AsyncMock(return_value=mock_pool)
    # Ensure mock_pool is not treated as a DatabaseBackend/BudgetedPool wrapper
    # (AsyncMock returns truthy for any attr; explicitly set _wraps_backend to False)
    mock_pool._wraps_backend = False

    request_context = RequestContext(tenant_id="tenant-a", api_key_id="key-a")
    contents = [{"content": "Async retain payload test."}]
    document_tags = ["scope:tools", "user:alice"]

    # Stub the lazy bank-create/default-template hook to a no-op (created=False)
    # so the inline transaction path runs against the mock connection without
    # real DB work. The hook itself is covered by dedicated tests.
    engine._ensure_bank_exists = AsyncMock(return_value=False)

    result = await MemoryEngine.submit_async_retain(
        engine,
        bank_id="bank-1",
        contents=contents,
        document_tags=document_tags,
        request_context=request_context,
    )

    # Check result structure
    assert "operation_id" in result
    assert "items_count" in result
    assert result["items_count"] == 1

    # Verify authentication was called
    engine._authenticate_tenant.assert_awaited_once_with(request_context)

    # The parent + child INSERTs both went through mock_conn.execute. There
    # should be exactly two: one for the parent (no task_payload), one for the
    # single child (with task_payload). The child INSERT serializes
    # full_payload — which carries document_tags — to JSON.
    assert mock_conn.execute.await_count == 2, (
        f"Expected two INSERTs (parent + child), got {mock_conn.execute.await_count}"
    )

    # Verify the post-commit submit_task fires once with a payload containing
    # the document_tags (this is what gets handed to SyncTaskBackend in tests,
    # and what carries the work into the worker in production).
    engine._task_backend.submit_task.assert_awaited_once()
    full_payload = engine._task_backend.submit_task.await_args.args[0]
    assert full_payload["type"] == "batch_retain"
    assert full_payload["bank_id"] == "bank-1"
    assert full_payload["contents"] == contents
    assert full_payload["document_tags"] == document_tags
    assert full_payload["_tenant_id"] == "tenant-a"
    assert full_payload["_api_key_id"] == "key-a"

    # Cross-check: the child INSERT's task_payload column also contains
    # the document_tags (same JSON the worker poller would later read).
    child_insert_args = mock_conn.execute.await_args_list[1].args
    # Positional: (sql, operation_id, bank_id, operation_type, result_metadata,
    #              status, task_payload_json)
    child_task_payload_json = child_insert_args[6]
    child_task_payload = json.loads(child_task_payload_json)
    assert child_task_payload["document_tags"] == document_tags
    assert child_task_payload["_tenant_id"] == "tenant-a"
    assert child_task_payload["_api_key_id"] == "key-a"


@pytest.mark.asyncio
async def test_handle_batch_retain_forwards_document_tags_to_retain_batch_async():
    """Worker handler should forward document_tags from task payload."""
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._initialized = True
    engine.retain_batch_async = AsyncMock(return_value={"items_count": 1})

    task_dict = {
        "bank_id": "bank-1",
        "contents": [{"content": "Forward tags test."}],
        "document_tags": ["scope:client"],
        "_tenant_id": "tenant-a",
        "_api_key_id": "key-a",
    }

    await MemoryEngine._handle_batch_retain(engine, task_dict)

    engine.retain_batch_async.assert_awaited_once()
    kwargs = engine.retain_batch_async.await_args.kwargs
    assert kwargs["bank_id"] == "bank-1"
    assert kwargs["contents"] == task_dict["contents"]
    assert kwargs["document_tags"] == ["scope:client"]

    request_context = kwargs["request_context"]
    assert request_context.internal is True
    assert request_context.user_initiated is True
    assert request_context.tenant_id == "tenant-a"
    assert request_context.api_key_id == "key-a"
