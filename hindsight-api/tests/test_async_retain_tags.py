"""Unit tests for async retain tag propagation."""

from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.models import RequestContext


@pytest.mark.asyncio
async def test_submit_async_retain_includes_document_tags_in_task_payload():
    """submit_async_retain should include document_tags in queued task payload."""
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._authenticate_tenant = AsyncMock()
    engine._submit_async_operation = AsyncMock(return_value={"operation_id": "op-1"})

    request_context = RequestContext(tenant_id="tenant-a", api_key_id="key-a")
    contents = [{"content": "Async retain payload test."}]
    document_tags = ["scope:tools", "user:alice"]

    result = await MemoryEngine.submit_async_retain(
        engine,
        bank_id="bank-1",
        contents=contents,
        document_tags=document_tags,
        request_context=request_context,
    )

    assert result == {"operation_id": "op-1", "items_count": 1}
    engine._authenticate_tenant.assert_awaited_once_with(request_context)
    engine._submit_async_operation.assert_awaited_once()

    kwargs = engine._submit_async_operation.await_args.kwargs
    assert kwargs["bank_id"] == "bank-1"
    assert kwargs["operation_type"] == "retain"
    assert kwargs["task_type"] == "batch_retain"
    assert kwargs["task_payload"]["contents"] == contents
    assert kwargs["task_payload"]["document_tags"] == document_tags
    assert kwargs["task_payload"]["_tenant_id"] == "tenant-a"
    assert kwargs["task_payload"]["_api_key_id"] == "key-a"


@pytest.mark.asyncio
async def test_handle_batch_retain_forwards_document_tags_to_retain_batch_async():
    """Worker handler should forward document_tags from task payload."""
    engine = MemoryEngine.__new__(MemoryEngine)
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
