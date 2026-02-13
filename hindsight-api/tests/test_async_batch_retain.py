"""Test async batch retain with smart batching and parent-child operations."""

import asyncio
import json
import uuid

import pytest

from hindsight_api.extensions import RequestContext


@pytest.mark.asyncio
async def test_duplicate_document_ids_rejected_async(memory, request_context):
    """Test that async retain rejects batches with duplicate document_ids."""
    bank_id = "test_duplicate_async"
    contents = [
        {"content": "First item", "document_id": "doc1"},
        {"content": "Second item", "document_id": "doc2"},
        {"content": "Third item", "document_id": "doc1"},  # Duplicate!
    ]

    # Should raise ValueError due to duplicate document_ids
    with pytest.raises(ValueError, match="duplicate document_ids.*doc1"):
        await memory.submit_async_retain(
            bank_id=bank_id,
            contents=contents,
            request_context=request_context,
        )


@pytest.mark.asyncio
async def test_duplicate_document_ids_rejected_sync(memory, request_context):
    """Test that sync retain also rejects batches with duplicate document_ids."""
    bank_id = "test_duplicate_sync"
    contents = [
        {"content": "First item", "document_id": "doc1"},
        {"content": "Second item", "document_id": "doc1"},  # Duplicate!
    ]

    # Should raise ValueError due to duplicate document_ids
    with pytest.raises(ValueError, match="duplicate document_ids.*doc1"):
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=contents,
            request_context=request_context,
        )


@pytest.mark.asyncio
async def test_small_async_batch_no_splitting(memory, request_context):
    """Test that small async batches create parent with single child (simplified code path)."""
    bank_id = "test_small_async"
    contents = [{"content": "Alice works at Google", "document_id": f"doc{i}"} for i in range(5)]

    # Calculate total chars (should be well under threshold)
    total_chars = sum(len(item["content"]) for item in contents)
    assert total_chars < 10_000, "Test batch should be small"

    # Submit async retain
    result = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
    )

    # Verify we got an operation_id back
    assert "operation_id" in result
    assert "items_count" in result
    assert result["items_count"] == 5

    operation_id = result["operation_id"]

    # Wait for task to complete (SyncTaskBackend executes immediately)
    await asyncio.sleep(0.1)

    # Check operation status
    status = await memory.get_operation_status(
        bank_id=bank_id,
        operation_id=operation_id,
        request_context=request_context,
    )

    # Should be a parent operation with single child (simplified code path)
    assert status["status"] == "completed"
    assert status["operation_type"] == "batch_retain"
    assert "child_operations" in status
    assert status["result_metadata"]["num_sub_batches"] == 1  # Single sub-batch
    assert len(status["child_operations"]) == 1
    assert status["child_operations"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_large_async_batch_auto_splits(memory, request_context):
    """Test that large async batches automatically split into sub-batches with parent operation."""
    from hindsight_api.engine.memory_engine import count_tokens

    bank_id = "test_large_async"

    # Create a large batch that exceeds the threshold (10k tokens default)
    # Repeating "A"s gets heavily compressed by tokenizer, use varied content
    # Use ~22k chars per item = ~5.5k tokens per item, 2 items = ~11k tokens total (exceeds 10k)
    large_content = "The quick brown fox jumps over the lazy dog. " * 500  # ~22k chars = ~5.5k tokens
    contents = [{"content": large_content + f" item {i}", "document_id": f"doc{i}"} for i in range(2)]

    # Calculate total tokens (should exceed threshold)
    total_tokens = sum(count_tokens(item["content"]) for item in contents)
    assert total_tokens > 10_000, "Test batch should exceed threshold"

    # Submit async retain
    result = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
    )

    # Verify we got an operation_id back
    assert "operation_id" in result
    assert "items_count" in result
    assert result["items_count"] == 2

    parent_operation_id = result["operation_id"]

    # Wait for tasks to complete
    await asyncio.sleep(0.5)

    # Check parent operation status
    parent_status = await memory.get_operation_status(
        bank_id=bank_id,
        operation_id=parent_operation_id,
        request_context=request_context,
    )

    # Should be a parent operation with children
    assert parent_status["operation_type"] == "batch_retain"
    assert "child_operations" in parent_status
    assert "num_sub_batches" in parent_status["result_metadata"]
    assert parent_status["result_metadata"]["num_sub_batches"] >= 2  # Should split into at least 2 batches
    assert parent_status["result_metadata"]["items_count"] == 2

    # Verify child operations
    child_ops = parent_status["child_operations"]
    assert len(child_ops) >= 2, "Should have at least 2 child operations"

    # All children should be completed (SyncTaskBackend executes immediately)
    for child in child_ops:
        assert child["status"] == "completed"
        assert child["sub_batch_index"] is not None
        assert child["items_count"] > 0

    # Parent status should be aggregated as "completed"
    assert parent_status["status"] == "completed"


@pytest.mark.asyncio
async def test_parent_operation_status_aggregation_pending(memory, request_context):
    """Test that parent operation shows 'pending' when children are pending."""
    bank_id = "test_parent_pending"
    pool = await memory._get_pool()

    # Manually create a parent operation
    parent_id = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status)
            VALUES ($1, $2, $3, $4, $5)
            """,
            parent_id,
            bank_id,
            "batch_retain",
            json.dumps({"items_count": 20, "num_sub_batches": 2, "is_parent": True}),
            "pending",
        )

        # Create 2 child operations - one completed, one pending
        child1_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status)
            VALUES ($1, $2, $3, $4, $5)
            """,
            child1_id,
            bank_id,
            "retain",
            json.dumps(
                {
                    "items_count": 10,
                    "parent_operation_id": str(parent_id),
                    "sub_batch_index": 1,
                    "total_sub_batches": 2,
                }
            ),
            "completed",
        )

        child2_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status)
            VALUES ($1, $2, $3, $4, $5)
            """,
            child2_id,
            bank_id,
            "retain",
            json.dumps(
                {
                    "items_count": 10,
                    "parent_operation_id": str(parent_id),
                    "sub_batch_index": 2,
                    "total_sub_batches": 2,
                }
            ),
            "pending",
        )

    # Check parent status
    parent_status = await memory.get_operation_status(
        bank_id=bank_id,
        operation_id=str(parent_id),
        request_context=request_context,
    )

    # Parent should aggregate as "pending" since one child is still pending
    assert parent_status["status"] == "pending"
    assert len(parent_status["child_operations"]) == 2


@pytest.mark.asyncio
async def test_parent_operation_status_aggregation_failed(memory, request_context):
    """Test that parent operation shows 'failed' when any child fails."""
    bank_id = "test_parent_failed"
    pool = await memory._get_pool()

    # Manually create a parent operation
    parent_id = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status)
            VALUES ($1, $2, $3, $4, $5)
            """,
            parent_id,
            bank_id,
            "batch_retain",
            json.dumps({"items_count": 20, "num_sub_batches": 2, "is_parent": True}),
            "pending",
        )

        # Create 2 child operations - one completed, one failed
        child1_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status)
            VALUES ($1, $2, $3, $4, $5)
            """,
            child1_id,
            bank_id,
            "retain",
            json.dumps(
                {
                    "items_count": 10,
                    "parent_operation_id": str(parent_id),
                    "sub_batch_index": 1,
                    "total_sub_batches": 2,
                }
            ),
            "completed",
        )

        child2_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status, error_message)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            child2_id,
            bank_id,
            "retain",
            json.dumps(
                {
                    "items_count": 10,
                    "parent_operation_id": str(parent_id),
                    "sub_batch_index": 2,
                    "total_sub_batches": 2,
                }
            ),
            "failed",
            "Test error",
        )

    # Check parent status
    parent_status = await memory.get_operation_status(
        bank_id=bank_id,
        operation_id=str(parent_id),
        request_context=request_context,
    )

    # Parent should aggregate as "failed" since one child failed
    assert parent_status["status"] == "failed"
    assert len(parent_status["child_operations"]) == 2

    # Verify child with error is included
    failed_child = [c for c in parent_status["child_operations"] if c["status"] == "failed"][0]
    assert failed_child["error_message"] == "Test error"


@pytest.mark.asyncio
async def test_parent_operation_status_aggregation_completed(memory, request_context):
    """Test that parent operation shows 'completed' when all children are completed."""
    bank_id = "test_parent_completed"
    pool = await memory._get_pool()

    # Manually create a parent operation
    parent_id = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status)
            VALUES ($1, $2, $3, $4, $5)
            """,
            parent_id,
            bank_id,
            "batch_retain",
            json.dumps({"items_count": 20, "num_sub_batches": 2, "is_parent": True}),
            "pending",
        )

        # Create 2 child operations - both completed
        child1_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status)
            VALUES ($1, $2, $3, $4, $5)
            """,
            child1_id,
            bank_id,
            "retain",
            json.dumps(
                {
                    "items_count": 10,
                    "parent_operation_id": str(parent_id),
                    "sub_batch_index": 1,
                    "total_sub_batches": 2,
                }
            ),
            "completed",
        )

        child2_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status)
            VALUES ($1, $2, $3, $4, $5)
            """,
            child2_id,
            bank_id,
            "retain",
            json.dumps(
                {
                    "items_count": 10,
                    "parent_operation_id": str(parent_id),
                    "sub_batch_index": 2,
                    "total_sub_batches": 2,
                }
            ),
            "completed",
        )

    # Check parent status
    parent_status = await memory.get_operation_status(
        bank_id=bank_id,
        operation_id=str(parent_id),
        request_context=request_context,
    )

    # Parent should aggregate as "completed" since all children are completed
    assert parent_status["status"] == "completed"
    assert len(parent_status["child_operations"]) == 2
    assert all(c["status"] == "completed" for c in parent_status["child_operations"])


@pytest.mark.asyncio
async def test_config_retain_batch_tokens_respected(memory, request_context):
    """Test that the retain_batch_tokens config setting is respected."""
    from hindsight_api.config import get_config
    from hindsight_api.engine.memory_engine import count_tokens

    bank_id = "test_config_batch_tokens"
    config = get_config()

    # Check that config has the retain_batch_tokens setting
    assert hasattr(config, "retain_batch_tokens")
    assert config.retain_batch_tokens > 0

    # Create a batch that's just under the threshold
    # Use content that produces roughly half the token limit per item
    content_size = config.retain_batch_tokens * 2  # chars (rough estimate: 1 token ~= 4 chars)
    contents = [{"content": "A" * content_size, "document_id": f"doc{i}"} for i in range(2)]

    total_tokens = sum(count_tokens(item["content"]) for item in contents)
    # Should be equal to threshold (boundary case, no splitting since we use > not >=)
    assert total_tokens <= config.retain_batch_tokens

    # Submit - should NOT split
    result = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
    )

    # Wait for completion
    await asyncio.sleep(0.1)

    # Check status - should be a parent with single child (even for small batches)
    status = await memory.get_operation_status(
        bank_id=bank_id,
        operation_id=result["operation_id"],
        request_context=request_context,
    )

    # Even small batches use parent-child pattern now (simpler code path)
    assert "child_operations" in status
    assert status["result_metadata"]["num_sub_batches"] == 1
