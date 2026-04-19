"""Test async batch retain with smart batching and parent-child operations."""

import asyncio
import json
import uuid

import pytest

from hindsight_api.extensions import RequestContext


async def _ensure_bank(pool, bank_id: str) -> None:
    """Upsert a minimal bank row so FK on async_operations passes."""
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


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
    await _ensure_bank(pool, bank_id)

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
    await _ensure_bank(pool, bank_id)

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
    await _ensure_bank(pool, bank_id)

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


async def _child_metadata(memory, bank_id: str, parent_operation_id: str, request_context):
    """Fetch the first child operation's result_metadata for a parent batch_retain."""
    parent = await memory.get_operation_status(
        bank_id=bank_id,
        operation_id=parent_operation_id,
        request_context=request_context,
    )
    assert parent["status"] == "completed", parent
    assert parent["child_operations"], "expected at least one child operation"
    child_id = parent["child_operations"][0]["operation_id"]
    child = await memory.get_operation_status(
        bank_id=bank_id,
        operation_id=child_id,
        request_context=request_context,
    )
    return child["result_metadata"]


@pytest.mark.asyncio
async def test_retain_records_user_provided_document_ids(memory, request_context):
    """User-supplied document_ids land in child op result_metadata.document_ids."""
    bank_id = "test_doc_ids_user_supplied"
    d1 = str(uuid.uuid4())
    d2 = str(uuid.uuid4())
    contents = [
        {"content": "User-supplied doc one content.", "document_id": d1},
        {"content": "User-supplied doc two content.", "document_id": d2},
    ]

    result = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
    )
    await asyncio.sleep(0.2)

    meta = await _child_metadata(memory, bank_id, result["operation_id"], request_context)
    assert "document_ids" in meta, meta
    assert set(meta["document_ids"]) == {d1, d2}


@pytest.mark.asyncio
async def test_retain_records_generated_document_id(memory, request_context):
    """With no document_ids supplied, retain records the single generated id."""
    bank_id = "test_doc_ids_generated"
    contents = [
        {"content": "Generated doc item one."},
        {"content": "Generated doc item two."},
    ]

    result = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
    )
    await asyncio.sleep(0.2)

    meta = await _child_metadata(memory, bank_id, result["operation_id"], request_context)
    assert "document_ids" in meta, meta
    assert isinstance(meta["document_ids"], list)
    assert len(meta["document_ids"]) == 1
    # Must be a valid UUID string (generated by the orchestrator)
    uuid.UUID(meta["document_ids"][0])


@pytest.mark.asyncio
async def test_retain_records_shared_document_id_once(memory, request_context):
    """Items sharing one document_id record it exactly once (idempotent set-append)."""
    bank_id = "test_doc_ids_shared"
    shared = str(uuid.uuid4())
    # Duplicate per-item doc_ids are rejected up front, so shared-doc mode
    # is exercised by a single item carrying the id.
    contents = [{"content": "Shared doc, chunk A.", "document_id": shared}]

    result = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
    )
    await asyncio.sleep(0.2)

    meta = await _child_metadata(memory, bank_id, result["operation_id"], request_context)
    assert meta.get("document_ids") == [shared]


@pytest.mark.asyncio
async def test_get_operation_status_include_payload(memory, request_context):
    """include_payload=True returns the original submission payload; default omits it."""
    bank_id = "test_include_payload"
    contents = [{"content": "Payload roundtrip test item."}]

    result = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
    )
    await asyncio.sleep(0.2)

    parent = await memory.get_operation_status(
        bank_id=bank_id,
        operation_id=result["operation_id"],
        request_context=request_context,
    )
    child_id = parent["child_operations"][0]["operation_id"]

    # Default: no payload
    without = await memory.get_operation_status(
        bank_id=bank_id,
        operation_id=child_id,
        request_context=request_context,
    )
    assert without.get("task_payload") is None

    # With flag: payload populated
    with_payload = await memory.get_operation_status(
        bank_id=bank_id,
        operation_id=child_id,
        request_context=request_context,
        include_payload=True,
    )
    payload = with_payload.get("task_payload")
    assert payload is not None, with_payload
    assert payload.get("bank_id") == bank_id
    assert payload.get("contents")
    assert payload["contents"][0]["content"] == "Payload roundtrip test item."


@pytest.mark.asyncio
async def test_submit_async_operation_leaves_claimable_row_when_submit_task_fails(memory):
    """Regression for the crash-window orphan bug fixed in #1091.

    Previously, _submit_async_operation INSERTed the async_operations row without
    task_payload, then called submit_task as a separate step to fill it in. If
    submit_task failed (crash, timeout, dropped connection) after the INSERT
    committed, the row was left with task_payload IS NULL and became permanently
    stuck because the worker claim query filters on task_payload IS NOT NULL.

    With the atomic INSERT, even if submit_task raises afterwards the row is born
    claimable. This test simulates the crash by forcing submit_task to raise.
    """
    bank_id = f"test_orphan_prevention_{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)

    async def failing_submit_task(_task_dict):
        raise RuntimeError("Simulated crash between INSERT and submit_task")

    memory._task_backend.submit_task = failing_submit_task  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Simulated crash"):
        await memory._submit_async_operation(
            bank_id=bank_id,
            operation_type="retain",
            task_type="batch_retain",
            task_payload={"contents": [{"content": "hello", "document_id": "d1"}]},
        )

    rows = await pool.fetch(
        """
        SELECT status, task_payload
        FROM async_operations
        WHERE bank_id = $1 AND operation_type = 'retain'
        """,
        bank_id,
    )
    assert len(rows) == 1, f"Expected exactly one retain row for bank_id={bank_id}, got {len(rows)}"
    row = rows[0]
    assert row["status"] == "pending"
    assert row["task_payload"] is not None, (
        "task_payload must be set atomically by the INSERT — a NULL here means "
        "the worker claim query (task_payload IS NOT NULL) will never pick this row up"
    )
    payload = json.loads(row["task_payload"])
    assert payload["type"] == "batch_retain"
    assert payload["bank_id"] == bank_id
    assert payload["contents"] == [{"content": "hello", "document_id": "d1"}]
