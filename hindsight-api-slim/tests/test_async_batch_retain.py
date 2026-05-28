"""Test async batch retain with smart batching and parent-child operations."""

import asyncio
import json
import uuid

import pytest

from hindsight_api.extensions import RequestContext

# These tests submit async operations and rely on the engine-owned worker to
# drain them. test_worker.py drives its own WorkerPoller.claim_batch() against
# the same pool, so running the two files on different xdist workers causes
# them to steal each other's pending rows. Share the "worker_tests" group so
# they serialize on the same xdist process.
pytestmark = pytest.mark.xdist_group("worker_tests")


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
@pytest.mark.timeout(600)
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
async def test_operation_status_exposes_retry_count_and_next_retry_at(memory, request_context):
    """get_operation_status and list_operations return retry_count and next_retry_at.

    Consumers need these to distinguish a freshly-queued pending task from
    one that's parked for a future retry (e.g. because an extension raised
    DeferOperation). Without them, "pending" is ambiguous and callers can't
    render a helpful "deferred until X" state.
    """
    from datetime import datetime, timedelta, timezone

    bank_id = "test_retry_fields"
    result = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=[{"content": "retry-fields test item"}],
        request_context=request_context,
    )
    await asyncio.sleep(0.1)
    parent_id = result["operation_id"]
    child_id = None

    # Get the child op (the batch_retain parent holds a single child in the
    # sync/simplified path used by SyncTaskBackend tests).
    parent_status = await memory.get_operation_status(
        bank_id=bank_id,
        operation_id=parent_id,
        request_context=request_context,
    )
    assert "retry_count" in parent_status
    assert "next_retry_at" in parent_status
    assert parent_status["retry_count"] == 0
    # Completed tasks should have next_retry_at cleared on the row (or the
    # status field doesn't include it meaningfully), so we don't assert a
    # specific value here — only that the key is present.
    if parent_status.get("child_operations"):
        child_id = parent_status["child_operations"][0]["operation_id"]

    # list_operations also exposes both fields
    listed = await memory.list_operations(
        bank_id=bank_id,
        request_context=request_context,
        limit=10,
        offset=0,
    )
    assert listed["operations"], listed
    for op in listed["operations"]:
        assert "retry_count" in op
        assert "next_retry_at" in op
        assert isinstance(op["retry_count"], int)

    # Simulate a deferred op: set next_retry_at to 15 min in the future for
    # the child row directly in the DB, then fetch via the API and confirm
    # the value round-trips as an ISO-8601 string.
    if child_id:
        pool = await memory._get_pool()
        future = datetime.now(timezone.utc) + timedelta(minutes=15)
        await pool.execute(
            "UPDATE async_operations SET status = 'pending', next_retry_at = $1, retry_count = 2 WHERE operation_id = $2",
            future,
            uuid.UUID(child_id),
        )
        fetched = await memory.get_operation_status(
            bank_id=bank_id,
            operation_id=child_id,
            request_context=request_context,
        )
        assert fetched["retry_count"] == 2
        assert fetched["next_retry_at"] is not None
        # Round-trip tolerance: within 1 second.
        parsed = datetime.fromisoformat(fetched["next_retry_at"])
        assert abs((parsed - future).total_seconds()) < 1.0


@pytest.mark.asyncio
async def test_list_operations_exclude_parents(memory, request_context):
    """list_operations with exclude_parents=True hides parent batch operations."""
    bank_id = "test_exclude_parents"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)

    # Create a parent operation (is_parent=True)
    parent_id = uuid.uuid4()
    child_id = uuid.uuid4()
    standalone_id = uuid.uuid4()

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status)
            VALUES ($1, $2, $3, $4, $5)
            """,
            parent_id,
            bank_id,
            "batch_retain",
            json.dumps({"items_count": 10, "num_sub_batches": 1, "is_parent": True}),
            "completed",
        )
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status)
            VALUES ($1, $2, $3, $4, $5)
            """,
            child_id,
            bank_id,
            "retain",
            json.dumps(
                {"items_count": 10, "parent_operation_id": str(parent_id), "sub_batch_index": 1, "total_sub_batches": 1}
            ),
            "completed",
        )
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status)
            VALUES ($1, $2, $3, $4, $5)
            """,
            standalone_id,
            bank_id,
            "consolidation",
            json.dumps({}),
            "completed",
        )

    # Without exclude_parents: all 3 operations visible
    all_ops = await memory.list_operations(
        bank_id=bank_id,
        request_context=request_context,
        limit=10,
        offset=0,
    )
    all_ids = {op["id"] for op in all_ops["operations"]}
    assert str(parent_id) in all_ids
    assert str(child_id) in all_ids
    assert str(standalone_id) in all_ids
    assert all_ops["total"] == 3

    # With exclude_parents: parent is hidden
    filtered_ops = await memory.list_operations(
        bank_id=bank_id,
        request_context=request_context,
        limit=10,
        offset=0,
        exclude_parents=True,
    )
    filtered_ids = {op["id"] for op in filtered_ops["operations"]}
    assert str(parent_id) not in filtered_ids
    assert str(child_id) in filtered_ids
    assert str(standalone_id) in filtered_ids
    assert filtered_ops["total"] == 2


@pytest.mark.asyncio
async def test_request_context_retry_count_propagated_to_validator(memory_no_llm_verify, request_context):
    """_handle_batch_retain forwards the task's _retry_count as
    RequestContext.retry_count, so validator extensions can compute
    exponential backoff without querying async_operations themselves.
    """
    from hindsight_api.extensions import (
        OperationValidatorExtension,
        RecallContext,
        ReflectContext,
        RetainContext,
        ValidationResult,
    )

    captured: dict[str, int] = {"retry_count": -1}

    class CapturingValidator(OperationValidatorExtension):
        def __init__(self):
            super().__init__({})

        async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
            captured["retry_count"] = ctx.request_context.retry_count
            return ValidationResult.accept()

        async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
            return ValidationResult.accept()

        async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
            return ValidationResult.accept()

    memory_no_llm_verify._operation_validator = CapturingValidator()

    bank_id = f"test-retry-propagate-{uuid.uuid4().hex[:8]}"
    pool = await memory_no_llm_verify._get_pool()
    await _ensure_bank(pool, bank_id)

    task_dict = {
        "type": "batch_retain",
        "bank_id": bank_id,
        "contents": [{"content": "retry-propagate test"}],
        "_tenant_id": "default",
        "_retry_count": 3,  # simulate 3rd retry
    }
    await memory_no_llm_verify._handle_batch_retain(task_dict)

    assert captured["retry_count"] == 3, (
        f"Validator should see retry_count=3 from task_dict['_retry_count']; got {captured['retry_count']}"
    )

    # Default (missing _retry_count key) must surface as 0, not raise.
    captured["retry_count"] = -1
    task_dict_no_retry = {
        "type": "batch_retain",
        "bank_id": bank_id,
        "contents": [{"content": "retry-propagate default test"}],
        "_tenant_id": "default",
    }
    await memory_no_llm_verify._handle_batch_retain(task_dict_no_retry)
    assert captured["retry_count"] == 0


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


# ---------------------------------------------------------------------------
# Regression tests for issue #1795: submit_async_retain must NOT fragment a
# single oversized item across multiple child async-operations. Workers have
# no per-document serialization for retain (the busy-bank guard in
# claim_tasks only covers consolidation), so concurrent siblings sharing one
# document_id race on handle_document_tracking(is_first_batch=True), cascade-
# delete each other's memory_units, and trip FK violations on memory_links in
# the final ANN pass. The fix: keep the oversized item un-chunked in a single
# child; the worker's in-process splitter handles intra-document chunking
# sequentially with correct is_first_batch semantics.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversized_single_item_creates_one_child_not_many(memory, request_context, monkeypatch):
    """One oversized document → one child operation holding the un-chunked content."""
    from hindsight_api.config import get_config
    from hindsight_api.engine.memory_engine import count_tokens

    bank_id = f"test_oversized_one_child_{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)

    # No-op the worker dispatch: this is a structural assertion on the
    # async_operations rows that submit_async_retain inserts (those rows are
    # committed before submit_task is invoked), so we don't need the
    # SyncTaskBackend to drive the slow LLM pipeline to completion.
    async def noop_submit_task(_task_dict):
        return None

    monkeypatch.setattr(memory._task_backend, "submit_task", noop_submit_task)

    tokens_per_batch = get_config().retain_batch_tokens
    # Build content that comfortably exceeds the per-batch token budget.
    # The pre-fix code would split this into many siblings, all sharing
    # the same document_id and racing on handle_document_tracking.
    unit = "The quick brown fox jumps over the lazy dog. " * 50  # ~570 tokens
    repetitions = max(1, (tokens_per_batch * 4) // count_tokens(unit) + 1)
    big_content = unit * repetitions
    document_id = f"doc-oversize-{uuid.uuid4().hex[:8]}"

    total_tokens = count_tokens(big_content)
    assert total_tokens > tokens_per_batch, (
        f"Test setup error: content has {total_tokens} tokens, must exceed the batch budget of {tokens_per_batch}"
    )

    result = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=[{"content": big_content, "document_id": document_id}],
        request_context=request_context,
    )
    parent_operation_id = result["operation_id"]

    # Parent metadata must say one child — even though token count is huge,
    # the un-fragmenting splitter keeps the single item in one child.
    parent_row = await pool.fetchrow(
        "SELECT result_metadata FROM async_operations WHERE operation_id = $1",
        uuid.UUID(parent_operation_id),
    )
    parent_meta = (
        json.loads(parent_row["result_metadata"])
        if isinstance(parent_row["result_metadata"], str)
        else parent_row["result_metadata"]
    )
    assert parent_meta["num_sub_batches"] == 1, (
        f"Expected 1 child for an oversized single item, got {parent_meta['num_sub_batches']}. "
        "Issue #1795: per-chunk children race on the shared document_id."
    )

    # Exactly one retain child row.
    children = await pool.fetch(
        """
        SELECT operation_id, status, task_payload, result_metadata
        FROM async_operations
        WHERE bank_id = $1 AND operation_type = 'retain'
        """,
        bank_id,
    )
    assert len(children) == 1, (
        f"Expected exactly 1 child retain operation for bank_id={bank_id}, "
        f"got {len(children)}. Issue #1795 regression: oversized item was "
        f"fragmented into per-chunk children."
    )

    child = children[0]
    payload = child["task_payload"]
    payload = json.loads(payload) if isinstance(payload, str) else payload
    assert payload["type"] == "batch_retain"
    assert payload["bank_id"] == bank_id
    # Child holds the full un-chunked content — the worker's in-process
    # splitter will re-chunk it sequentially with is_first_batch=(i==1).
    assert len(payload["contents"]) == 1, (
        f"Child payload should hold the original single item, got {len(payload['contents'])} items"
    )
    assert payload["contents"][0]["document_id"] == document_id
    assert payload["contents"][0]["content"] == big_content, (
        "Child must carry the FULL un-chunked content — chunking happens inside the worker, not at submit time"
    )


@pytest.mark.asyncio
async def test_oversized_item_among_small_items_keeps_small_items_packed(memory, request_context, monkeypatch):
    """Mixed batch: small items pack together; oversized item isolates to its own child."""
    from hindsight_api.config import get_config
    from hindsight_api.engine.memory_engine import count_tokens

    bank_id = f"test_mixed_pack_{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)

    # Structural assertion on async_operations rows only — skip worker dispatch.
    async def noop_submit_task(_task_dict):
        return None

    monkeypatch.setattr(memory._task_backend, "submit_task", noop_submit_task)

    tokens_per_batch = get_config().retain_batch_tokens
    big_unit = "The quick brown fox jumps over the lazy dog. " * 50
    big_repetitions = max(1, (tokens_per_batch * 3) // count_tokens(big_unit) + 1)
    big_content = big_unit * big_repetitions

    big_doc = f"doc-big-{uuid.uuid4().hex[:8]}"
    small_docs = [f"doc-small-{i}-{uuid.uuid4().hex[:6]}" for i in range(3)]
    contents = [
        {"content": "Alice works at Google.", "document_id": small_docs[0]},
        {"content": big_content, "document_id": big_doc},
        {"content": "Bob loves Python.", "document_id": small_docs[1]},
        {"content": "Carol writes Rust.", "document_id": small_docs[2]},
    ]

    result = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
    )

    children = await pool.fetch(
        """
        SELECT task_payload
        FROM async_operations
        WHERE bank_id = $1 AND operation_type = 'retain'
        ORDER BY created_at, operation_id
        """,
        bank_id,
    )

    # Find the child holding the big doc — it must hold ONLY the big doc.
    big_child = None
    small_doc_ids_seen: list[str] = []
    for row in children:
        payload = row["task_payload"]
        payload = json.loads(payload) if isinstance(payload, str) else payload
        items = payload["contents"]
        doc_ids = [item["document_id"] for item in items]
        if big_doc in doc_ids:
            assert big_child is None, f"Big doc {big_doc} should appear in exactly one child, found in 2"
            big_child = items
            assert doc_ids == [big_doc], f"Child holding the oversized item must hold ONLY that item, got {doc_ids}"
            assert items[0]["content"] == big_content
        else:
            small_doc_ids_seen.extend(doc_ids)

    assert big_child is not None, "Big doc must appear in some child"
    # Every small input present, exactly once, across the other children.
    assert sorted(small_doc_ids_seen) == sorted(small_docs)


@pytest.mark.asyncio
async def test_submit_async_batch_retain_rolls_back_parent_on_child_failure(
    memory_no_llm_verify, request_context, monkeypatch
):
    """Regression for orphaned-parent rows.

    submit_async_batch_retain inserts a parent row (status='pending',
    task_payload=NULL — it's a status aggregator, not directly executable) and
    then loops to insert one child row per sub-batch. If the parent INSERT and
    the child INSERTs were not transactionally coupled, any failure during the
    child loop (connection drop, timeout, schema-cache invalidation under
    concurrent load) would leave a parent with zero children. Workers ignore
    such rows forever (task_payload IS NULL filter), the status aggregator
    never fires (no children to complete), and the row sits pending
    indefinitely — visible in queue-depth metrics and growing without bound.

    This test simulates a child-step failure by raising on the second
    BatchRetainChildMetadata construction. After the failure we expect zero
    async_operations rows for the bank: the parent INSERT must roll back
    together with the children.
    """
    import hindsight_api.engine.memory_engine as me
    from hindsight_api.engine.memory_engine import count_tokens

    bank_id = f"test_parent_rollback_{uuid.uuid4().hex[:8]}"
    pool = await memory_no_llm_verify._get_pool()
    await _ensure_bank(pool, bank_id)

    # Force at least 2 sub-batches so the child loop runs more than once
    # (matches the existing large-batch fixture's sizing).
    large_content = "The quick brown fox jumps over the lazy dog. " * 500
    contents = [{"content": large_content + f" item {i}", "document_id": f"doc{i}"} for i in range(2)]
    assert sum(count_tokens(item["content"]) for item in contents) > 10_000

    real_class = me.BatchRetainChildMetadata
    call_count = {"n": 0}

    def failing_child_metadata(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("Simulated child-step failure mid-batch")
        return real_class(*args, **kwargs)

    monkeypatch.setattr(me, "BatchRetainChildMetadata", failing_child_metadata)

    with pytest.raises(RuntimeError, match="Simulated child-step failure"):
        await memory_no_llm_verify.submit_async_retain(
            bank_id=bank_id,
            contents=contents,
            request_context=request_context,
        )

    rows = await pool.fetch(
        "SELECT operation_id, operation_type, status, task_payload FROM async_operations WHERE bank_id = $1",
        bank_id,
    )
    assert rows == [], (
        f"Expected zero rows for bank_id={bank_id} after rollback, got {len(rows)}: "
        f"{[(r['operation_type'], r['status'], r['task_payload'] is not None) for r in rows]}. "
        "The parent INSERT must be transactionally coupled to the child INSERTs."
    )
