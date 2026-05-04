"""Test automatic batch chunking based on token count.

Verifies that retain_batch_async splits large batches into sub-batches
and processes them sequentially. Uses mocking to verify the split happens
without making real LLM calls.
"""

from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.llm_wrapper import TokenUsage


@pytest.mark.asyncio
async def test_large_batch_splits_into_sub_batches(memory: MemoryEngine, request_context):
    """A batch exceeding the token threshold should be split into sub-batches."""
    bank_id = "test_chunking_agent"

    # Create a batch large enough to trigger chunking.
    # Default retain_batch_tokens is 10000. Each item is ~326 tokens,
    # so 40 items = ~13000 tokens, which exceeds the threshold.
    large_content = "Alice met with Bob at the coffee shop to discuss the project. " * 25
    contents = [
        {"content": large_content, "context": f"conversation_{i}"}
        for i in range(40)
    ]

    with patch.object(
        memory, "_retain_batch_async_internal", new_callable=AsyncMock
    ) as mock_internal:
        mock_internal.return_value = (["unit_id"], TokenUsage(), 0)

        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=contents,
            request_context=request_context,
        )

        # Should have been called multiple times (once per sub-batch)
        assert mock_internal.call_count > 1, (
            f"Expected multiple sub-batch calls, got {mock_internal.call_count}. "
            "Batch should have been split."
        )

        # Total items across all calls should equal input
        total_items = sum(len(call.kwargs["contents"]) for call in mock_internal.call_args_list)
        assert total_items == 40


@pytest.mark.asyncio
async def test_small_batch_not_split(memory: MemoryEngine, request_context):
    """A batch under the token threshold should NOT be split."""
    bank_id = "test_no_chunking_agent"

    contents = [
        {"content": "Alice works at Google", "context": "conversation_1"},
        {"content": "Bob loves Python", "context": "conversation_2"},
    ]

    with patch.object(
        memory, "_retain_batch_async_internal", new_callable=AsyncMock
    ) as mock_internal:
        mock_internal.return_value = (["unit_id"], TokenUsage(), 0)

        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=contents,
            request_context=request_context,
        )

        # Should be called exactly once (no splitting)
        assert mock_internal.call_count == 1
        assert len(mock_internal.call_args.kwargs["contents"]) == 2
