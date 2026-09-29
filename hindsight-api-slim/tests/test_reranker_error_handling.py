"""
Regression test for UnboundLocalError in recall when the reranker raises.

Before the fix, `scored_results` and `pre_filtered_count` were only assigned
inside the `try` block, but referenced in the `finally` block.  If
`reranker_instance.rerank()` (or `ensure_initialized()`) raised, the `finally`
block crashed with `UnboundLocalError` instead of propagating the original
exception.

Fix: initialise both variables to safe defaults before the try/finally block.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

# These patch the ENGINE's cross-encoder and expect recall to surface its failure. A store that
# answers a whole recall reranks inside itself and never calls that collaborator, so the error
# these pin cannot arise -- the store's own rerank failure path is covered by its suite.
pytestmark = pytest.mark.memory_backend_incompatible


@pytest.mark.asyncio
async def test_recall_reranker_error_does_not_raise_unbound_local(memory, request_context):
    """Recall must propagate the reranker's exception, not an UnboundLocalError."""
    bank_id = f"test_reranker_err_{datetime.now(timezone.utc).timestamp()}"

    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Paris is the capital of France",
            request_context=request_context,
        )

        # Simulate a reranker failure (e.g. Cohere API error on empty/small candidate set)
        rerank_mock = AsyncMock(side_effect=RuntimeError("reranker API error"))
        memory._cross_encoder_reranker._initialized = True  # skip ensure_initialized

        with patch.object(memory._cross_encoder_reranker, "rerank", rerank_mock):
            with pytest.raises(Exception, match="reranker API error"):
                await memory.recall_async(
                    bank_id=bank_id,
                    query="capital of France",
                    request_context=request_context,
                )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_recall_reranker_init_error_does_not_raise_unbound_local(memory, request_context):
    """Same regression when ensure_initialized() raises (before pre_filtered_count is set)."""
    bank_id = f"test_reranker_init_err_{datetime.now(timezone.utc).timestamp()}"

    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Paris is the capital of France",
            request_context=request_context,
        )

        init_mock = AsyncMock(side_effect=RuntimeError("reranker init failed"))
        memory._cross_encoder_reranker._initialized = False

        with patch.object(memory._cross_encoder_reranker, "ensure_initialized", init_mock):
            with pytest.raises(Exception, match="reranker init failed"):
                await memory.recall_async(
                    bank_id=bank_id,
                    query="capital of France",
                    request_context=request_context,
                )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
