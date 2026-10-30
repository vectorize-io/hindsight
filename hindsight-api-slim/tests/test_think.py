"""
Test reflect (think) function.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.response_models import Budget


@pytest.mark.asyncio
async def test_think_without_prior_context(memory, request_context):
    """
    Test that think function handles queries when there's no relevant context.
    """
    bank_id = f"test_think_no_context_{datetime.now(timezone.utc).timestamp()}"

    # Call think without storing any prior facts
    result = await memory.reflect_async(
        bank_id=bank_id,
        query="What is the capital of France?",
        budget=Budget.LOW,
        request_context=request_context,
    )

    print(f"\n=== Think Without Context ===")
    print(f"Answer: {result.text}")

    # Should still return an answer (even if it says it doesn't have enough info)
    assert result.text, "Should return some answer"
    assert result.based_on, "Should return based_on structure"
