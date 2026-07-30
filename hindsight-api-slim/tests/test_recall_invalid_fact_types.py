"""Regression tests for invalid recall fact types."""

import pytest

from hindsight_api.extensions.operation_validator import OperationValidationError


@pytest.mark.asyncio
async def test_recall_invalid_fact_type_raises_422(memory, request_context):
    with pytest.raises(OperationValidationError) as exc_info:
        await memory.recall_async(
            bank_id="invalid-fact-type-test",
            query="test",
            fact_type=["bogus"],
            request_context=request_context,
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.reason == ("Invalid fact type(s): bogus. Must be one of: experience, observation, world")
