"""Unit tests for mental model operation validator hooks.

Tests that the default hook implementations on OperationValidatorExtension
behave correctly (accept by default, no-op on complete).
"""

from unittest.mock import MagicMock

import pytest

from hindsight_api.extensions.operation_validator import (
    MentalModelGetContext,
    MentalModelGetResult,
    MentalModelRefreshResult,
    OperationValidatorExtension,
    ValidationResult,
)


@pytest.fixture
def validator():
    """Create a concrete subclass for testing default behavior."""

    class TestValidator(OperationValidatorExtension):
        async def validate_retain(self, ctx):
            return ValidationResult.accept()

        async def validate_recall(self, ctx):
            return ValidationResult.accept()

        async def validate_reflect(self, ctx):
            return ValidationResult.accept()

    return TestValidator(config={})


@pytest.mark.asyncio
async def test_default_hooks_accept_and_noop(validator):
    """Default mental model hooks accept validation and don't raise on complete."""
    ctx = MentalModelGetContext(
        bank_id="bank-1",
        mental_model_id="mm-1",
        request_context=MagicMock(),
    )
    result = await validator.validate_mental_model_get(ctx)
    assert result.allowed is True

    # on_complete hooks should not raise
    get_result = MentalModelGetResult(
        bank_id="bank-1",
        mental_model_id="mm-1",
        request_context=MagicMock(),
        output_tokens=100,
    )
    await validator.on_mental_model_get_complete(get_result)

    refresh_result = MentalModelRefreshResult(
        bank_id="bank-1",
        mental_model_id="mm-1",
        request_context=MagicMock(),
        query_tokens=50,
        output_tokens=500,
        context_tokens=0,
        facts_used=5,
        mental_models_used=1,
    )
    await validator.on_mental_model_refresh_complete(refresh_result)
