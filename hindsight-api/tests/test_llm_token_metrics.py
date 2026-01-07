"""
Test that LLM calls record token metrics via the metrics collector.
"""
import os
from unittest.mock import MagicMock, patch
import pytest
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.metrics import (
    MetricsCollector,
    NoOpMetricsCollector,
    get_metrics_collector,
    initialize_metrics,
    create_metrics_collector,
)


def get_groq_api_key() -> str | None:
    """Get Groq API key from environment."""
    return os.getenv("GROQ_API_KEY")


@pytest.mark.asyncio
async def test_token_metrics_recorded_for_groq():
    """
    Test that token metrics are recorded when making LLM calls via Groq.
    Uses openai/gpt-oss-20b as recommended by Hindsight.
    """
    api_key = get_groq_api_key()
    if not api_key:
        pytest.skip("Skipping: GROQ_API_KEY not set")

    # Create a mock metrics collector to track record_tokens calls
    mock_collector = MagicMock(spec=MetricsCollector)

    with patch("hindsight_api.engine.llm_wrapper.get_metrics_collector", return_value=mock_collector):
        llm = LLMProvider(
            provider="groq",
            api_key=api_key,
            base_url="",
            model="openai/gpt-oss-20b",
        )

        # Make an LLM call with clear instruction
        response = await llm.call(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond."},
                {"role": "user", "content": "What is 2+2? Reply with just the number."}
            ],
            max_completion_tokens=50,
            scope="test_metrics",
        )

        # Verify record_tokens was called - this is the main test
        assert mock_collector.record_tokens.called, "record_tokens should have been called"

        # Get the call arguments
        call_kwargs = mock_collector.record_tokens.call_args.kwargs

        # Verify the call had correct structure
        assert call_kwargs["operation"] == "test_metrics", f"Expected operation='test_metrics', got {call_kwargs}"
        assert call_kwargs["bank_id"] == "llm", f"Expected bank_id='llm', got {call_kwargs}"
        assert call_kwargs["input_tokens"] > 0, f"Expected input_tokens > 0, got {call_kwargs['input_tokens']}"
        # Output tokens may be 0 for some edge cases, but input should always be > 0
        assert call_kwargs["output_tokens"] >= 0, f"Expected output_tokens >= 0, got {call_kwargs['output_tokens']}"

        print(f"\nToken metrics recorded:")
        print(f"  operation: {call_kwargs['operation']}")
        print(f"  input_tokens: {call_kwargs['input_tokens']}")
        print(f"  output_tokens: {call_kwargs['output_tokens']}")
        print(f"  response: {response}")


@pytest.mark.asyncio
async def test_token_metrics_recorded_for_structured_output():
    """
    Test that token metrics are recorded for structured output (JSON) calls.
    """
    api_key = get_groq_api_key()
    if not api_key:
        pytest.skip("Skipping: GROQ_API_KEY not set")

    from pydantic import BaseModel

    class SimpleResponse(BaseModel):
        greeting: str
        language: str

    mock_collector = MagicMock(spec=MetricsCollector)

    with patch("hindsight_api.engine.llm_wrapper.get_metrics_collector", return_value=mock_collector):
        llm = LLMProvider(
            provider="groq",
            api_key=api_key,
            base_url="",
            model="openai/gpt-oss-20b",
        )

        # Make a structured output call
        response = await llm.call(
            messages=[{"role": "user", "content": "Say hello in French. Return greeting and language."}],
            response_format=SimpleResponse,
            max_completion_tokens=100,
            scope="structured_output_test",
        )

        # Verify structured response
        assert isinstance(response, SimpleResponse)
        assert response.greeting is not None
        assert response.language is not None

        # Verify record_tokens was called
        assert mock_collector.record_tokens.called, "record_tokens should have been called"

        call_kwargs = mock_collector.record_tokens.call_args.kwargs
        assert call_kwargs["input_tokens"] > 0
        assert call_kwargs["output_tokens"] > 0

        print(f"\nStructured output token metrics:")
        print(f"  greeting: {response.greeting}")
        print(f"  language: {response.language}")
        print(f"  input_tokens: {call_kwargs['input_tokens']}")
        print(f"  output_tokens: {call_kwargs['output_tokens']}")


@pytest.mark.asyncio
async def test_noop_collector_when_metrics_disabled():
    """
    Test that NoOpMetricsCollector is returned when metrics are not initialized.
    This verifies the fallback behavior doesn't break LLM calls.
    """
    api_key = get_groq_api_key()
    if not api_key:
        pytest.skip("Skipping: GROQ_API_KEY not set")

    # Without initializing metrics, get_metrics_collector returns NoOpMetricsCollector
    collector = get_metrics_collector()
    assert isinstance(collector, NoOpMetricsCollector), "Should return NoOpMetricsCollector when not initialized"

    # Make an LLM call - should work fine with NoOp collector
    llm = LLMProvider(
        provider="groq",
        api_key=api_key,
        base_url="",
        model="openai/gpt-oss-20b",
    )

    response = await llm.call(
        messages=[{"role": "user", "content": "Say 'test' in one word."}],
        max_completion_tokens=50,
    )

    assert response is not None
    print(f"\nLLM call succeeded with NoOpMetricsCollector: {response}")
