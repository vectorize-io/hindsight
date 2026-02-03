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
)


def get_groq_api_key() -> str | None:
    """Get Groq API key from environment."""
    return os.getenv("GROQ_API_KEY")


@pytest.mark.asyncio
async def test_llm_metrics_recorded_for_groq():
    """
    Test that LLM metrics are recorded when making LLM calls via Groq.
    Uses openai/gpt-oss-20b as recommended by Hindsight.
    """
    api_key = get_groq_api_key()
    if not api_key:
        pytest.skip("Skipping: GROQ_API_KEY not set")

    # Create a mock metrics collector to track record_llm_call calls
    mock_collector = MagicMock(spec=MetricsCollector)

    # Patch the provider module where get_metrics_collector is actually called
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector", return_value=mock_collector):
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

        # Verify record_llm_call was called - this is the main test
        assert mock_collector.record_llm_call.called, "record_llm_call should have been called"

        # Get the call arguments
        call_kwargs = mock_collector.record_llm_call.call_args.kwargs

        # Verify the call had correct structure
        assert call_kwargs["provider"] == "groq", f"Expected provider='groq', got {call_kwargs}"
        assert call_kwargs["model"] == "openai/gpt-oss-20b", f"Expected model='openai/gpt-oss-20b', got {call_kwargs}"
        assert call_kwargs["scope"] == "test_metrics", f"Expected scope='test_metrics', got {call_kwargs}"
        assert call_kwargs["duration"] > 0, f"Expected duration > 0, got {call_kwargs['duration']}"
        assert call_kwargs["input_tokens"] > 0, f"Expected input_tokens > 0, got {call_kwargs['input_tokens']}"
        assert call_kwargs["output_tokens"] >= 0, f"Expected output_tokens >= 0, got {call_kwargs['output_tokens']}"
        assert call_kwargs["success"] is True, f"Expected success=True, got {call_kwargs['success']}"

        print(f"\nLLM metrics recorded:")
        print(f"  provider: {call_kwargs['provider']}")
        print(f"  model: {call_kwargs['model']}")
        print(f"  scope: {call_kwargs['scope']}")
        print(f"  duration: {call_kwargs['duration']:.3f}s")
        print(f"  input_tokens: {call_kwargs['input_tokens']}")
        print(f"  output_tokens: {call_kwargs['output_tokens']}")
        print(f"  response: {response}")


@pytest.mark.asyncio
async def test_llm_metrics_recorded_for_structured_output():
    """
    Test that LLM metrics are recorded for structured output (JSON) calls.
    """
    api_key = get_groq_api_key()
    if not api_key:
        pytest.skip("Skipping: GROQ_API_KEY not set")

    from pydantic import BaseModel

    class SimpleResponse(BaseModel):
        greeting: str
        language: str

    mock_collector = MagicMock(spec=MetricsCollector)

    # Patch the provider module where get_metrics_collector is actually called
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector", return_value=mock_collector):
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

        # Verify record_llm_call was called
        assert mock_collector.record_llm_call.called, "record_llm_call should have been called"

        call_kwargs = mock_collector.record_llm_call.call_args.kwargs
        assert call_kwargs["input_tokens"] > 0
        assert call_kwargs["output_tokens"] > 0

        print(f"\nStructured output LLM metrics:")
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


@pytest.mark.asyncio
async def test_return_usage_returns_tuple():
    """
    Test that return_usage=True returns (result, TokenUsage) tuple.
    """
    from hindsight_api.engine.response_models import TokenUsage

    api_key = get_groq_api_key()
    if not api_key:
        pytest.skip("Skipping: GROQ_API_KEY not set")

    llm = LLMProvider(
        provider="groq",
        api_key=api_key,
        base_url="",
        model="openai/gpt-oss-20b",
    )

    # Call with return_usage=True
    result, usage = await llm.call(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Reply with just the number."}
        ],
        max_completion_tokens=50,
        return_usage=True,
    )

    # Verify result is the response text
    assert result is not None
    assert isinstance(result, str)

    # Verify usage is TokenUsage model with valid counts
    assert isinstance(usage, TokenUsage)
    assert usage.input_tokens > 0, f"Expected input_tokens > 0, got {usage.input_tokens}"
    assert usage.output_tokens >= 0, f"Expected output_tokens >= 0, got {usage.output_tokens}"
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens

    print(f"\nreturn_usage=True test:")
    print(f"  result: {result}")
    print(f"  usage: {usage}")


@pytest.mark.asyncio
async def test_return_usage_with_structured_output():
    """
    Test that return_usage=True works with structured output (JSON).
    """
    from pydantic import BaseModel
    from hindsight_api.engine.response_models import TokenUsage

    api_key = get_groq_api_key()
    if not api_key:
        pytest.skip("Skipping: GROQ_API_KEY not set")

    class MathAnswer(BaseModel):
        answer: int
        explanation: str

    llm = LLMProvider(
        provider="groq",
        api_key=api_key,
        base_url="",
        model="openai/gpt-oss-20b",
    )

    # Call with return_usage=True and structured output
    result, usage = await llm.call(
        messages=[{"role": "user", "content": "What is 5+3? Return the answer and a brief explanation."}],
        response_format=MathAnswer,
        max_completion_tokens=100,
        return_usage=True,
    )

    # Verify result is the parsed response
    assert isinstance(result, MathAnswer)
    assert result.answer == 8
    assert result.explanation is not None

    # Verify usage is TokenUsage model
    assert isinstance(usage, TokenUsage)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0

    print(f"\nStructured output with return_usage=True:")
    print(f"  result: {result}")
    print(f"  usage: {usage}")
