"""
Test Anthropic provider implementation.
"""

import os

import pytest

from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM


@pytest.mark.asyncio
async def test_anthropic_provider_basic_call():
    """Test basic call functionality with Anthropic provider."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicLLM(
        provider="anthropic",
        api_key=api_key,
        base_url="",
        model="claude-sonnet-4-20250514",
    )

    # Test basic text completion
    messages = [{"role": "user", "content": "Say 'test passed' and nothing else"}]
    result = await provider.call(messages=messages, max_completion_tokens=50, temperature=0.0, scope="test")

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"Anthropic response: {result}")

    await provider.cleanup()


@pytest.mark.asyncio
async def test_anthropic_provider_structured_output():
    """Test structured output with Anthropic provider."""
    from pydantic import BaseModel, Field

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicLLM(
        provider="anthropic",
        api_key=api_key,
        base_url="",
        model="claude-sonnet-4-20250514",
    )

    class TestResponse(BaseModel):
        """Test response model."""

        answer: str = Field(description="The answer")
        confidence: float = Field(description="Confidence score 0-1")

    messages = [{"role": "user", "content": "What is 2+2? Respond with answer and confidence."}]
    result = await provider.call(
        messages=messages,
        response_format=TestResponse,
        max_completion_tokens=100,
        temperature=0.0,
        scope="test",
    )

    assert result is not None
    assert isinstance(result, TestResponse)
    assert result.answer is not None
    assert 0 <= result.confidence <= 1
    print(f"Anthropic structured response: {result}")

    await provider.cleanup()


@pytest.mark.asyncio
async def test_anthropic_provider_tool_calling():
    """Test tool calling with Anthropic provider."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicLLM(
        provider="anthropic",
        api_key=api_key,
        base_url="",
        model="claude-sonnet-4-20250514",
    )

    # Define test tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "What's the weather in Paris? Use celsius."}]

    result = await provider.call_with_tools(
        messages=messages,
        tools=tools,
        max_completion_tokens=100,
        temperature=0.0,
        scope="test",
    )

    assert result is not None
    assert result.tool_calls is not None
    assert len(result.tool_calls) > 0

    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_weather"
    assert "location" in tool_call.arguments
    assert tool_call.arguments["location"].lower() == "paris"
    print(f"Anthropic tool call: {tool_call}")

    await provider.cleanup()


@pytest.mark.asyncio
async def test_anthropic_provider_token_usage():
    """Test that token usage is properly tracked."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicLLM(
        provider="anthropic",
        api_key=api_key,
        base_url="",
        model="claude-sonnet-4-20250514",
    )

    messages = [{"role": "user", "content": "Say hello"}]
    result, usage = await provider.call(
        messages=messages,
        max_completion_tokens=50,
        temperature=0.0,
        scope="test",
        return_usage=True,
    )

    assert result is not None
    assert usage is not None
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    print(f"Token usage: {usage}")

    await provider.cleanup()
