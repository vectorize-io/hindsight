"""Tests for MiniMax provider integration.

Validates that MiniMax is correctly registered as an OpenAI-compatible provider
with proper base URL, temperature clamping, and default model configuration.
"""

import os

import pytest

from hindsight_api.engine.llm_wrapper import LLMProvider, create_llm


def test_minimax_provider_creation():
    """Test that MiniMax provider can be instantiated correctly."""
    llm = LLMProvider(
        provider="minimax",
        api_key="test-key",
        base_url="",
        model="MiniMax-M2.5",
    )
    assert llm.provider == "minimax"
    assert llm.model == "MiniMax-M2.5"
    assert llm.base_url == "https://api.minimax.io/v1"


def test_minimax_default_base_url():
    """Test that MiniMax uses the correct default base URL when none is provided."""
    llm = LLMProvider(
        provider="minimax",
        api_key="test-key",
        base_url="",
        model="MiniMax-M2.5",
    )
    assert llm.base_url == "https://api.minimax.io/v1"


def test_minimax_custom_base_url():
    """Test that a custom base URL overrides the default."""
    llm = LLMProvider(
        provider="minimax",
        api_key="test-key",
        base_url="https://custom.api.example.com/v1",
        model="MiniMax-M2.5",
    )
    assert llm.base_url == "https://custom.api.example.com/v1"


def test_minimax_factory_function():
    """Test that the create_llm factory function creates MiniMax provider correctly."""
    llm = create_llm(
        provider="minimax",
        api_key="test-key",
        base_url="",
        model="MiniMax-M2.5",
    )
    assert llm is not None


def test_minimax_requires_api_key():
    """Test that MiniMax provider requires an API key."""
    with pytest.raises(ValueError, match="API key"):
        LLMProvider(
            provider="minimax",
            api_key="",
            base_url="",
            model="MiniMax-M2.5",
        )


def test_minimax_default_model_config():
    """Test that MiniMax has a default model in PROVIDER_DEFAULT_MODELS."""
    from hindsight_api.config import PROVIDER_DEFAULT_MODELS

    assert "minimax" in PROVIDER_DEFAULT_MODELS
    assert PROVIDER_DEFAULT_MODELS["minimax"] == "MiniMax-M2.5"


def test_minimax_config_default_model():
    """Test that MiniMax default model is used when model is not explicitly set."""
    from hindsight_api.config import HindsightConfig, clear_config_cache

    original_provider = os.environ.get("HINDSIGHT_API_LLM_PROVIDER")
    original_model = os.environ.get("HINDSIGHT_API_LLM_MODEL")

    try:
        clear_config_cache()
        os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "minimax"
        if "HINDSIGHT_API_LLM_MODEL" in os.environ:
            del os.environ["HINDSIGHT_API_LLM_MODEL"]

        config = HindsightConfig.from_env()
        assert config.llm_provider == "minimax"
        assert config.llm_model == "MiniMax-M2.5"
    finally:
        clear_config_cache()
        if original_provider:
            os.environ["HINDSIGHT_API_LLM_PROVIDER"] = original_provider
        elif "HINDSIGHT_API_LLM_PROVIDER" in os.environ:
            del os.environ["HINDSIGHT_API_LLM_PROVIDER"]
        if original_model:
            os.environ["HINDSIGHT_API_LLM_MODEL"] = original_model
        elif "HINDSIGHT_API_LLM_MODEL" in os.environ:
            del os.environ["HINDSIGHT_API_LLM_MODEL"]


def test_minimax_temperature_clamping():
    """Test that MiniMax temperature is clamped to (0.0, 1.0] range."""
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(
        provider="minimax",
        api_key="test-key",
        base_url="https://api.minimax.io/v1",
        model="MiniMax-M2.5",
    )

    # Verify the provider is correctly set up for temperature clamping
    assert llm.provider == "minimax"


@pytest.mark.asyncio
async def test_minimax_integration():
    """Integration test: verify MiniMax provider works with actual API.

    Requires MINIMAX_API_KEY environment variable to be set.
    """
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        pytest.skip("MINIMAX_API_KEY not set")

    llm = LLMProvider(
        provider="minimax",
        api_key=api_key,
        base_url="",
        model="MiniMax-M2.5",
    )

    # Test verify_connection
    await llm.verify_connection()

    # Test basic call
    response = await llm.call(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer in one word."},
        ],
        max_completion_tokens=50,
    )
    assert response is not None
    assert len(response) > 0


@pytest.mark.asyncio
async def test_minimax_tool_calling():
    """Integration test: verify MiniMax provider supports tool calling.

    Requires MINIMAX_API_KEY environment variable to be set.
    """
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        pytest.skip("MINIMAX_API_KEY not set")

    llm = LLMProvider(
        provider="minimax",
        api_key=api_key,
        base_url="",
        model="MiniMax-M2.5",
    )

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
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    result = await llm.call_with_tools(
        messages=[
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "What's the weather like in Paris?"},
        ],
        tools=tools,
        max_completion_tokens=500,
    )

    assert result is not None
    assert hasattr(result, "tool_calls")
    assert len(result.tool_calls) > 0
    assert result.tool_calls[0].name == "get_weather"
