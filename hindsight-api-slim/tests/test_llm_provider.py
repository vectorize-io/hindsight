"""
LLM Minimum Acceptance Tests — provider API surface.

Validates that a given LLM provider/model works correctly with Hindsight's
low-level LLM API methods: plain text, structured output, and tool calling.

The provider/model under test comes from HINDSIGHT_API_LLM_PROVIDER /
HINDSIGHT_API_LLM_MODEL env vars, which are set by the CI matrix in the
test-api-llm-acceptance job.

These tests are excluded from the regular test-api CI job via the
hs_llm_mat marker.
"""

import os

import pytest

from hindsight_api.engine.llm_wrapper import LLMProvider

pytestmark = pytest.mark.hs_llm_mat

_PROVIDER = os.environ.get("HINDSIGHT_API_LLM_PROVIDER", "")
_MODEL = os.environ.get("HINDSIGHT_API_LLM_MODEL", "")


def _get_api_key() -> str:
    """Get API key from HINDSIGHT_API_LLM_API_KEY (CI) or provider-specific env var."""
    key = os.environ.get("HINDSIGHT_API_LLM_API_KEY", "")
    if key:
        return key
    # Fallback to provider-specific env vars for local dev
    provider_key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    env_var = provider_key_map.get(_PROVIDER, "")
    return os.environ.get(env_var, "") if env_var else ""


def _make_llm() -> LLMProvider:
    # LLMProvider uses provider-specific settings as-passed (it does not resolve
    # them from global config), so forward the ones whose providers require them:
    # Vertex AI needs project/region, and litellmrouter needs its router config.
    # Without these, provider=vertexai/litellmrouter raise at construction.
    from hindsight_api.config import get_config

    config = get_config()
    return LLMProvider(
        provider=_PROVIDER,
        api_key=_get_api_key(),
        base_url=os.environ.get("HINDSIGHT_API_LLM_BASE_URL", ""),
        model=_MODEL,
        vertexai_project_id=config.llm_vertexai_project_id,
        vertexai_region=config.llm_vertexai_region,
        vertexai_service_account_key=config.llm_vertexai_service_account_key,
        litellmrouter_config=config.llm_litellmrouter_config,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
# Tool-calling output is sampled, so some providers occasionally return zero
# tool calls even when the prompt clearly asks for one.  Retry to ride out
# the sampling miss; a persistent break still surfaces after 3 attempts.
@pytest.mark.flaky(reruns=2, reruns_delay=2)
async def test_llm_api_methods():
    """
    Test all LLM API methods used by Hindsight at runtime.

    Tests:
    1. verify_connection() - Connection verification
    2. call() with plain text - Basic LLM call
    3. call() with response_format - Structured output (used in fact extraction)
    4. call_with_tools() - Tool calling (used in reflect agent)
    """
    llm = _make_llm()

    # Test 1: verify_connection()
    await llm.verify_connection()

    # Test 2: call() with plain text
    response = (
        await llm.call(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2? Answer in one word."},
            ],
            max_completion_tokens=50,
        )
    ).content
    assert response is not None, "call() returned None"
    assert len(response) > 0, "call() returned empty string"

    # Test 3: call() with response_format (structured output)
    from pydantic import BaseModel

    class TestResponse(BaseModel):
        answer: str
        confidence: str

    structured = (
        await llm.call(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            response_format=TestResponse,
            max_completion_tokens=100,
        )
    ).content
    assert isinstance(structured, TestResponse), f"Expected TestResponse, got {type(structured)}"
    assert structured.answer, "Structured output missing 'answer'"
    assert structured.confidence, "Structured output missing 'confidence'"

    # Test 4: call_with_tools() (tool calling)
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
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
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

    assert result is not None, "call_with_tools() returned None"
    assert hasattr(result, "tool_calls"), "Result missing 'tool_calls' attribute"
    assert len(result.tool_calls) > 0, f"Expected at least 1 tool call, got {len(result.tool_calls)}"

    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_weather", f"Expected 'get_weather', got '{tool_call.name}'"
    assert "location" in tool_call.arguments, "Tool call arguments missing 'location'"
