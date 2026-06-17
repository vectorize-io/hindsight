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
from datetime import datetime

import pytest

from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.utils import extract_facts
from hindsight_api.engine.search.think_utils import reflect

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
    return LLMProvider(
        provider=_PROVIDER,
        api_key=_get_api_key(),
        base_url=os.environ.get("HINDSIGHT_API_LLM_BASE_URL", ""),
        model=_MODEL,
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
    response = await llm.call(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer in one word."},
        ],
        max_completion_tokens=50,
    )
    assert response is not None, "call() returned None"
    assert len(response) > 0, "call() returned empty string"

    # Test 3: call() with response_format (structured output)
    from pydantic import BaseModel

    class TestResponse(BaseModel):
        answer: str
        confidence: str

    structured = await llm.call(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        response_format=TestResponse,
        max_completion_tokens=100,
    )
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


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_llm_memory_operations():
    """
    Test fact extraction and reflect with the configured LLM provider.
    """
    llm = _make_llm()

    # Fact extraction (structured output)
    test_text = """
    User: I just got back from my trip to Paris last week. The Eiffel Tower was amazing!
    Assistant: That sounds wonderful! How long were you there?
    User: About 5 days. I also visited the Louvre and saw the Mona Lisa.
    """

    facts, chunks = await extract_facts(
        text=test_text,
        event_date=datetime(2024, 12, 10),
        context="Travel conversation",
        llm_config=llm,
    )

    assert facts is not None, "fact extraction returned None"
    assert len(facts) > 0, "should extract at least one fact"

    for fact in facts:
        assert fact.fact, "fact missing text"
        assert fact.fact_type in ["world", "experience"], f"invalid fact_type: {fact.fact_type}"

    # Reflect
    response = await reflect(
        llm_config=llm,
        query="What was the highlight of my Paris trip?",
        experience_facts=[
            "I visited Paris in December 2024",
            "I saw the Eiffel Tower and it was amazing",
            "I visited the Louvre and saw the Mona Lisa",
            "The trip lasted 5 days",
        ],
        world_facts=[
            "The Eiffel Tower is a famous landmark in Paris",
            "The Mona Lisa is displayed at the Louvre museum",
        ],
        name="Traveler",
    )

    assert response is not None, "reflect returned None"
    assert len(response) > 10, "reflect response too short"
