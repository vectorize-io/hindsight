"""
Test LLM provider with different models using actual memory operations.
"""
import os
from datetime import datetime
import pytest
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.utils import extract_facts
from hindsight_api.engine.search.think_utils import reflect


# Model matrix: (provider, model)
MODEL_MATRIX = [
    # OpenAI models
    ("openai", "gpt-4o-mini"),
    ("openai", "gpt-4.1-mini"),
    ("openai", "gpt-4.1-nano"),
    ("openai", "gpt-5-mini"),
    ("openai", "gpt-5-nano"),
    ("openai", "gpt-5"),
    ("openai", "gpt-5.2"),
    # Anthropic models
    ("anthropic", "claude-sonnet-4-20250514"),
    ("anthropic", "claude-opus-4-5-20251101"),
    ("anthropic", "claude-haiku-4-20250514"),
    # Groq models
    ("groq", "openai/gpt-oss-120b"),
    ("groq", "openai/gpt-oss-20b"),
    # Gemini models
    ("gemini", "gemini-2.5-flash"),
    ("gemini", "gemini-2.5-flash-lite"),
    ("gemini", "gemini-3-pro-preview"),
    # Ollama models (local)
    ("ollama", "gemma3:12b"),
    ("ollama", "gemma3:1b"),
    # Claude Code (uses Claude Agent SDK with Claude models)
    ("claude-code", "claude-sonnet-4-20250514"),
    # OpenAI Codex (uses MCP with Codex-specific models)
    ("openai-codex", "gpt-5.2-codex"),
    # Mock provider (for testing)
    ("mock", "mock"),
]


def get_api_key_for_provider(provider: str) -> str | None:
    """Get API key for provider from environment variables."""
    provider_key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = provider_key_map.get(provider)
    return os.getenv(env_var) if env_var else None


def should_skip_provider(provider: str, model: str = "") -> tuple[bool, str]:
    """Check if provider should be skipped and return reason."""
    # Never skip mock provider
    if provider == "mock":
        return False, ""

    # Skip Ollama in CI (no models available)
    if provider == "ollama" and os.getenv("CI"):
        return True, "Ollama not available in CI"

    # Skip Ollama gemma models (don't support tool calling)
    if provider == "ollama" and "gemma" in model.lower():
        return True, f"Ollama {model} does not support tool calling"

    # Don't skip claude-code or openai-codex - let them run if configured
    # They will fail with clear errors if not set up properly

    # Other providers need an API key
    if provider not in ("ollama", "claude-code", "openai-codex", "mock"):
        api_key = get_api_key_for_provider(provider)
        if not api_key:
            return True, f"No API key available (set {provider.upper()}_API_KEY)"

    return False, ""


@pytest.mark.parametrize("provider,model", MODEL_MATRIX)
@pytest.mark.asyncio
async def test_llm_provider_memory_operations(provider: str, model: str):
    """
    Test LLM provider with actual memory operations: fact extraction and reflect.
    All models must pass this test.
    """
    should_skip, reason = should_skip_provider(provider, model)
    if should_skip:
        pytest.skip(f"Skipping {provider}/{model}: {reason}")

    api_key = get_api_key_for_provider(provider)

    llm = LLMProvider(
        provider=provider,
        api_key=api_key or "",
        base_url="",
        model=model,
    )

    # Test 1: Fact extraction (structured output)
    test_text = """
    User: I just got back from my trip to Paris last week. The Eiffel Tower was amazing!
    Assistant: That sounds wonderful! How long were you there?
    User: About 5 days. I also visited the Louvre and saw the Mona Lisa.
    """
    event_date = datetime(2024, 12, 10)

    facts, chunks = await extract_facts(
        text=test_text,
        event_date=event_date,
        context="Travel conversation",
        llm_config=llm,
    )

    print(f"\n{provider}/{model} - Fact extraction:")
    print(f"  Extracted {len(facts)} facts from {len(chunks)} chunks")
    for fact in facts:
        print(f"  - {fact.fact}")

    assert facts is not None, f"{provider}/{model} fact extraction returned None"
    assert len(facts) > 0, f"{provider}/{model} should extract at least one fact"

    # Verify facts have required fields
    for fact in facts:
        assert fact.fact, f"{provider}/{model} fact missing text"
        assert fact.fact_type in ["world", "experience", "opinion"], f"{provider}/{model} invalid fact_type: {fact.fact_type}"

    # Test 2: Reflect (actual reflect function)
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

    print(f"\n{provider}/{model} - Reflect response:")
    print(f"  {response[:200]}...")

    assert response is not None, f"{provider}/{model} reflect returned None"
    assert len(response) > 10, f"{provider}/{model} reflect response too short"


@pytest.mark.parametrize("provider,model", MODEL_MATRIX)
@pytest.mark.asyncio
async def test_llm_provider_tool_calling(provider: str, model: str):
    """
    Test LLM provider tool calling capability.
    This verifies that the provider can properly call tools (critical for reflect agent).
    """
    should_skip, reason = should_skip_provider(provider, model)
    if should_skip:
        pytest.skip(f"Skipping {provider}/{model}: {reason}")

    api_key = get_api_key_for_provider(provider)

    llm = LLMProvider(
        provider=provider,
        api_key=api_key or "",
        base_url="",
        model=model,
    )

    # For mock provider, set it to return tool calls
    if provider == "mock":
        llm.set_mock_response([
            {"name": "get_memory", "arguments": {"topic": "Paris trip"}},
        ])

    # Define simple test tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_memory",
                "description": "Retrieve a memory about the user's trip",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "What to retrieve (e.g., 'Paris trip')"},
                    },
                    "required": ["topic"],
                },
            },
        },
    ]

    # Call with tools - should return tool calls
    result = await llm.call_with_tools(
        messages=[
            {"role": "system", "content": "You are a helpful assistant with access to the user's memories."},
            {"role": "user", "content": "What do you remember about my Paris trip? Use the get_memory tool."},
        ],
        tools=tools,
        max_completion_tokens=500,
    )

    print(f"\n{provider}/{model} - Tool calling test:")
    print(f"  Tool calls: {len(result.tool_calls)}")
    print(f"  Content: {result.content[:200] if result.content else 'None'}...")
    if result.tool_calls:
        for tc in result.tool_calls:
            print(f"    - {tc.name}({tc.arguments})")

    # Verify tool calling worked
    assert result.tool_calls is not None, f"{provider}/{model} returned None for tool_calls"
    assert len(result.tool_calls) > 0, (
        f"{provider}/{model} did not call any tools - provider may not support tool calling properly. "
        f"Content: {result.content[:200] if result.content else 'None'}"
    )

    # Verify at least one tool call is the expected tool
    tool_names = [tc.name for tc in result.tool_calls]
    assert "get_memory" in tool_names, (
        f"{provider}/{model} called wrong tools: {tool_names}. "
        f"Expected 'get_memory'. This indicates tool calling is not working correctly."
    )

    # Verify content doesn't have meta-commentary (the bug we're fixing with Claude Code)
    # Some providers return both tool calls AND explanatory text
    if result.content:
        content_lower = result.content.lower()
        meta_phrases = ["i'll search", "let me search", "i'll look", "let me look", "i'm going to", "i will"]
        has_meta = any(phrase in content_lower for phrase in meta_phrases)
        if has_meta:
            print(f"  WARNING: {provider}/{model} content has meta-commentary: {result.content[:100]}")
            # Don't fail for this - some providers do return explanatory text alongside tool calls
