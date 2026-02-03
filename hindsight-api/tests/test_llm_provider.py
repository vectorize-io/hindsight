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


@pytest.mark.parametrize("provider,model", MODEL_MATRIX)
@pytest.mark.asyncio
async def test_llm_provider_memory_operations(provider: str, model: str):
    """
    Test LLM provider with actual memory operations: fact extraction and reflect.
    All models must pass this test.
    """
    api_key = get_api_key_for_provider(provider)

    # Skip Ollama tests in CI (no models available)
    if provider == "ollama" and os.getenv("CI"):
        pytest.skip(f"Skipping {provider}/{model}: Ollama not available in CI")

    # Other providers need an API key
    if provider != "ollama" and not api_key:
        pytest.skip(f"Skipping {provider}/{model}: no API key available")

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
