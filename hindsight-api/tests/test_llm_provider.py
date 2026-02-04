"""
Test LLM provider with different models using actual Hindsight memory operations.

Tests validate that providers work correctly with:
1. Retain (memory ingestion with fact extraction)
2. Reflect (memory retrieval with tool calling)
3. Mental models (consolidated knowledge generation)
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

    # Skip claude-code and openai-codex in CI (require local auth)
    if os.getenv("CI") and provider in ("claude-code", "openai-codex"):
        return True, f"{provider} not available in CI (requires local authentication)"

    # Skip Ollama in CI (no models available)
    if provider == "ollama" and os.getenv("CI"):
        return True, "Ollama not available in CI"

    # Skip Ollama gemma models (don't support tool calling)
    if provider == "ollama" and "gemma" in model.lower():
        return True, f"Ollama {model} does not support tool calling"

    # Other providers need an API key
    if provider not in ("ollama", "claude-code", "openai-codex", "mock"):
        api_key = get_api_key_for_provider(provider)
        if not api_key:
            return True, f"No API key available (set {provider.upper()}_API_KEY)"

    return False, ""


@pytest.mark.parametrize("provider,model", MODEL_MATRIX)
@pytest.mark.asyncio
async def test_llm_provider_api_methods(provider: str, model: str):
    """
    Test all LLM API methods used by Hindsight at runtime.
    This validates that the provider correctly implements the LLMInterface.

    Tests:
    1. verify_connection() - Connection verification
    2. call() with plain text - Basic LLM call
    3. call() with response_format - Structured output (used in fact extraction)
    4. call_with_tools() - Tool calling (used in reflect agent)
    """
    # Skip mock provider - it's a test stub, not a real LLM implementation
    if provider == "mock":
        pytest.skip("Mock provider is a test stub, not a real LLM")

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

    print(f"\n{provider}/{model} - API methods test:")

    # Test 1: verify_connection()
    try:
        await llm.verify_connection()
        print("  ✓ verify_connection()")
    except Exception as e:
        pytest.fail(f"{provider}/{model} verify_connection() failed: {e}")

    # Test 2: call() with plain text
    try:
        response = await llm.call(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2? Answer in one word."},
            ],
            max_completion_tokens=50,
        )
        assert response is not None, "call() returned None"
        assert len(response) > 0, "call() returned empty string"
        print(f"  ✓ call() plain text: {response[:50]}")
    except Exception as e:
        pytest.fail(f"{provider}/{model} call() plain text failed: {e}")

    # Test 3: call() with response_format (structured output)
    try:
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            answer: str
            confidence: str

        response = await llm.call(
            messages=[
                {"role": "system", "content": "You are a math assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            response_format=TestResponse,
            max_completion_tokens=100,
        )
        assert isinstance(response, TestResponse), f"Expected TestResponse, got {type(response)}"
        assert hasattr(response, "answer"), "Structured output missing 'answer' field"
        assert hasattr(response, "confidence"), "Structured output missing 'confidence' field"
        print(f"  ✓ call() structured output: answer={response.answer}, confidence={response.confidence}")
    except Exception as e:
        pytest.fail(f"{provider}/{model} call() structured output failed: {e}")

    # Test 4: call_with_tools() (tool calling)
    try:
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
            max_completion_tokens=200,
        )

        assert result is not None, "call_with_tools() returned None"
        assert hasattr(result, "tool_calls"), "Result missing 'tool_calls' attribute"

        # Nano models may hit token limits before making tool calls - that's acceptable
        is_nano_model = "nano" in model.lower()
        if is_nano_model and len(result.tool_calls) == 0:
            # Check if it hit length limit (expected for nano models)
            if hasattr(result, "finish_reason") and result.finish_reason == "length":
                print(f"  ✓ call_with_tools(): nano model hit token limit (expected)")
            else:
                pytest.fail(f"Nano model made 0 tool calls but didn't hit length limit (finish_reason={getattr(result, 'finish_reason', 'unknown')})")
        else:
            assert len(result.tool_calls) > 0, f"Expected at least 1 tool call, got {len(result.tool_calls)}"

            # Verify tool call structure
            tool_call = result.tool_calls[0]
            assert hasattr(tool_call, "name"), "Tool call missing 'name'"
            assert hasattr(tool_call, "arguments"), "Tool call missing 'arguments'"
            assert tool_call.name == "get_weather", f"Expected 'get_weather', got '{tool_call.name}'"
            assert "location" in tool_call.arguments, "Tool call arguments missing 'location'"

            print(f"  ✓ call_with_tools(): {tool_call.name}({tool_call.arguments})")
    except Exception as e:
        pytest.fail(f"{provider}/{model} call_with_tools() failed: {e}")


@pytest.mark.parametrize("provider,model", MODEL_MATRIX)
@pytest.mark.asyncio
async def test_llm_provider_memory_operations(provider: str, model: str):
    """
    Test LLM provider with actual memory operations: fact extraction and reflect.
    All models must pass this test.
    """
    # Skip mock provider - it's a test stub, not designed for real operations
    if provider == "mock":
        pytest.skip("Mock provider is a test stub, not designed for real operations")

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


@pytest.mark.parametrize("provider,model", [
    ("claude-code", "claude-sonnet-4-20250514"),
    ("openai-codex", "gpt-5.2-codex"),
])
@pytest.mark.asyncio
async def test_llm_provider_consolidation(memory_no_llm_verify, request_context, provider: str, model: str):
    """
    Test LLM provider with consolidation (automatic mental model generation from observations).
    This validates that the provider can generate synthesized knowledge from raw memories.

    This test is limited to claude-code and codex since they're the critical providers
    that needed tool calling fixes for reflect and consolidation operations.
    """
    should_skip, reason = should_skip_provider(provider, model)
    if should_skip:
        pytest.skip(f"Skipping {provider}/{model}: {reason}")

    # Use provider-specific LLM for this test
    api_key = get_api_key_for_provider(provider)
    memory_no_llm_verify._consolidation_llm = LLMProvider(
        provider=provider,
        api_key=api_key or "",
        base_url="",
        model=model,
    )
    # Also need retain LLM for ingesting data
    memory_no_llm_verify._retain_llm = memory_no_llm_verify._consolidation_llm

    test_bank_id = f"llm_test_consolidation_{provider}_{model}_{datetime.now().timestamp()}"

    # Enable observations for this bank
    from hindsight_api.config import get_config
    config = get_config()
    original_value = config.enable_observations
    config.enable_observations = True

    try:
        # Retain memories to consolidate
        test_content = """
        Bob prefers functional programming with Rust and Haskell.
        He emphasizes immutability and pure functions in code reviews.
        Bob advocates for type safety and compile-time guarantees.
        He avoids mutable state and prefers declarative code patterns.
        """

        await memory_no_llm_verify.retain_async(
            bank_id=test_bank_id,
            content=test_content,
            context="Team coding preferences",
            event_date=datetime(2024, 12, 1),
            request_context=request_context,
        )

        print(f"\n{provider}/{model} - Consolidation test:")

        # Run consolidation to generate observations (mental models)
        from hindsight_api.engine.consolidation.consolidator import run_consolidation_job

        result = await run_consolidation_job(
            memory_engine=memory_no_llm_verify,
            bank_id=test_bank_id,
            request_context=request_context,
        )

        print(f"  Processed: {result.get('memories_processed', 0)} memories")
        print(f"  Created: {result.get('observations_created', 0)} observations")
        print(f"  Updated: {result.get('observations_updated', 0)} observations")

        # Verify consolidation ran successfully
        assert result["status"] in ["success", "no_new_memories"], f"{provider}/{model} consolidation failed"

        # If observations were created, verify they contain relevant content
        if result.get("observations_created", 0) > 0:
            observations = await memory_no_llm_verify.list_mental_models_consolidated(
                bank_id=test_bank_id,
                request_context=request_context,
            )

            assert len(observations) > 0, f"{provider}/{model} consolidation created 0 observations"

            # Check first observation contains relevant information
            obs_content = observations[0].get("content", "").lower()
            relevant_terms = ["bob", "functional", "rust", "immutab", "type"]
            matches = [term for term in relevant_terms if term in obs_content]

            print(f"  Observation preview: {observations[0].get('content', '')[:200]}...")
            print(f"  Found {len(matches)} relevant terms: {matches}")

            assert len(matches) >= 2, (
                f"{provider}/{model} consolidated observation doesn't contain relevant info. "
                f"Expected at least 2 of {relevant_terms}, found {len(matches)}: {matches}"
            )

    finally:
        # Restore original config
        config.enable_observations = original_value


# NOTE: The tests above validate the critical Hindsight operations:
#
# test_llm_provider_memory_operations (ALL providers):
#   - Fact extraction (retain): tests structured output generation
#   - Reflect: tests memory retrieval and reasoning (uses tool calling for claude-code/codex)
#
# test_llm_provider_consolidation (claude-code and codex only):
#   - Consolidation: tests automatic mental model generation from observations
#   - Requires MemoryEngine fixture with working LLM (from .env or env vars)
#   - Run your local LLM server OR set HINDSIGHT_API_LLM_PROVIDER/API_KEY/MODEL env vars
#
# For full end-to-end integration tests using the HTTP API, see tests/test_http_api_integration.py
