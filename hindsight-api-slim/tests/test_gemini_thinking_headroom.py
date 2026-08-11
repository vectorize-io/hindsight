"""Tests for the thinking-token headroom on ``max_output_tokens`` (#3365).

Gemini charges thinking tokens against ``max_output_tokens`` along with the
visible reply, so passing a caller's budget straight through leaves a thinking
model nothing to reply with. A mental-model page configured at 3072 came back as
337 visible tokens ending mid-word, and the refresh still reported success with
no warnings, so the truncation was invisible. ``OpenAICompatibleLLM`` already
floors the budget for o1/o3/GPT-5 (#2630); these tests pin the same treatment on
the Gemini adapter, across all three places it builds a request.

Deterministic: the genai client is mocked and we assert on the config/JSON the
adapter builds, so no API key and no network are involved.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("google.genai")

from hindsight_api.engine.providers.gemini_llm import (  # noqa: E402
    MIN_THINKING_MAX_OUTPUT_TOKENS,
    GeminiLLM,
    _max_output_tokens_with_headroom,
)

# The budget from the issue report: a page sized for 3072 lost ~2735 of it to
# thinking and rendered a single heading.
TRUNCATED_PAGE_BUDGET = 3072


def _make_gemini_provider(model: str = "gemini-2.5-flash") -> GeminiLLM:
    """A GeminiLLM whose client is a mock, so nothing leaves the process."""
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        provider = GeminiLLM(provider="gemini", api_key="fake-api-key", base_url="", model=model)
    provider._client = MagicMock()
    return provider


def _fake_response():
    response = MagicMock()
    response.text = "hello"
    response.candidates = [MagicMock(finish_reason="STOP")]
    response.usage_metadata = MagicMock(prompt_token_count=5, candidates_token_count=2)
    return response


def _openai_body(max_completion_tokens: int = TRUNCATED_PAGE_BUDGET) -> dict:
    return {
        "messages": [{"role": "user", "content": "Paris is the capital of France."}],
        "max_completion_tokens": max_completion_tokens,
    }


# ─── the sizing rule ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model",
    [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "google/gemini-3.1-flash-lite",  # vertexai keeps the vendor prefix
        "gemini-flash-latest",  # unrecognized alias: assume it thinks
    ],
)
def test_thinking_models_get_headroom(model):
    assert _max_output_tokens_with_headroom(model, TRUNCATED_PAGE_BUDGET) == MIN_THINKING_MAX_OUTPUT_TOKENS


@pytest.mark.parametrize("model", ["gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-1.5-pro"])
def test_non_thinking_models_are_left_alone(model):
    """Pre-2.5 models never reason, so their budget already is the visible budget."""
    assert _max_output_tokens_with_headroom(model, TRUNCATED_PAGE_BUDGET) == TRUNCATED_PAGE_BUDGET


def test_budget_above_the_floor_is_never_lowered():
    """The floor only ever raises: a caller asking for more keeps it."""
    generous = MIN_THINKING_MAX_OUTPUT_TOKENS * 2
    assert _max_output_tokens_with_headroom("gemini-3.5-flash", generous) == generous


# ─── call() ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_sends_headroom_on_a_thinking_model():
    provider = _make_gemini_provider()
    provider._client.aio.models.generate_content = AsyncMock(return_value=_fake_response())

    await provider.call(
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=TRUNCATED_PAGE_BUDGET,
        scope="test",
    )

    config = provider._client.aio.models.generate_content.call_args.kwargs["config"]
    assert config.max_output_tokens == MIN_THINKING_MAX_OUTPUT_TOKENS


@pytest.mark.asyncio
async def test_call_passes_the_budget_through_on_a_non_thinking_model():
    provider = _make_gemini_provider(model="gemini-2.0-flash")
    provider._client.aio.models.generate_content = AsyncMock(return_value=_fake_response())

    await provider.call(
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=TRUNCATED_PAGE_BUDGET,
        scope="test",
    )

    config = provider._client.aio.models.generate_content.call_args.kwargs["config"]
    assert config.max_output_tokens == TRUNCATED_PAGE_BUDGET


@pytest.mark.asyncio
async def test_verify_connection_gets_headroom_too():
    """Startup verification asks for 100 tokens, which a thinking model spends on thinking.

    This is #2630 in Gemini's clothing: the OpenAI-compatible path hit the same wall
    when its verification budget was exhausted by reasoning. Verification goes through
    ``call``, so the floor covers it, and this pins that it keeps doing so.
    """
    provider = _make_gemini_provider()
    provider._client.aio.models.generate_content = AsyncMock(return_value=_fake_response())

    await provider.verify_connection()

    config = provider._client.aio.models.generate_content.call_args.kwargs["config"]
    assert config.max_output_tokens == MIN_THINKING_MAX_OUTPUT_TOKENS


@pytest.mark.asyncio
async def test_call_without_a_budget_sets_no_ceiling():
    """No caller budget means no max_output_tokens, headroom or not."""
    provider = _make_gemini_provider()
    provider._client.aio.models.generate_content = AsyncMock(return_value=_fake_response())

    await provider.call(messages=[{"role": "user", "content": "hi"}], scope="test")

    config = provider._client.aio.models.generate_content.call_args.kwargs["config"]
    assert config is None or config.max_output_tokens is None


# ─── call_with_tools() ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_with_tools_sends_headroom_on_a_thinking_model():
    provider = _make_gemini_provider()

    part = MagicMock()
    part.text = "answer"
    part.function_call = None
    response = MagicMock()
    response.candidates = [MagicMock(content=MagicMock(parts=[part]))]
    response.usage_metadata = MagicMock(prompt_token_count=5, candidates_token_count=3)
    provider._client.aio.models.generate_content = AsyncMock(return_value=response)

    await provider.call_with_tools(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ],
        max_completion_tokens=TRUNCATED_PAGE_BUDGET,
        scope="test",
    )

    config = provider._client.aio.models.generate_content.call_args.kwargs["config"]
    assert config.max_output_tokens == MIN_THINKING_MAX_OUTPUT_TOKENS


# ─── batch translation ────────────────────────────────────────────────────────


def test_batch_request_gets_headroom_on_a_thinking_model():
    """The batch path builds raw request JSON, so it needs the same sizing."""
    request = GeminiLLM._openai_body_to_gemini_request(_openai_body(), "gemini-3.5-flash")
    assert request["generationConfig"]["maxOutputTokens"] == MIN_THINKING_MAX_OUTPUT_TOKENS


def test_batch_request_passes_the_budget_through_on_a_non_thinking_model():
    request = GeminiLLM._openai_body_to_gemini_request(_openai_body(), "gemini-2.0-flash")
    assert request["generationConfig"]["maxOutputTokens"] == TRUNCATED_PAGE_BUDGET


def test_submit_batch_sizes_against_the_configured_model():
    """submit_batch must hand its own model down; the per-line JSON carries none."""
    jsonl = GeminiLLM._translate_requests(
        [{"custom_id": "chunk_0", "body": _openai_body()}],
        "gemini-3.5-flash",
    )
    assert f'"maxOutputTokens": {MIN_THINKING_MAX_OUTPUT_TOKENS}' in jsonl
