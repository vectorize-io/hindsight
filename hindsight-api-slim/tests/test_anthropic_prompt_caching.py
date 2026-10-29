"""Anthropic prompt caching via inline cache_control markers.

``LLMInterface.get_or_create_cached_prefix`` documents Anthropic as an
"inline-marker provider": rather than returning an explicit cache handle, the
provider marks the reusable prefix inside ``call`` with a ``cache_control``
breakpoint on the system prompt. It is stable per scope (fact extraction reuses
it across every chunk; reflect/consolidation put their stable instructions
there), so it caches across calls. Cache reads bill at ~10% of the base input
price; a marker below the model's minimum cacheable prefix is silently ignored
by the API (no premium), so marking is safe unconditionally.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.asyncio

EPHEMERAL = {"type": "ephemeral"}


def _make_provider():
    with patch("anthropic.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

        provider = AnthropicLLM(
            provider="anthropic",
            api_key="fake-key",
            base_url="",
            model="claude-sonnet-5",
        )
    provider._client = MagicMock()
    return provider


def _text_response(text: str = "ok"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=10, output_tokens=2, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    resp.stop_reason = "end_turn"
    return resp


def _tool_response():
    resp = MagicMock()
    resp.content = []
    resp.usage = MagicMock(input_tokens=10, output_tokens=2, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    resp.stop_reason = "end_turn"
    return resp


class _Out(BaseModel):
    facts: list[str]


async def test_call_marks_system_prompt_for_caching():
    provider = _make_provider()
    provider._client.messages.create = AsyncMock(return_value=_text_response())

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        await provider.call(
            messages=[
                {"role": "system", "content": "Stable extraction instructions."},
                {"role": "user", "content": "Chunk text."},
            ],
            scope="test",
            max_retries=0,
        )

    params = provider._client.messages.create.await_args.kwargs
    assert params["system"] == [{"type": "text", "text": "Stable extraction instructions.", "cache_control": EPHEMERAL}]
    # User messages are untouched in call() — one-shot calls share no
    # conversation prefix with each other, only the system prompt.
    assert params["messages"] == [{"role": "user", "content": "Chunk text."}]


async def test_call_non_strict_schema_lands_inside_cached_system_block():
    """Schema injection happens before marking, so the marked block includes it."""
    provider = _make_provider()
    provider._client.messages.create = AsyncMock(return_value=_text_response('{"facts": []}'))

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        await provider.call(
            messages=[
                {"role": "system", "content": "Extract."},
                {"role": "user", "content": "Text."},
            ],
            response_format=_Out,
            scope="test",
            max_retries=0,
        )

    params = provider._client.messages.create.await_args.kwargs
    assert len(params["system"]) == 1
    system_block = params["system"][0]
    assert system_block["cache_control"] == EPHEMERAL
    assert "Extract." in system_block["text"]
    assert "valid JSON" in system_block["text"]


async def test_call_without_system_prompt_sends_no_system_param():
    provider = _make_provider()
    provider._client.messages.create = AsyncMock(return_value=_text_response())

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            scope="test",
            max_retries=0,
        )

    assert "system" not in provider._client.messages.create.await_args.kwargs


def test_usage_counts_cache_reads_and_writes_as_input():
    """Anthropic reports cache reads/writes apart from input_tokens; usage counts them in it (#5541)."""
    from types import SimpleNamespace

    from hindsight_api.engine.providers.anthropic_llm import _usage_from_anthropic_response

    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=30, cache_read_input_tokens=1000, cache_creation_input_tokens=200
        )
    )
    usage = _usage_from_anthropic_response(response)

    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens) == (1210, 30, 1000)
