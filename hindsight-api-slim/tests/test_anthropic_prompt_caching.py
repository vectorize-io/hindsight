"""Anthropic prompt caching via inline cache_control markers.

``LLMInterface.get_or_create_cached_prefix`` documents Anthropic as an
"inline-marker provider": rather than returning an explicit cache handle, the
provider marks the reusable prefix inside ``call`` / ``call_with_tools`` with
``cache_control`` breakpoints. Cache reads bill at ~10% of the base input
price; a marker below the model's minimum cacheable prefix is silently
ignored by the API (no premium), so marking is safe unconditionally.

Two breakpoints (of the 4 allowed):
- the system prompt, in both entry points — it is stable per scope (fact
  extraction reuses it across every chunk; reflect/consolidation put their
  stable instructions there), so tools+system cache across calls;
- the last message content block, in ``call_with_tools`` only — the reflect
  agent loop resends the whole growing conversation each iteration, so each
  request's end-marker becomes the next iteration's cache read point.
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
    resp.usage = MagicMock(input_tokens=10, output_tokens=2, cache_read_input_tokens=0)
    resp.stop_reason = "end_turn"
    return resp


def _tool_response():
    resp = MagicMock()
    resp.content = []
    resp.usage = MagicMock(input_tokens=10, output_tokens=2, cache_read_input_tokens=0)
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


async def test_call_with_tools_marks_system_and_last_message():
    provider = _make_provider()
    provider._client.messages.create = AsyncMock(return_value=_tool_response())

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        await provider.call_with_tools(
            messages=[
                {"role": "system", "content": "Reflect agent instructions."},
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "Working on it."},
                {"role": "user", "content": "Latest turn."},
            ],
            tools=[{"function": {"name": "recall", "description": "d", "parameters": {"type": "object"}}}],
            max_retries=0,
        )

    params = provider._client.messages.create.await_args.kwargs
    assert params["system"] == [{"type": "text", "text": "Reflect agent instructions.", "cache_control": EPHEMERAL}]

    messages = params["messages"]
    # Earlier messages carry no markers — only the final block gets one, so
    # the next iteration of the agent loop reads the whole prefix from cache.
    assert messages[0] == {"role": "user", "content": "Question?"}
    assert messages[1] == {"role": "assistant", "content": "Working on it."}
    assert messages[2]["content"] == [{"type": "text", "text": "Latest turn.", "cache_control": EPHEMERAL}]


async def test_call_with_tools_marks_last_block_of_tool_result_message():
    """Tool-result turns arrive as block lists; the marker goes on the last block."""
    provider = _make_provider()
    provider._client.messages.create = AsyncMock(return_value=_tool_response())

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        await provider.call_with_tools(
            messages=[
                {"role": "user", "content": "Question?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "t1", "function": {"name": "recall", "arguments": "{}"}},
                        {"id": "t2", "function": {"name": "recall", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "t1", "content": "result one"},
                {"role": "tool", "tool_call_id": "t2", "content": "result two"},
            ],
            tools=[{"function": {"name": "recall", "description": "d", "parameters": {"type": "object"}}}],
            max_retries=0,
        )

    messages = provider._client.messages.create.await_args.kwargs["messages"]
    last_blocks = messages[-1]["content"]
    assert last_blocks[-1]["type"] == "tool_result"
    assert last_blocks[-1]["cache_control"] == EPHEMERAL
    # The earlier tool-result message is unmarked.
    assert all("cache_control" not in block for block in messages[-2]["content"])
