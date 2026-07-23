from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


def _make_minimax(extra_body=None) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="minimax",
        api_key="test-key",
        base_url="",
        model="MiniMax-M3",
        extra_body=extra_body,
    )


def _text_response(content: str = "ok"):
    return SimpleNamespace(
        error=None,
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content, tool_calls=None, refusal=None),
            )
        ],
    )


def _tool_response():
    tool_call = SimpleNamespace(
        id="call_minimax_123",
        function=SimpleNamespace(name="recall", arguments='{"query": "Project Rin"}'),
    )
    return SimpleNamespace(
        error=None,
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(content=None, tool_calls=[tool_call], refusal=None),
            )
        ],
    )


@pytest.mark.asyncio
async def test_minimax_call_disables_thinking_by_default():
    llm = _make_minimax()
    llm._client.chat.completions.create = AsyncMock(return_value=_text_response())

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert llm._client.chat.completions.create.call_args.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_minimax_call_preserves_configured_thinking_extra_body():
    llm = _make_minimax(extra_body={"thinking": {"type": "enabled"}, "reasoning_split": True})
    llm._client.chat.completions.create = AsyncMock(return_value=_text_response())

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert llm._client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "thinking": {"type": "enabled"},
        "reasoning_split": True,
    }


@pytest.mark.asyncio
async def test_minimax_tool_call_disables_thinking_by_default():
    llm = _make_minimax()
    llm._client.chat.completions.create = AsyncMock(return_value=_tool_response())

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "Search memory."}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "recall",
                        "description": "Recall memories",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
            max_retries=0,
        )

    assert llm._client.chat.completions.create.call_args.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
