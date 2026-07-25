"""Regression tests for LiteLLM named tool-choice serialization.

Hindsight sends the canonical named choice to ``litellm.acompletion``. LiteLLM
must flatten it when a model uses the Responses API while preserving the Chat
Completions shape for chat models (#2953).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.types.llms.openai import ResponsesAPIResponse

from hindsight_api.engine.llm_interface import LLMToolChoice
from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Recall semantic memories",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def _chat_response():
    message = SimpleNamespace(content=None, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0)
    return SimpleNamespace(choices=[choice], usage=usage)


def _responses_response() -> ResponsesAPIResponse:
    return ResponsesAPIResponse.model_construct(
        id="resp_1",
        created_at=0,
        model="gpt-5.3-codex",
        object="response",
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        usage={
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    )


@pytest.mark.asyncio
async def test_litellm_responses_bridge_flattens_named_tool_choice():
    llm = LiteLLMLLM(
        provider="litellm",
        api_key="test-key",
        base_url="",
        model="github_copilot/gpt-5.3-codex",
    )

    with (
        patch.object(Authenticator, "get_api_key", return_value="copilot-key"),
        patch.object(Authenticator, "get_api_base", return_value="https://api.githubcopilot.com"),
        patch.object(litellm, "aresponses", new=AsyncMock(return_value=_responses_response())) as mock_responses,
    ):
        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "recall the memory"}],
            tools=TOOLS,
            tool_choice=LLMToolChoice.named("recall"),
            max_retries=0,
        )

    assert mock_responses.call_args.kwargs["tool_choice"] == {"type": "function", "name": "recall"}
    assert result.content == "done"


@pytest.mark.asyncio
async def test_litellm_keeps_named_tool_choice_for_chat_completions():
    llm = LiteLLMLLM(
        provider="litellm",
        api_key="test-key",
        base_url="",
        model="openai/gpt-4o-mini",
    )
    llm._acompletion = AsyncMock(return_value=_chat_response())

    await llm.call_with_tools(
        messages=[{"role": "user", "content": "recall the memory"}],
        tools=TOOLS,
        tool_choice=LLMToolChoice.named("recall"),
        max_retries=0,
    )

    assert llm._acompletion.call_args.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "recall"},
    }
