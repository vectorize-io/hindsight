"""
Regression tests for Codex provider tool-choice serialization.

The reflect agent selects a named tool through the canonical typed contract.

The Codex Responses API expects the function name at the top level instead:
  {"type": "function", "name": "recall"}

Sending the Chat Completions wire shape to Codex would fail with:
  400 Unknown parameter: 'tool_choice.function'
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.llm_interface import LLMToolChoice
from hindsight_api.engine.providers.codex_llm import CodexLLM

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


def build_llm() -> CodexLLM:
    with patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.4-mini",
        )


@pytest.mark.asyncio
async def test_codex_serializes_named_tool_choice_for_responses():
    llm = build_llm()
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    with patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response
        with patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = (None, [])
            await llm.call_with_tools(
                messages=[{"role": "user", "content": "recall the memory"}],
                tools=TOOLS,
                tool_choice=LLMToolChoice.named("recall"),
                max_retries=0,
            )
        sent_payload = mock_post.call_args.kwargs["json"]

    assert sent_payload["tool_choice"] == {"type": "function", "name": "recall"}


@pytest.mark.asyncio
async def test_codex_forced_tool_choice_still_yields_tool_calls():
    llm = build_llm()
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    tool_call = {"id": "call-1", "name": "recall", "arguments": {"query": "memory"}}
    with patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response
        with patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = (None, [tool_call])
            result = await llm.call_with_tools(
                messages=[{"role": "user", "content": "recall the memory"}],
                tools=TOOLS,
                tool_choice=LLMToolChoice.named("recall"),
                max_retries=0,
            )
        sent_payload = mock_post.call_args.kwargs["json"]

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "recall"
    assert sent_payload["tool_choice"] == {"type": "function", "name": "recall"}
