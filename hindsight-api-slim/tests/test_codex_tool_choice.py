"""
Regression tests for Codex provider tool-choice serialization.

The reflect agent selects a named tool through the canonical typed contract.

The Codex Responses API expects the function name at the top level instead:
  {"type": "function", "name": "recall"}

Sending the Chat Completions wire shape to Codex would fail with:
  400 Unknown parameter: 'tool_choice.function'

The same file covers how earlier tool calls and results are replayed in ``input``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.llm_interface import LLMToolChoice
from hindsight_api.engine.providers.codex_llm import CodexLLM, _CodexStreamReply
from tests.codex_stream_stub import stub_codex_stream

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
    with stub_codex_stream(llm, response) as mock_stream:
        with patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = _CodexStreamReply(text="", tool_calls=[], usage=None)
            await llm.call_with_tools(
                messages=[{"role": "user", "content": "recall the memory"}],
                tools=TOOLS,
                tool_choice=LLMToolChoice.named("recall"),
                max_retries=0,
            )
        sent_payload = mock_stream.call_args.kwargs["json"]

    assert sent_payload["tool_choice"] == {"type": "function", "name": "recall"}


@pytest.mark.asyncio
async def test_codex_forced_tool_choice_still_yields_tool_calls():
    llm = build_llm()
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    tool_call = {"id": "call-1", "name": "recall", "arguments": {"query": "memory"}}
    with stub_codex_stream(llm, response) as mock_stream:
        with patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = _CodexStreamReply(text="", tool_calls=[tool_call], usage=None)
            result = await llm.call_with_tools(
                messages=[{"role": "user", "content": "recall the memory"}],
                tools=TOOLS,
                tool_choice=LLMToolChoice.named("recall"),
                max_retries=0,
            )
        sent_payload = mock_stream.call_args.kwargs["json"]

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "recall"
    assert sent_payload["tool_choice"] == {"type": "function", "name": "recall"}


@pytest.mark.asyncio
async def test_call_with_tools_replays_tool_history_with_call_ids() -> None:
    """Earlier tool calls replay as linked function_call / function_call_output items (#5701).

    They used to be flattened to an empty assistant message plus a "Tool result: ..."
    user message, so the model lost which result answered which call.
    """
    llm = build_llm()
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    history = [
        {"role": "system", "content": "Use retrieved evidence."},
        {"role": "user", "content": "What is the recorded value?"},
        {
            "role": "assistant",
            "content": "Searching.",
            "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "recall", "arguments": '{"query":"value"}'}},
                {"id": "call_b", "type": "function", "function": {"name": "expand", "arguments": {"id": "m1"}}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": '{"value":42}'},
        {"role": "tool", "tool_call_id": "call_b", "content": {"text": "more"}},
    ]

    with (
        stub_codex_stream(llm, response) as mock_stream,
        patch.object(
            llm,
            "_parse_sse_tool_stream",
            new_callable=AsyncMock,
            return_value=_CodexStreamReply(text="", tool_calls=[], usage=None),
        ),
    ):
        await llm.call_with_tools(messages=history, tools=[], max_retries=0)

    payload = mock_stream.call_args.kwargs["json"]
    assert payload["instructions"] == "Use retrieved evidence."
    assert payload["input"] == [
        {"role": "user", "content": "What is the recorded value?"},
        {"role": "assistant", "content": "Searching."},
        {"type": "function_call", "call_id": "call_a", "name": "recall", "arguments": '{"query":"value"}'},
        {"type": "function_call", "call_id": "call_b", "name": "expand", "arguments": '{"id": "m1"}'},
        {"type": "function_call_output", "call_id": "call_a", "output": '{"value":42}'},
        {"type": "function_call_output", "call_id": "call_b", "output": '{"text": "more"}'},
    ]
