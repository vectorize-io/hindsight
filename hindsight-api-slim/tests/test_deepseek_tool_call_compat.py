"""Regression tests for DeepSeek OpenAI-compatible tool-call quirks."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_observations",
            "description": "Search observations",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
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
    },
]


def _make_deepseek_llm(model: str = "deepseek-v4-flash") -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="sk-test",
        base_url="https://api.deepseek.com",
        model=model,
    )


def _make_tool_call_response(tool_name: str = "search_observations") -> MagicMock:
    mock_tc = MagicMock()
    mock_tc.id = "call_deepseek_123"
    mock_tc.function.name = tool_name
    mock_tc.function.arguments = json.dumps({"query": "test"})

    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 120
    mock_response.choices[0].finish_reason = "tool_calls"
    mock_response.choices[0].message.content = None
    mock_response.choices[0].message.tool_calls = [mock_tc]
    return mock_response


def test_deepseek_flash_is_not_treated_as_reasoning_model():
    llm = _make_deepseek_llm("deepseek-v4-flash")

    assert llm._supports_reasoning_model() is False


def test_deepseek_reasoning_models_still_use_reasoning_parameters():
    llm = _make_deepseek_llm("deepseek-v4-pro")

    assert llm._supports_reasoning_model() is True


@pytest.mark.asyncio
async def test_deepseek_named_tool_choice_filters_tools_but_omits_tool_choice():
    """DeepSeek rejects required/named tool_choice but accepts a narrowed tools list."""
    llm = _make_deepseek_llm()
    named_tool_choice = {"type": "function", "function": {"name": "search_observations"}}

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _make_tool_call_response("search_observations")

        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "Search observations for Project-Rin."}],
            tools=TOOLS,
            tool_choice=named_tool_choice,
            max_retries=0,
        )

    assert result.tool_calls[0].name == "search_observations"

    sent_kwargs = mock_create.call_args.kwargs
    assert "tool_choice" not in sent_kwargs
    assert len(sent_kwargs["tools"]) == 1
    assert sent_kwargs["tools"][0]["function"]["name"] == "search_observations"


@pytest.mark.asyncio
async def test_deepseek_tool_history_gets_empty_reasoning_content_fallback():
    """DeepSeek requires reasoning_content when replaying assistant tool_calls."""
    llm = _make_deepseek_llm()
    messages = [
        {"role": "user", "content": "Search observations."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_deepseek_123",
                    "type": "function",
                    "function": {"name": "search_observations", "arguments": json.dumps({"query": "test"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_deepseek_123", "content": "{}"},
    ]

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _make_tool_call_response("recall")

        await llm.call_with_tools(
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_retries=0,
        )

    sent_messages = mock_create.call_args.kwargs["messages"]
    assert sent_messages[1]["reasoning_content"] == ""
    assert "reasoning_content" not in messages[1]


@pytest.mark.asyncio
async def test_deepseek_tool_history_preserves_existing_reasoning_content():
    llm = _make_deepseek_llm()
    messages = [
        {"role": "user", "content": "Search observations."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "provider reasoning scratchpad",
            "tool_calls": [
                {
                    "id": "call_deepseek_123",
                    "type": "function",
                    "function": {"name": "search_observations", "arguments": json.dumps({"query": "test"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_deepseek_123", "content": "{}"},
    ]

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _make_tool_call_response("recall")

        await llm.call_with_tools(
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_retries=0,
        )

    sent_messages = mock_create.call_args.kwargs["messages"]
    assert sent_messages[1]["reasoning_content"] == "provider reasoning scratchpad"
