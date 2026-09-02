"""Structured output via a forced tool call on the LiteLLM-backed providers.

Covers the flag (HINDSIGHT_API_LLM_STRUCTURED_OUTPUT_FORCED_TOOL) end to end:
config parsing, the config -> LLMProvider -> LiteLLMLLM wiring, the request shape
it produces, and the response path that substitutes the tool call's arguments for
the message content. Motivation: Bedrock Claude rejects the ``response_format``
route outright (``output_config.format: Extra inputs are not permitted``, #3300)
while accepting the identical schema as a tool.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from hindsight_api.config import ENV_LLM_STRUCTURED_OUTPUT_FORCED_TOOL, HindsightConfig
from hindsight_api.engine.llm_wrapper import LLMConfig
from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM

_MESSAGES = [{"role": "user", "content": "hi"}]


class _Facts(BaseModel):
    facts: list[str]


def _make_litellm(*, forced_tool: bool) -> LiteLLMLLM:
    return LiteLLMLLM(
        provider="bedrock",
        api_key="",
        base_url="",
        model="bedrock/au.anthropic.claude-haiku-4-5-20251001-v1:0",
        structured_output_forced_tool=forced_tool,
    )


class _Function:
    def __init__(self, name: str, arguments: Any):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name: str, arguments: Any):
        self.id = "call_1"
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, content: str | None, tool_calls: list[_ToolCall] | None = None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message, finish_reason: str):
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, message: _Message, finish_reason: str = "tool_calls"):
        self.choices = [_Choice(message, finish_reason)]
        self.usage = None


async def _call_capturing_request(llm: LiteLLMLLM, response: _Response) -> dict[str, Any]:
    """Run ``call`` against a stubbed completion and return the request kwargs."""
    completion = AsyncMock(return_value=response)
    llm._acompletion = completion  # type: ignore[method-assign]
    result = await llm.call(messages=_MESSAGES, response_format=_Facts, max_retries=0)
    return {"kwargs": completion.await_args.kwargs, "result": result}


# ── config ───────────────────────────────────────────────────────────────────


def test_forced_tool_defaults_off(monkeypatch):
    monkeypatch.delenv(ENV_LLM_STRUCTURED_OUTPUT_FORCED_TOOL, raising=False)
    assert HindsightConfig.from_env().llm_structured_output_forced_tool is False


def test_forced_tool_can_be_enabled(monkeypatch):
    monkeypatch.setenv(ENV_LLM_STRUCTURED_OUTPUT_FORCED_TOOL, "true")
    assert HindsightConfig.from_env().llm_structured_output_forced_tool is True


@pytest.mark.parametrize("value", ["", "yes", "tru", "enabled"])
def test_forced_tool_rejects_ambiguous_values(monkeypatch, value):
    monkeypatch.setenv(ENV_LLM_STRUCTURED_OUTPUT_FORCED_TOOL, value)

    with pytest.raises(ValueError, match=ENV_LLM_STRUCTURED_OUTPUT_FORCED_TOOL):
        HindsightConfig.from_env()


def test_llm_config_threads_flag_to_provider_impl():
    """LLMConfig -> create_llm_provider -> LiteLLMLLM carries the flag.

    Without this bridge the env var is inert: the provider silently keeps the
    default ``response_format`` transport.
    """
    llm = LLMConfig(
        provider="bedrock",
        api_key="",
        base_url="",
        model="au.anthropic.claude-haiku-4-5-20251001-v1:0",
        structured_output_forced_tool=True,
    )
    assert llm._provider_impl.structured_output_forced_tool is True


# ── request shape ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forced_tool_replaces_response_format_with_a_forced_tool():
    llm = _make_litellm(forced_tool=True)
    response = _Response(_Message(None, [_ToolCall("structured_response", '{"facts": ["the sky is blue"]}')]))

    captured = await _call_capturing_request(llm, response)

    kwargs = captured["kwargs"]
    assert "response_format" not in kwargs
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "structured_response"}}
    assert len(kwargs["tools"]) == 1
    function = kwargs["tools"][0]["function"]
    assert function["name"] == "structured_response"
    assert function["parameters"] == _Facts.model_json_schema()
    assert captured["result"] == _Facts(facts=["the sky is blue"])


@pytest.mark.asyncio
async def test_flag_off_keeps_response_format():
    llm = _make_litellm(forced_tool=False)
    response = _Response(_Message('{"facts": ["the sky is blue"]}'), finish_reason="stop")

    captured = await _call_capturing_request(llm, response)

    kwargs = captured["kwargs"]
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert kwargs["response_format"]["json_schema"]["schema"] == _Facts.model_json_schema()
    assert captured["result"] == _Facts(facts=["the sky is blue"])


@pytest.mark.asyncio
async def test_plain_calls_are_untouched_by_the_flag():
    """No ``response_format`` -> no tool is forced, so free-text calls still work."""
    llm = _make_litellm(forced_tool=True)
    completion = AsyncMock(return_value=_Response(_Message("hello"), finish_reason="stop"))
    llm._acompletion = completion  # type: ignore[method-assign]

    result = await llm.call(messages=_MESSAGES, max_retries=0)

    assert result == "hello"
    assert "tools" not in completion.await_args.kwargs


# ── response path ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forced_tool_accepts_already_decoded_arguments():
    """Some providers hand back decoded arguments instead of a JSON string."""
    llm = _make_litellm(forced_tool=True)
    response = _Response(_Message(None, [_ToolCall("structured_response", {"facts": ["grass is green"]})]))

    captured = await _call_capturing_request(llm, response)

    assert captured["result"] == _Facts(facts=["grass is green"])


@pytest.mark.asyncio
async def test_falls_back_to_text_when_the_tool_call_is_missing():
    """A gateway that drops ``tool_choice`` must not hard-fail the call."""
    llm = _make_litellm(forced_tool=True)
    response = _Response(_Message('{"facts": ["parsed from text"]}'), finish_reason="stop")

    captured = await _call_capturing_request(llm, response)

    assert captured["result"] == _Facts(facts=["parsed from text"])


@pytest.mark.asyncio
async def test_skip_validation_returns_the_raw_tool_arguments():
    llm = _make_litellm(forced_tool=True)
    response = _Response(_Message(None, [_ToolCall("structured_response", '{"facts": ["raw"]}')]))
    llm._acompletion = AsyncMock(return_value=response)  # type: ignore[method-assign]

    result = await llm.call(
        messages=_MESSAGES,
        response_format=_Facts,
        skip_validation=True,
        max_retries=0,
    )

    assert result == {"facts": ["raw"]}
