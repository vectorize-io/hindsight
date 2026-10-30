"""Structured output with the schema in the prompt on the LiteLLM-backed providers.

Covers the flag (HINDSIGHT_API_LLM_STRUCTURED_OUTPUT_IN_PROMPT) end to end: config
parsing, the config -> LLMProvider -> LiteLLMLLM wiring, and the request it produces —
no ``response_format`` and no tools, the schema in the system prompt, the reply parsed.
Motivation: Bedrock Claude rejects the ``response_format`` route outright in some regions
(``output_config.format: Extra inputs are not permitted``, #3300). It used to send the
schema as a forced tool; Hindsight sends no tools any more.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from hindsight_api.config import ENV_LLM_STRUCTURED_OUTPUT_IN_PROMPT, HindsightConfig
from hindsight_api.engine.llm_wrapper import LLMConfig
from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM

_MESSAGES = [{"role": "user", "content": "hi"}]


class _Facts(BaseModel):
    facts: list[str]


def _make_litellm(*, in_prompt: bool) -> LiteLLMLLM:
    return LiteLLMLLM(
        provider="bedrock",
        api_key="",
        base_url="",
        model="bedrock/au.anthropic.claude-haiku-4-5-20251001-v1:0",
        structured_output_in_prompt=in_prompt,
    )


class _Message:
    def __init__(self, content: str | None):
        self.content = content
        self.tool_calls = None


class _Choice:
    def __init__(self, message: _Message, finish_reason: str):
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, message: _Message, finish_reason: str = "stop"):
        self.choices = [_Choice(message, finish_reason)]
        self.usage = None


async def _call_capturing_request(llm: LiteLLMLLM, response: _Response) -> dict[str, Any]:
    """Run ``call`` against a stubbed completion and return the request kwargs."""
    completion = AsyncMock(return_value=response)
    llm._acompletion = completion  # type: ignore[method-assign]
    result = (await llm.call(messages=_MESSAGES, response_format=_Facts, max_retries=0)).content
    return {"kwargs": completion.await_args.kwargs, "result": result}


# ── config ───────────────────────────────────────────────────────────────────


def test_in_prompt_defaults_off(monkeypatch):
    monkeypatch.delenv(ENV_LLM_STRUCTURED_OUTPUT_IN_PROMPT, raising=False)
    assert HindsightConfig.from_env().llm_structured_output_in_prompt is False


def test_in_prompt_can_be_enabled(monkeypatch):
    monkeypatch.setenv(ENV_LLM_STRUCTURED_OUTPUT_IN_PROMPT, "true")
    assert HindsightConfig.from_env().llm_structured_output_in_prompt is True


@pytest.mark.parametrize("value", ["", "yes", "tru", "enabled"])
def test_in_prompt_rejects_ambiguous_values(monkeypatch, value):
    monkeypatch.setenv(ENV_LLM_STRUCTURED_OUTPUT_IN_PROMPT, value)

    with pytest.raises(ValueError, match=ENV_LLM_STRUCTURED_OUTPUT_IN_PROMPT):
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
        structured_output_in_prompt=True,
    )
    assert llm._provider_impl.structured_output_in_prompt is True


# ── request shape ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_schema_goes_in_the_prompt_instead_of_response_format():
    llm = _make_litellm(in_prompt=True)
    response = _Response(_Message('{"facts": ["the sky is blue"]}'), finish_reason="stop")

    captured = await _call_capturing_request(llm, response)

    kwargs = captured["kwargs"]
    assert "response_format" not in kwargs and "tools" not in kwargs and "tool_choice" not in kwargs
    system = kwargs["messages"][0]
    assert system["role"] == "system" and '"facts"' in system["content"]
    assert kwargs["messages"][1:] == _MESSAGES
    assert _MESSAGES == [{"role": "user", "content": "hi"}], "the caller's messages are left alone"
    assert captured["result"] == _Facts(facts=["the sky is blue"])


@pytest.mark.asyncio
async def test_flag_off_keeps_response_format():
    llm = _make_litellm(in_prompt=False)
    response = _Response(_Message('{"facts": ["the sky is blue"]}'), finish_reason="stop")

    captured = await _call_capturing_request(llm, response)

    kwargs = captured["kwargs"]
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert kwargs["response_format"]["json_schema"]["schema"] == _Facts.model_json_schema()
    assert captured["result"] == _Facts(facts=["the sky is blue"])


@pytest.mark.asyncio
async def test_plain_calls_are_untouched_by_the_flag():
    """No ``response_format`` -> nothing added to the prompt, so free-text calls still work."""
    llm = _make_litellm(in_prompt=True)
    completion = AsyncMock(return_value=_Response(_Message("hello"), finish_reason="stop"))
    llm._acompletion = completion  # type: ignore[method-assign]

    result = (await llm.call(messages=_MESSAGES, max_retries=0)).content

    assert result == "hello"
    assert "tools" not in completion.await_args.kwargs


# ── response path ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skip_validation_returns_the_parsed_reply():
    llm = _make_litellm(in_prompt=True)
    llm._acompletion = AsyncMock(return_value=_Response(_Message('{"facts": ["raw"]}'), finish_reason="stop"))  # type: ignore[method-assign]

    result = (
        await llm.call(
            messages=_MESSAGES,
            response_format=_Facts,
            skip_validation=True,
            max_retries=0,
        )
    ).content

    assert result == {"facts": ["raw"]}
