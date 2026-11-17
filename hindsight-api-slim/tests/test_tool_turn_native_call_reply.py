"""A tool turn answered with a native tool call is read as the reply (#6520).

A schema tool turn sends no ``tools``, but it describes the functions and asks for one by
name. gpt-oss (on Bedrock through LiteLLM, and on Ollama's native API) answers with a native
call and empty ``content``; every provider read only ``content``, so every forced reflect
step came back "no usable call" and reflect failed on its first turn.
"""

import json
import types
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_REQUIRED, LLMToolChoice
from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM, ProviderResponseError
from hindsight_api.engine.providers.tools_via_schema import (
    calls_from_reply,
    native_call_reply,
    reply_violation,
    response_schema,
    schema_format,
)
from tests.ollama_stub import chat_body, ollama_stub

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_mental_models",
            "description": "Search mental models",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}, "query": {"type": "string"}},
                "required": ["reason", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Answer",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]
FORCED = LLMToolChoice.named("search_mental_models")
ARGS = {"reason": "first step", "query": "payments team lead"}


class _Facts(BaseModel):
    facts: list[str]


def _openai_response(tool_calls: list | None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                finish_reason="tool_calls",
                message=types.SimpleNamespace(content="", tool_calls=tool_calls, refusal=None),
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="gpt-oss-20b",
    )


def _sdk_call(name: str, arguments: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(id="c1", function=types.SimpleNamespace(name=name, arguments=arguments))


def _read_as_call(reply: str | None, tool_choice: LLMToolChoice = FORCED) -> dict:
    calls = calls_from_reply(reply, TOOLS, tool_choice)
    assert reply_violation(reply, calls, TOOLS) is None
    return {"name": calls[0].name, "arguments": calls[0].arguments}


def test_native_call_reads_as_the_forced_turns_arguments() -> None:
    fmt = schema_format(response_schema(TOOLS, FORCED))
    reply = native_call_reply(fmt, [_sdk_call("search_mental_models", json.dumps(ARGS))])
    assert _read_as_call(reply) == {"name": "search_mental_models", "arguments": ARGS}


def test_native_call_reads_as_a_free_turns_call() -> None:
    free = LLM_TOOL_CHOICE_REQUIRED
    fmt = schema_format(response_schema(TOOLS, free))
    reply = native_call_reply(fmt, [{"function": {"name": "done", "arguments": {"answer": "Alice"}}}])
    assert _read_as_call(reply, free) == {"name": "done", "arguments": {"answer": "Alice"}}


def test_native_call_to_an_undeclared_function_is_a_violation() -> None:
    fmt = schema_format(response_schema(TOOLS, LLM_TOOL_CHOICE_REQUIRED))
    reply = native_call_reply(fmt, [{"function": {"name": "browser.open", "arguments": {}}}])
    calls = calls_from_reply(reply, TOOLS, LLM_TOOL_CHOICE_REQUIRED)
    assert reply_violation(reply, calls, TOOLS) == "`browser.open` is not one of the declared functions"


def test_only_tool_turns_take_a_native_call() -> None:
    call = [_sdk_call("search_mental_models", "{}")]
    assert native_call_reply(_Facts, call) is None  # an ordinary structured call keeps failing loudly
    assert native_call_reply(None, call) is None
    assert native_call_reply(schema_format(response_schema(TOOLS, FORCED)), []) is None


@pytest.mark.asyncio
async def test_openai_compatible_tool_turn_reads_a_native_call() -> None:
    llm = OpenAICompatibleLLM(provider="openai", api_key="sk-test", base_url="", model="gpt-oss-20b")
    fmt = schema_format(response_schema(TOOLS, FORCED))
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _openai_response([_sdk_call("search_mental_models", json.dumps(ARGS))])
        result = await llm.call(
            messages=[{"role": "user", "content": "q"}], response_format=fmt, skip_validation=True, max_retries=0
        )
        assert result.content == {"call": {"name": "search_mental_models", "arguments": ARGS}}

        # A structured call that is not a tool turn still fails on empty content.
        with pytest.raises(ProviderResponseError, match="has_tool_calls=True"):
            await llm.call(
                messages=[{"role": "user", "content": "q"}],
                response_format=_Facts,
                skip_validation=True,
                max_retries=0,
            )


@pytest.mark.asyncio
async def test_litellm_tool_turn_reads_a_native_call() -> None:
    llm = LiteLLMLLM(provider="litellm", api_key="k", base_url="", model="bedrock/openai.gpt-oss-20b-1:0")
    fmt = schema_format(response_schema(TOOLS, FORCED))
    with patch.object(llm, "_acompletion", new_callable=AsyncMock) as completion:
        completion.return_value = _openai_response([_sdk_call("search_mental_models", json.dumps(ARGS))])
        result = await llm.call(
            messages=[{"role": "user", "content": "q"}], response_format=fmt, skip_validation=True, max_retries=0
        )
    assert result.content == {"call": {"name": "search_mental_models", "arguments": ARGS}}


@pytest.mark.asyncio
async def test_ollama_native_tool_turn_reads_a_native_call() -> None:
    body = chat_body("", done_reason="stop")
    body["message"]["tool_calls"] = [{"function": {"name": "search_mental_models", "arguments": ARGS}}]
    async with ollama_stub(body) as stub:
        llm = OpenAICompatibleLLM(provider="ollama", api_key="", base_url=stub.openai_base_url, model="gpt-oss:20b")
        result = await llm.call(
            messages=[{"role": "user", "content": "q"}],
            response_format=schema_format(response_schema(TOOLS, FORCED)),
            skip_validation=True,
            max_retries=0,
        )
        await llm.cleanup()
    assert result.content == {"call": {"name": "search_mental_models", "arguments": ARGS}}
