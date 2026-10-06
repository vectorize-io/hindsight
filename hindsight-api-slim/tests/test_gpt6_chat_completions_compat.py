"""GPT-6 on the `openai` (chat/completions) provider — issue #4891.

Two failures, both verified against the live API with `gpt-6-luna`:

1. `temperature` -> HTTP 400 "Unsupported value: 'temperature' does not support
   0.3 with this model. Only the default (1) value is supported." The model is a
   reasoning model, but the frozen request-shape list only knew `gpt-5`/`o1`/`o3`,
   so the suppression that already covers GPT-5 did not apply.

2. Function tools (reflect's search loop) -> HTTP 400 "Function tools with
   reasoning_effort are not supported for gpt-6-luna in /v1/chat/completions. To
   use function tools, use /v1/responses or set reasoning_effort to 'none'."
   An absent `reasoning_effort` fails the same way, so the default config
   (no effort configured) could not call tools at all.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIStatusError

from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_REQUIRED
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_observations",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    },
]

TOOLS_REJECTED_BODY = {
    "error": {
        "message": (
            "Function tools with reasoning_effort are not supported for gpt-6-luna in "
            "/v1/chat/completions. To use function tools, use /v1/responses or set "
            "reasoning_effort to 'none'."
        ),
        "type": "invalid_request_error",
        "param": "reasoning_effort",
    }
}


def _make_llm(model: str = "gpt-6-luna", reasoning_effort: str | None = None) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="test",
        base_url="",
        model=model,
        reasoning_effort=reasoning_effort,
    )


def _tools_400() -> APIStatusError:
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    return APIStatusError("400", response=response, body=TOOLS_REJECTED_BODY)


def _tool_call_response() -> MagicMock:
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "search_observations"
    tc.function.arguments = json.dumps({"query": "x"})
    response = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    response.usage.completion_tokens_details = None
    response.choices[0].finish_reason = "tool_calls"
    response.choices[0].message.content = None
    response.choices[0].message.tool_calls = [tc]
    return response


def _plain_response() -> MagicMock:
    response = MagicMock()
    response.error = None
    response.model_dump.return_value = {}
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    response.usage.completion_tokens_details = None
    response.choices[0].finish_reason = "stop"
    response.choices[0].message.content = "ok"
    response.choices[0].message.tool_calls = None
    return response


@pytest.mark.asyncio
async def test_temperature_is_suppressed_for_gpt6():
    # Intention: GPT-6 rejects any temperature but the default, exactly like GPT-5.
    # Expected: the parameter is dropped, and the reasoning-model request shape
    # (max_completion_tokens, not max_tokens) is used.
    llm = _make_llm()
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _plain_response()
        await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.3,
            max_completion_tokens=1000,
            max_retries=0,
        )
    params = create.await_args.kwargs
    assert "temperature" not in params
    assert "max_completion_tokens" in params


@pytest.mark.asyncio
async def test_tool_call_retries_with_reasoning_effort_none():
    # Intention: the API names its own remedy in the 400; apply it instead of
    # failing reflect until the operator finds the setting.
    # Expected: one repaired retry, and the retry carries reasoning_effort="none".
    llm = _make_llm()
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.side_effect = [_tools_400(), _tool_call_response()]
        result = await llm.call_with_tools(messages=[{"role": "user", "content": "hi"}], tools=TOOLS, max_retries=0)
    assert [tc.name for tc in result.tool_calls] == ["search_observations"]
    assert "reasoning_effort" not in create.await_args_list[0].kwargs
    assert create.await_args_list[1].kwargs["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_repair_is_not_retried_forever():
    # Intention: the repair grants exactly one extra attempt, so a model that
    # keeps rejecting the call still surfaces the error instead of looping.
    llm = _make_llm()
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.side_effect = [_tools_400(), _tools_400()]
        with pytest.raises(APIStatusError):
            await llm.call_with_tools(messages=[{"role": "user", "content": "hi"}], tools=TOOLS, max_retries=0)
    assert create.await_count == 2


@pytest.mark.asyncio
async def test_unrelated_400_is_not_repaired():
    # Intention: keep the repair to the one error that asks for it.
    llm = _make_llm()
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    err = APIStatusError("400", response=response, body={"error": {"message": "context length exceeded"}})
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.side_effect = err
        with pytest.raises(APIStatusError):
            await llm.call_with_tools(messages=[{"role": "user", "content": "hi"}], tools=TOOLS, max_retries=0)
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_rejected_tool_choice_retries_without_it():
    # Intention: Alibaba's Qwen 3.8 host refuses a forced tool_choice in thinking
    # mode; reflect forces its first call, so drop the field instead of failing.
    # Expected: one immediate retry without tool_choice, and only one.
    llm = _make_llm(model="qwen/qwen3.8-flash")
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    err = APIStatusError(
        "400",
        response=response,
        body={
            "error": {
                "message": "The tool_choice parameter does not support being set to required or object in thinking mode"
            }
        },
    )
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.side_effect = [err, _tool_call_response()]
        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=TOOLS,
            tool_choice=LLM_TOOL_CHOICE_REQUIRED,
            max_retries=0,
        )
        assert [tc.name for tc in result.tool_calls] == ["search_observations"]
        assert create.await_args_list[0].kwargs["tool_choice"] == "required"
        assert "tool_choice" not in create.await_args_list[1].kwargs

        create.reset_mock()
        create.side_effect = [err, err]
        with pytest.raises(APIStatusError):
            await llm.call_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=TOOLS,
                tool_choice=LLM_TOOL_CHOICE_REQUIRED,
                max_retries=0,
            )
        assert create.await_count == 2
