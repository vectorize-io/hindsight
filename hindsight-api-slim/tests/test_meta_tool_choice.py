"""Meta Model API rejects every ``tool_choice`` except ``"auto"``.

Verified live against https://api.meta.ai/v1 while wiring up the ``meta``
provider: reflect's forced first-turn retrieval came back as

    HTTP 400 - only `"auto"` is supported for `tool_choice`. `"none"`,
    `"required"`, and named function choices are not currently supported

which failed the whole reflect call rather than degrading it. This is the
opposite failure mode to LM Studio / Ollama (see test_lmstudio_tool_choice.py),
which accept the field and silently ignore it.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.llm_interface import (
    LLM_TOOL_CHOICE_AUTO,
    LLM_TOOL_CHOICE_REQUIRED,
    LLMToolChoice,
)
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_observations",
            "description": "Search raw observations",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Recall semantic memories",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    },
]


def _make_meta_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(provider="meta", api_key="test-key", base_url="", model="muse-spark-1.3")


def _tool_call_response() -> MagicMock:
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "search_observations"
    tc.function.arguments = json.dumps({"query": "tooling preferences"})

    resp = MagicMock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 20
    resp.usage.total_tokens = 120
    resp.usage.completion_tokens_details = None
    resp.choices[0].finish_reason = "tool_calls"
    resp.choices[0].message.content = None
    resp.choices[0].message.tool_calls = [tc]
    return resp


async def _captured_params(llm: OpenAICompatibleLLM, tool_choice) -> dict:
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _tool_call_response()
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "What does the user prefer?"}],
            tools=TOOLS,
            tool_choice=tool_choice,
            max_retries=0,
        )
        return create.call_args.kwargs


def test_meta_is_flagged_as_rejecting_non_auto_tool_choice():
    assert _make_meta_llm()._rejects_non_auto_tool_choice() is True


def test_other_openai_compatible_providers_keep_the_required_contract():
    """The carve-out is Meta's alone — it must not weaken every compatible endpoint."""
    other = OpenAICompatibleLLM(provider="openai", api_key="k", base_url="", model="gpt-5.6")
    assert other._rejects_non_auto_tool_choice() is False


@pytest.mark.asyncio
async def test_meta_required_tool_choice_is_omitted():
    """``required`` would be a hard 400, so the field comes off the request."""
    params = await _captured_params(_make_meta_llm(), LLM_TOOL_CHOICE_REQUIRED)
    assert "tool_choice" not in params


@pytest.mark.asyncio
async def test_meta_named_tool_choice_is_omitted_but_still_narrows_the_tools():
    """Named choices stay practically forced: the tools list is filtered to the one."""
    params = await _captured_params(_make_meta_llm(), LLMToolChoice.named("search_observations"))
    assert "tool_choice" not in params
    assert [t["function"]["name"] for t in params["tools"]] == ["search_observations"]


@pytest.mark.asyncio
async def test_meta_auto_tool_choice_sends_no_tool_choice_either():
    """``auto`` is the provider default and is already sent by omission."""
    params = await _captured_params(_make_meta_llm(), LLM_TOOL_CHOICE_AUTO)
    assert "tool_choice" not in params
