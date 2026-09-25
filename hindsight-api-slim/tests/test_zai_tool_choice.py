"""Z.AI rejects every ``tool_choice`` except ``"auto"``, direct or via a gateway.

Reported in #4246 against ``z-ai/glm-5.3-flash``: reflect's forced first-turn
retrieval comes back as HTTP 400 ``Tool choice must be auto``, with the same
request under ``auto`` calling the tool. Same failure mode as Meta Model API
(see test_meta_tool_choice.py), except that the reported route reaches Z.AI
through OpenRouter, where ``provider`` is the gateway and only the model id
names the endpoint.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.llm_interface import (
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


def _llm(provider: str, model: str) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(provider=provider, api_key="test-key", base_url="", model=model)


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


async def _captured_params(llm: OpenAICompatibleLLM, tool_choice: LLMToolChoice) -> dict[str, Any]:
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _tool_call_response()
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "What does the user prefer?"}],
            tools=TOOLS,
            tool_choice=tool_choice,
            max_retries=0,
        )
        return create.call_args.kwargs


def test_zai_provider_is_flagged_as_rejecting_non_auto_tool_choice():
    assert _llm("zai", "glm-4.5-flash")._rejects_non_auto_tool_choice() is True


@pytest.mark.parametrize(
    "provider,model",
    [("openrouter", "z-ai/glm-5.3-flash"), ("requesty", "zai/glm-4.6")],
)
def test_gateway_routed_zai_is_flagged_by_the_model_namespace(provider, model):
    """The reported configuration: ``provider`` is the gateway, not the endpoint."""
    assert _llm(provider, model)._rejects_non_auto_tool_choice() is True


@pytest.mark.parametrize(
    "provider,model",
    [("openrouter", "openai/gpt-5.6"), ("openai", "gpt-5.6"), ("openai", "glm-4.6")],
)
def test_other_routes_keep_the_required_contract(provider, model):
    """Including a bare GLM model name: those weights can be served by anyone."""
    assert _llm(provider, model)._rejects_non_auto_tool_choice() is False


@pytest.mark.asyncio
async def test_zai_required_tool_choice_is_omitted():
    """``required`` would be a hard 400, so the field comes off the request."""
    params = await _captured_params(_llm("zai", "glm-4.5-flash"), LLM_TOOL_CHOICE_REQUIRED)
    assert "tool_choice" not in params


@pytest.mark.asyncio
async def test_gateway_routed_zai_required_tool_choice_is_omitted():
    params = await _captured_params(_llm("openrouter", "z-ai/glm-5.3-flash"), LLM_TOOL_CHOICE_REQUIRED)
    assert "tool_choice" not in params


@pytest.mark.asyncio
async def test_gateway_routed_zai_named_tool_choice_is_omitted_but_still_narrows_the_tools():
    """Named choices stay practically forced: the tools list is filtered to the one."""
    params = await _captured_params(_llm("openrouter", "z-ai/glm-5.3-flash"), LLMToolChoice.named("recall"))
    assert "tool_choice" not in params
    assert [t["function"]["name"] for t in params["tools"]] == ["recall"]


@pytest.mark.asyncio
async def test_other_models_on_the_same_gateway_still_send_required():
    """The carve-out follows the endpoint, not the gateway it was reached through."""
    params = await _captured_params(_llm("openrouter", "openai/gpt-5.6"), LLM_TOOL_CHOICE_REQUIRED)
    assert params["tool_choice"] == "required"
