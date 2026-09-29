"""An explicitly configured `reasoning_effort` reaches the endpoint (issue #3449).

The parameter used to be gated on a substring match against the model *name*
(`gpt-5`, `o1`, `o3`), which can only ever match OpenAI's own products. Every
`HINDSIGHT_API_*_REASONING_EFFORT` variable was therefore a silent no-op on
self-hosted reasoning models — vLLM, Ollama, llama.cpp, TGI — exactly the
deployments where controlling thinking-token volume matters most, and nothing was
logged when the value was dropped. On the reporter's single-GPU vLLM box that was
the difference between consolidation finishing in ~24s and never finishing at all,
because only `reasoning_effort="none"` removes the thinking block and no
documented variable could produce it.

`provider=openai` with a custom base_url serves arbitrary models under arbitrary
names, so the name proves nothing: a configured value is a statement about the
deployment and is sent as given. Unset is equally a statement — no provider sends
a reasoning parameter at all, and every model runs at its own default effort
rather than one Hindsight picked (it used to resolve unset to "low").
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

VLLM_BASE_URL = "http://vllm-host:8000/v1"
# A self-hosted reasoning model: honours reasoning_effort, matches none of the names.
VLLM_MODEL = "Qwen/Qwen3-32B"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_observations",
            "description": "Search raw observations",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    },
]


def _make_llm(model: str, reasoning_effort: str | None, base_url: str = VLLM_BASE_URL) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="test",
        base_url=base_url,
        model=model,
        reasoning_effort=reasoning_effort,
    )


async def _capture_call_params(llm: OpenAICompatibleLLM) -> dict:
    """Run call() against a mocked client and return the request kwargs."""
    response = MagicMock()
    # call() inspects .error and model_dump() for a provider error payload; an
    # auto-MagicMock is truthy and would look like an error.
    response.error = None
    response.model_dump.return_value = {}
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    response.usage.completion_tokens_details = None
    response.choices[0].finish_reason = "stop"
    response.choices[0].message.content = "ok"
    response.choices[0].message.tool_calls = None
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = response
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
    return create.await_args.kwargs


async def _capture_tool_call_params(llm: OpenAICompatibleLLM) -> dict:
    """Run call_with_tools() against a mocked client and return the request kwargs."""
    tool_call = MagicMock()
    tool_call.id = "call_abc123"
    tool_call.function.name = "search_observations"
    tool_call.function.arguments = json.dumps({"query": "x"})

    response = MagicMock()
    response.usage.prompt_tokens = 120
    response.usage.completion_tokens = 40
    response.usage.total_tokens = 160
    response.usage.completion_tokens_details = None
    response.choices[0].finish_reason = "tool_calls"
    response.choices[0].message.content = None
    response.choices[0].message.tool_calls = [tool_call]
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = response
        await llm.call_with_tools(messages=[{"role": "user", "content": "hi"}], tools=TOOLS, max_retries=0)
    return create.await_args.kwargs


class TestConfiguredEffortReachesTheEndpoint:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("effort", ["none", "low", "high"])
    async def test_configured_effort_is_sent_to_unrecognised_model(self, effort):
        # Intention: the reported bug. The operator set the variable, the backend
        # honours it, and the model name matches nothing in the heuristic.
        # Expected: the value arrives verbatim — "none" especially, the only value
        # that removes the thinking block and previously unreachable through any
        # documented configuration.
        params = await _capture_call_params(_make_llm(VLLM_MODEL, effort))
        assert params["reasoning_effort"] == effort

    @pytest.mark.asyncio
    async def test_configured_effort_is_sent_on_the_tool_path_too(self):
        # Intention: reflect is a tool-calling loop, so a fix that only covers call()
        # leaves the operation that generates the most thinking tokens unchanged.
        # Expected: both paths agree.
        llm = _make_llm(VLLM_MODEL, "none")
        assert (await _capture_tool_call_params(llm))["reasoning_effort"] == "none"
        assert (await _capture_call_params(llm))["reasoning_effort"] == "none"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["gpt-oss-120b", "magistral-small", "glm-4.6", "custom-model"])
    async def test_name_is_not_used_to_infer_capability_on_custom_endpoints(self, model):
        # Intention: pin the contract, not one model name. Any name can appear on a
        # custom base_url, so no name may veto a configured value.
        # Expected: every one of these gets the operator's setting.
        params = await _capture_call_params(_make_llm(model, "none"))
        assert params["reasoning_effort"] == "none"

    @pytest.mark.asyncio
    async def test_unconfigured_effort_is_not_sent_to_unrecognised_model(self):
        # Intention: the heuristic still governs when the operator said nothing, so
        # endpoints that never received the parameter do not suddenly start getting
        # it (a backend that rejects the field would break on upgrade).
        # Expected: absent.
        params = await _capture_call_params(_make_llm(VLLM_MODEL, None))
        assert "reasoning_effort" not in params

    @pytest.mark.asyncio
    async def test_recognised_reasoning_model_gets_nothing_without_config(self):
        # Intention: unset means unset, even for a model whose name is recognisable.
        # Hindsight used to resolve unset to "low" and send it here, picking an effort
        # on the operator's behalf; the model now runs at its own default instead.
        # Expected: absent.
        params = await _capture_call_params(_make_llm("gpt-5.6-terra", None, base_url=""))
        assert "reasoning_effort" not in params

    @pytest.mark.asyncio
    async def test_the_request_shape_for_a_reasoning_model_survives_an_unset_effort(self):
        # Intention: `_supports_reasoning_model()` no longer decides whether the
        # parameter is sent, but it still governs the shape a reasoning model requires.
        # Expected: the max-completion-tokens floor and temperature suppression still
        # apply to gpt-5 with nothing configured.
        llm = _make_llm("gpt-5.6-terra", None, base_url="")
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
        with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
            create.return_value = response
            await llm.call(
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=2000,
                temperature=0.3,
                max_retries=0,
            )
        params = create.await_args.kwargs
        assert params["max_completion_tokens"] == 16000
        assert "temperature" not in params

    @pytest.mark.asyncio
    async def test_known_non_reasoning_model_drops_the_value_loudly(self, caplog):
        # Intention: the single carve-out. gpt-4o rejects the parameter outright, so
        # honouring the setting there would turn a silently ignored value into a hard
        # 400 for anyone who set the global variable and runs gpt-4o.
        # Expected: dropped — but logged at WARNING, never silently.
        with caplog.at_level("WARNING"):
            llm = _make_llm("gpt-4o", "high", base_url="")
        params = await _capture_call_params(llm)
        assert "reasoning_effort" not in params
        assert any("reasoning_effort" in r.message and "gpt-4o" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_configured_effort_does_not_change_the_request_shape(self):
        # Intention: keep the fix to the one parameter. `_supports_reasoning_model()`
        # also drives the ≥16000 max-completion-tokens floor, temperature suppression
        # and the max_tokens parameter name; a configured effort must not drag a
        # self-hosted model onto OpenAI's reasoning-model request shape.
        # Expected: max_tokens (not max_completion_tokens), the caller's budget
        # untouched, and temperature still forwarded.
        llm = _make_llm(VLLM_MODEL, "none")
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
        with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
            create.return_value = response
            await llm.call(
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=2000,
                temperature=0.3,
                max_retries=0,
            )
        params = create.await_args.kwargs
        assert params["max_tokens"] == 2000
        assert "max_completion_tokens" not in params
        assert params["temperature"] == 0.3
