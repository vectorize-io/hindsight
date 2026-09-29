"""
`call_with_tools()` must send `reasoning_effort` for reasoning models, like `call()` does.

`call()` sets it; `call_with_tools()` built its own `call_params` and left it out.
Omitting it is not a neutral default — OpenAI rejects function tools on a
reasoning model unless `reasoning_effort` is present and set to "none":

    Function tools with reasoning_effort are not supported for gpt-5.6-terra in
    /v1/chat/completions. To use function tools, use /v1/responses or set
    reasoning_effort to 'none'.

Verified directly against the API for gpt-5.6-terra + function tools:
  reasoning_effort="low"  -> HTTP 400
  reasoning_effort absent -> HTTP 400
  reasoning_effort="none" -> tool call succeeds

So the omission makes `HINDSIGHT_API_LLM_REASONING_EFFORT` unable to fix the
failure, because that setting only ever reached the non-tool path. In practice
reflect (a tool-calling search loop) degrades to a no-search fallback while
still stamping `last_refreshed_at`, so the mental model looks refreshed.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_observations",
            "description": "Search raw observations",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def _make_llm(model: str, reasoning_effort: str = "none") -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="test",
        base_url="https://api.openai.com/v1",
        model=model,
        reasoning_effort=reasoning_effort,
    )


def _make_tool_call_response() -> MagicMock:
    mock_tc = MagicMock()
    mock_tc.id = "call_abc123"
    mock_tc.function.name = "search_observations"
    mock_tc.function.arguments = json.dumps({"query": "x"})

    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 120
    mock_response.usage.completion_tokens = 40
    mock_response.usage.total_tokens = 160
    # Explicit None: an auto-MagicMock is truthy and the reasoning-token
    # accounting would do arithmetic on it and crash.
    mock_response.usage.completion_tokens_details = None
    mock_response.choices[0].finish_reason = "tool_calls"
    mock_response.choices[0].message.content = None
    mock_response.choices[0].message.tool_calls = [mock_tc]
    return mock_response


async def _capture_call_params(llm: OpenAICompatibleLLM) -> dict:
    """Run call_with_tools against a mocked client and return the request kwargs."""
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _make_tool_call_response()
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=TOOLS,
            max_retries=0,
        )
    return create.await_args.kwargs


class TestToolPathReasoningEffort:
    @pytest.mark.asyncio
    async def test_reasoning_model_receives_reasoning_effort_on_tool_path(self):
        # Intention: a reasoning model must get reasoning_effort on the tool path.
        # Expected: the configured value is forwarded verbatim — "none" in
        # particular, which is the only value OpenAI accepts alongside function
        # tools, so it must not be dropped or coerced.
        params = await _capture_call_params(_make_llm("gpt-5.6-terra", "none"))
        assert params["reasoning_effort"] == "none"

    @pytest.mark.asyncio
    async def test_configured_effort_is_forwarded_verbatim(self):
        # Intention: the tool path must not substitute its own default.
        # Expected: "high" arrives as "high", proving the value comes from config
        # rather than the constructor default ("low").
        params = await _capture_call_params(_make_llm("gpt-5.6-terra", "high"))
        assert params["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", ["openai", "openrouter", "deepseek"])
    async def test_blast_radius_is_the_capability_check_not_the_provider(self, provider):
        # Intention: pin the intended contract deliberately. `_supports_reasoning_model()`
        # is a model-name check, not a provider check, and `call()` already sends the
        # parameter to these same provider/model pairs. Gating the tool path by provider
        # would reintroduce exactly the asymmetry this fix removes, so the tool path
        # must follow the capability check alone.
        # Expected: every OpenAI-compatible provider gets the value for a reasoning model.
        model = "deepseek-reasoner" if provider == "deepseek" else "gpt-5.6-terra"
        llm = OpenAICompatibleLLM(
            provider=provider,
            api_key="test",
            base_url="https://example.invalid/v1",
            model=model,
            reasoning_effort="none",
        )
        params = await _capture_call_params(llm)
        assert params["reasoning_effort"] == "none"

    @pytest.mark.asyncio
    async def test_non_reasoning_model_gets_no_reasoning_effort(self):
        # Intention: don't start sending the param to models that reject it.
        # Expected: gpt-4o is not a reasoning model, so the key is absent —
        # matching call()'s behaviour exactly.
        params = await _capture_call_params(_make_llm("gpt-4o", "none"))
        assert "reasoning_effort" not in params

    @pytest.mark.asyncio
    async def test_tool_path_matches_plain_path_for_same_model(self):
        # Intention: pin the two paths together so they cannot drift again —
        # this asymmetry is the whole bug.
        # Expected: call() and call_with_tools() agree on whether the param is
        # sent and on its value.
        llm = _make_llm("gpt-5.6-terra", "none")
        tool_params = await _capture_call_params(llm)

        plain_response = MagicMock()
        # call() inspects .error and model_dump() for a provider error payload;
        # auto-MagicMock values are truthy and would look like an error.
        plain_response.error = None
        plain_response.model_dump.return_value = {}
        plain_response.usage.prompt_tokens = 10
        plain_response.usage.completion_tokens = 5
        plain_response.usage.total_tokens = 15
        plain_response.usage.completion_tokens_details = None
        plain_response.choices[0].finish_reason = "stop"
        plain_response.choices[0].message.content = "ok"
        plain_response.choices[0].message.tool_calls = None
        with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
            create.return_value = plain_response
            await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
        plain_params = create.await_args.kwargs

        assert ("reasoning_effort" in tool_params) == ("reasoning_effort" in plain_params)
        assert tool_params["reasoning_effort"] == plain_params["reasoning_effort"]
