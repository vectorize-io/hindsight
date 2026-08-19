"""Tests for the ``openai-responses`` provider (OpenAI Responses API).

Two surfaces are covered:
- registration/wiring (default model, factory routing, API-key requirement), and
- request shaping + response parsing, mocking ``_client.responses.create``.

The load-bearing assertion is that ``reasoning`` and ``tools`` are sent together
on the tool path — the exact combination chat/completions rejects for gpt-5.6-terra
(#2983), and the reason this provider exists.
"""

import json
import types
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.llm_interface import LLMToolChoice, OutputTooLongError
from hindsight_api.engine.providers.openai_responses_llm import OpenAIResponsesLLM


def _fake_response(
    *, output_text="", output=None, input_tokens=100, output_tokens=20, reasoning_tokens=5, status="completed"
):
    """Build a duck-typed OpenAI Responses reply for mocking ``responses.create``."""
    usage = types.SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=types.SimpleNamespace(cached_tokens=0),
        output_tokens_details=types.SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )
    return types.SimpleNamespace(output_text=output_text, output=output or [], usage=usage, status=status)


def _function_call(*, call_id, name, arguments):
    return types.SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=arguments)


def _make_llm(model="gpt-5.6", reasoning_effort="high"):
    return OpenAIResponsesLLM(
        provider="openai-responses",
        api_key="sk-test",
        base_url="",
        model=model,
        reasoning_effort=reasoning_effort,
    )


def _mock_create(llm, response):
    create = AsyncMock(return_value=response)
    llm._client.responses.create = create
    return create


@pytest.mark.asyncio
@pytest.mark.parametrize("with_tools", [False, True])
async def test_request_holds_attempt_context(with_tools):
    llm = _make_llm()
    permit_held = False

    @asynccontextmanager
    async def attempt_context():
        nonlocal permit_held
        permit_held = True
        try:
            yield
        finally:
            permit_held = False

    async def create(**_kwargs):
        assert permit_held
        return _fake_response(output_text="ok")

    llm._client.responses.create = create
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        if with_tools:
            await llm.call_with_tools(
                messages=[{"role": "user", "content": "test"}],
                tools=[],
                attempt_context=attempt_context,
            )
        else:
            await llm.call(
                messages=[{"role": "user", "content": "test"}],
                attempt_context=attempt_context,
            )

    assert not permit_held
    assert llm.supports_attempt_scoped_concurrency()


# --------------------------------------------------------------------------- #
# Registration / wiring
# --------------------------------------------------------------------------- #


def test_default_model_is_a_reasoning_model():
    from hindsight_api.config import PROVIDER_DEFAULT_MODELS

    assert PROVIDER_DEFAULT_MODELS["openai-responses"] == "gpt-5.6"


def test_from_env_routes_to_openai_responses_llm(monkeypatch):
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.llm_wrapper import LLMProvider

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "openai-responses")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-test")
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_LLM_BASE_URL", raising=False)
    clear_config_cache()

    try:
        llm = LLMProvider.from_env()
        assert llm.provider == "openai-responses"
        assert llm.model == "gpt-5.6"
        assert isinstance(llm._provider_impl, OpenAIResponsesLLM)
    finally:
        clear_config_cache()


def test_openai_responses_requires_api_key():
    from hindsight_api.engine.llm_wrapper import requires_api_key

    assert requires_api_key("openai-responses") is True


def test_rejects_missing_api_key():
    from hindsight_api.engine.llm_wrapper import LLMProvider

    with pytest.raises(ValueError, match="API key is required for openai-responses"):
        LLMProvider(provider="openai-responses", api_key="", base_url="", model="gpt-5.6")


# --------------------------------------------------------------------------- #
# call()
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_call_sends_reasoning_object_and_omits_temperature_for_reasoning_model():
    llm = _make_llm()
    create = _mock_create(llm, _fake_response(output_text="hello"))
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        result = await llm.call(
            messages=[{"role": "user", "content": "hi"}], temperature=0.7, max_completion_tokens=1000
        )

    kwargs = create.call_args.kwargs
    assert result == "hello"
    assert kwargs["reasoning"] == {"effort": "high"}
    assert "temperature" not in kwargs  # reasoning models reject temperature
    assert kwargs["max_output_tokens"] == 16000  # reasoning floor enforced
    assert kwargs["store"] is False
    assert kwargs["input"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_call_non_reasoning_model_keeps_temperature_and_sends_no_reasoning():
    llm = _make_llm(model="gpt-4o-mini", reasoning_effort="low")
    create = _mock_create(llm, _fake_response(output_text="ok", reasoning_tokens=0))
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], temperature=0.5, max_completion_tokens=500)

    kwargs = create.call_args.kwargs
    assert "reasoning" not in kwargs
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_output_tokens"] == 500


@pytest.mark.asyncio
async def test_call_splits_reasoning_tokens_out_of_visible_output():
    llm = _make_llm()
    _mock_create(llm, _fake_response(output_text="hi", input_tokens=100, output_tokens=30, reasoning_tokens=12))
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        _, usage = await llm.call(messages=[{"role": "user", "content": "hi"}], return_usage=True)

    assert usage.input_tokens == 100
    assert usage.output_tokens == 18  # 30 - 12 reasoning
    assert usage.thoughts_tokens == 12
    assert usage.total_tokens == 118


class _Answer(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_call_strict_schema_uses_text_format_json_schema():
    llm = _make_llm()
    create = _mock_create(llm, _fake_response(output_text=json.dumps({"answer": "42"})))
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        result = await llm.call(
            messages=[{"role": "user", "content": "q"}],
            response_format=_Answer,
            strict_schema=True,
        )

    text_format = create.call_args.kwargs["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    assert "schema" in text_format
    assert isinstance(result, _Answer) and result.answer == "42"


@pytest.mark.asyncio
async def test_call_soft_schema_injects_schema_and_uses_json_object():
    llm = _make_llm()
    create = _mock_create(llm, _fake_response(output_text=json.dumps({"answer": "x"})))
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call(
            messages=[{"role": "system", "content": "be terse"}, {"role": "user", "content": "q"}],
            response_format=_Answer,
            strict_schema=False,
        )

    kwargs = create.call_args.kwargs
    assert kwargs["text"]["format"] == {"type": "json_object"}
    # schema text is appended to the system message
    assert "valid JSON matching this schema" in kwargs["input"][0]["content"]


@pytest.mark.asyncio
async def test_call_raises_output_too_long_on_truncation():
    llm = _make_llm()
    truncated = _fake_response(output_text="", status="incomplete")
    truncated.incomplete_details = types.SimpleNamespace(reason="max_output_tokens")
    _mock_create(llm, truncated)
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        with pytest.raises(OutputTooLongError):
            await llm.call(messages=[{"role": "user", "content": "hi"}])


# --------------------------------------------------------------------------- #
# call_with_tools()
# --------------------------------------------------------------------------- #

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "search memory",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]


@pytest.mark.asyncio
async def test_tool_path_sends_reasoning_and_tools_together():
    """The regression pin: reasoning + function tools in one request.

    This is exactly what chat/completions rejects for gpt-5.6-terra (#2983);
    the Responses API accepts it, which is the whole point of this provider.
    """
    llm = _make_llm()
    create = _mock_create(
        llm,
        _fake_response(output_text="", output=[_function_call(call_id="c1", name="recall", arguments='{"query":"x"}')]),
    )
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        result = await llm.call_with_tools(messages=[{"role": "user", "content": "q"}], tools=_TOOLS)

    kwargs = create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["tools"] == [
        {
            "type": "function",
            "name": "recall",
            "description": "search memory",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]
    assert result.tool_calls[0].name == "recall"
    assert result.tool_calls[0].arguments == {"query": "x"}
    assert result.tool_calls[0].id == "c1"
    assert result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_named_tool_choice_flattens_and_filters():
    llm = _make_llm()
    create = _mock_create(
        llm, _fake_response(output_text="", output=[_function_call(call_id="c1", name="recall", arguments="{}")])
    )
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "q"}],
            tools=_TOOLS,
            tool_choice=LLMToolChoice.named("recall"),
        )

    kwargs = create.call_args.kwargs
    assert kwargs["tool_choice"] == {"type": "function", "name": "recall"}
    assert len(kwargs["tools"]) == 1


@pytest.mark.asyncio
async def test_auto_tool_choice_omits_tool_choice():
    llm = _make_llm()
    create = _mock_create(llm, _fake_response(output_text="done"))
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call_with_tools(messages=[{"role": "user", "content": "q"}], tools=_TOOLS)

    assert "tool_choice" not in create.call_args.kwargs


@pytest.mark.asyncio
async def test_message_history_translation():
    """Assistant tool_calls -> function_call items; role=tool -> function_call_output."""
    llm = _make_llm()
    create = _mock_create(llm, _fake_response(output_text="done"))
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "recall", "arguments": '{"query":"x"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"memories":[]}'},
    ]
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call_with_tools(messages=history, tools=_TOOLS)

    items = create.call_args.kwargs["input"]
    assert items[0] == {"role": "user", "content": "q"}
    assert items[1] == {"type": "function_call", "call_id": "c1", "name": "recall", "arguments": '{"query":"x"}'}
    assert items[2] == {"type": "function_call_output", "call_id": "c1", "output": '{"memories":[]}'}


@pytest.mark.asyncio
async def test_tool_path_returns_content_when_no_tool_calls():
    llm = _make_llm()
    _mock_create(llm, _fake_response(output_text="final answer"))
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        result = await llm.call_with_tools(messages=[{"role": "user", "content": "q"}], tools=_TOOLS)

    assert result.content == "final answer"
    assert result.tool_calls == []


# --------------------------------------------------------------------------- #
# Generic LLM config flags (extra_body, service_tier, default_headers)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_extra_body_is_merged_into_requests():
    llm = OpenAIResponsesLLM(
        provider="openai-responses",
        api_key="sk-test",
        base_url="",
        model="gpt-5.6",
        extra_body={"foo": "bar"},
    )
    create = _mock_create(llm, _fake_response(output_text="ok"))
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call(messages=[{"role": "user", "content": "hi"}])

    assert create.call_args.kwargs["extra_body"] == {"foo": "bar"}


@pytest.mark.asyncio
async def test_service_tier_applied_on_both_paths():
    llm = OpenAIResponsesLLM(
        provider="openai-responses",
        api_key="sk-test",
        base_url="",
        model="gpt-5.6",
        openai_service_tier="flex",
    )
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        create = _mock_create(llm, _fake_response(output_text="ok"))
        await llm.call(messages=[{"role": "user", "content": "hi"}])
        assert create.call_args.kwargs["service_tier"] == "flex"

        create_tools = _mock_create(llm, _fake_response(output_text="ok"))
        await llm.call_with_tools(messages=[{"role": "user", "content": "hi"}], tools=_TOOLS)
        assert create_tools.call_args.kwargs["service_tier"] == "flex"


def test_default_headers_passed_to_client():
    llm = OpenAIResponsesLLM(
        provider="openai-responses",
        api_key="sk-test",
        base_url="",
        model="gpt-5.6",
        default_headers={"X-Trace-Id": "abc123"},
    )
    # AsyncOpenAI merges custom headers into its default_headers.
    assert llm._client.default_headers.get("X-Trace-Id") == "abc123"


def test_custom_base_url_targets_compatible_responses_endpoint():
    """A custom base_url routes the OpenAI SDK at an OpenAI-compatible /v1/responses endpoint."""
    llm = OpenAIResponsesLLM(
        provider="openai-responses",
        api_key="sk-test",
        base_url="https://gateway.example.com/v1",
        model="gpt-5.6",
    )
    assert llm.base_url == "https://gateway.example.com/v1"
    assert str(llm._client.base_url).rstrip("/") == "https://gateway.example.com/v1"


def test_factory_threads_service_tier_and_headers():
    """The factory forwards the generic flags to the standalone provider."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    llm = LLMProvider(
        provider="openai-responses",
        api_key="sk-test",
        base_url="",
        model="gpt-5.6",
        openai_service_tier="flex",
        default_headers={"X-Trace-Id": "abc123"},
    )
    impl = llm._provider_impl
    assert isinstance(impl, OpenAIResponsesLLM)
    assert impl.openai_service_tier == "flex"
    assert impl.default_headers == {"X-Trace-Id": "abc123"}


@pytest.mark.asyncio
async def test_call_omits_the_reasoning_object_when_no_effort_is_configured():
    """Unset means the Responses API applies its own default effort.

    Hindsight used to resolve unset to "low" in the config layer and send it here, so a
    deployment that never configured reasoning still had a level chosen for it. The
    reasoning-model request shape (token floor, no temperature) is unaffected.
    """
    llm = _make_llm(model="gpt-5.6", reasoning_effort=None)
    create = _mock_create(llm, _fake_response(output_text="ok"))

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    kwargs = create.await_args.kwargs
    assert "reasoning" not in kwargs
    assert "temperature" not in kwargs
