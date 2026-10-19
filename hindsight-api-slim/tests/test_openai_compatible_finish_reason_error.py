"""A 200 response whose choice ends with ``finish_reason="error"`` is a provider failure (#5612).

OpenRouter returns this shape when an upstream endpoint dies mid-generation: HTTP 200,
``finish_reason="error"``, a choice-level ``error`` object and no output. Read as a normal
reply it became "the model chose no tool", which reflect handles as model behaviour and
skips the forced retrieval step instead of retrying the request.
"""

from unittest.mock import AsyncMock, patch

import litellm
import pytest
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import convert_to_model_response_object
from openai._models import construct_type
from openai.types.chat import ChatCompletion
from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_REQUIRED
from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM, ProviderResponseError
from hindsight_api.tracing import LLMSpanRecorder, register_span_recorder, unregister_span_recorder

_TOOLS = [{"type": "function", "function": {"name": "recall", "parameters": {"type": "object", "properties": {}}}}]


def _llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai", api_key="sk-test", base_url="https://openrouter.ai/api/v1", model="openai/gpt-oss-120b"
    )


def _completion(finish_reason: str, *, content: str | None = None, tool_call: bool = False) -> ChatCompletion:
    message: dict = {"role": "assistant", "content": content}
    if tool_call:
        message["tool_calls"] = [
            {"id": "c1", "type": "function", "function": {"name": "recall", "arguments": '{"query": "x"}'}}
        ]
    choice: dict = {"index": 0, "finish_reason": finish_reason, "message": message}
    if finish_reason == "error":
        choice["error"] = {"code": 502, "message": "upstream connection reset"}
    payload = {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-oss-120b",
        "provider": "SomeUpstream",
        "choices": [choice],
        "usage": {"prompt_tokens": 4100, "completion_tokens": 37, "total_tokens": 4137},
    }
    return construct_type(type_=ChatCompletion, value=payload)  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_tool_call_retries_an_error_finish_then_returns_the_real_answer():
    llm = _llm()
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.side_effect = [_completion("error"), _completion("tool_calls", tool_call=True)]
        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "q"}],
            tools=_TOOLS,
            tool_choice=LLM_TOOL_CHOICE_REQUIRED,
            max_retries=1,
            initial_backoff=0,
        )
    assert create.await_count == 2
    assert [tc.name for tc in result.tool_calls] == ["recall"]


@pytest.mark.asyncio
async def test_tool_call_raises_instead_of_returning_an_empty_result_when_retries_run_out():
    llm = _llm()
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _completion("error")
        with pytest.raises(ProviderResponseError, match="finish_reason=error") as exc:
            await llm.call_with_tools(
                messages=[{"role": "user", "content": "q"}], tools=_TOOLS, max_retries=0, initial_backoff=0
            )
    # The served upstream and its error reach the message, so the failure can be attributed.
    assert "SomeUpstream" in str(exc.value)
    assert "upstream connection reset" in str(exc.value)


@pytest.mark.asyncio
async def test_call_rejects_an_error_finish_even_with_partial_content():
    """Keyed on finish_reason, not emptiness: partial text from a failed generation is not an answer."""
    llm = _llm()
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.side_effect = [_completion("error", content="The answer is"), _completion("stop", content="Paris.")]
        result = await llm.call(messages=[{"role": "user", "content": "q"}], max_retries=1, initial_backoff=0)
    assert create.await_count == 2
    assert result.content == "Paris."


class _Recorder:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def record_llm_call(self, **kwargs) -> None:
        self.records.append(kwargs)


@pytest.mark.asyncio
async def test_a_retried_attempt_is_traced_with_its_upstream():
    """A retried attempt was billed and must be countable per upstream, not only logged."""
    recorder = _Recorder()
    register_span_recorder(recorder)
    llm = _llm()
    try:
        with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
            create.side_effect = [_completion("error"), _completion("tool_calls", tool_call=True)]
            await llm.call_with_tools(
                messages=[{"role": "user", "content": "q"}], tools=_TOOLS, max_retries=1, initial_backoff=0
            )
    finally:
        unregister_span_recorder(recorder)
    failed, succeeded = recorder.records
    assert isinstance(failed["error"], ProviderResponseError)
    assert failed["served_by"] == "SomeUpstream"
    assert failed["input_tokens"] == 4100
    assert succeeded["error"] is None
    assert succeeded["served_by"] == "SomeUpstream"


def test_span_carries_the_upstream_from_the_error():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    LLMSpanRecorder(provider.get_tracer("t")).record_llm_call(
        provider="openai",
        model="m",
        scope="tools",
        messages=[],
        response_content=None,
        input_tokens=1,
        output_tokens=0,
        duration=0.1,
        error=ProviderResponseError("boom", served_by="SomeUpstream"),
    )
    (span,) = exporter.get_finished_spans()
    assert span.attributes["hindsight.provider.served_by"] == "SomeUpstream"


def _litellm_completion(finish_reason: str, *, tool_call: bool = False) -> litellm.ModelResponse:
    """Built the way LiteLLM builds it, which rewrites finish_reason "error" to "stop"."""
    raw = _completion(finish_reason, tool_call=tool_call).model_dump(mode="json")
    return convert_to_model_response_object(response_object=raw, model_response_object=litellm.ModelResponse())


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [False, True])
async def test_litellm_retries_an_error_finish_it_rewrote_to_stop(tools):
    llm = LiteLLMLLM(provider="litellm", api_key="unused", base_url="http://localhost:0/v1", model="m")
    responses = [_litellm_completion("error"), _litellm_completion("tool_calls", tool_call=True)]
    assert responses[0].choices[0].finish_reason == "stop"
    llm._acompletion = AsyncMock(side_effect=responses)  # type: ignore[method-assign]
    messages = [{"role": "user", "content": "q"}]
    if tools:
        recorder = _Recorder()
        register_span_recorder(recorder)
        try:
            result = await llm.call_with_tools(messages=messages, tools=_TOOLS, max_retries=1, initial_backoff=0)
        finally:
            unregister_span_recorder(recorder)
        assert [tc.name for tc in result.tool_calls] == ["recall"]
        assert [(r["error"] is not None, r["served_by"]) for r in recorder.records] == [
            (True, "SomeUpstream"),
            (False, "SomeUpstream"),
        ]
    else:
        with pytest.raises(ProviderResponseError, match="served_by=SomeUpstream"):
            await llm.call(messages=messages, max_retries=0, initial_backoff=0)
    assert llm._acompletion.await_count == (2 if tools else 1)
