"""Returned LiteLLM responses must count even when Hindsight rejects them."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from hindsight_api.engine.llm_interface import OutputTooLongError
from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM


class _Answer(BaseModel):
    answer: int


def _provider() -> LiteLLMLLM:
    return LiteLLMLLM(provider="litellm", api_key="unused", base_url="http://localhost:0/v1", model="test-model")


def _response(content: str, input_tokens: int, output_tokens: int, finish_reason: str = "stop"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.mark.asyncio
async def test_completed_response_records_one_successful_attempt():
    provider = _provider()
    provider._acompletion = AsyncMock(return_value=_response("ok", 12, 3))
    collector = MagicMock()

    with patch("hindsight_api.engine.providers.litellm_llm.get_metrics_collector", return_value=collector):
        result = await provider.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert result.content == "ok"
    collector.record_llm_call.assert_called_once()
    assert collector.record_llm_call.call_args.kwargs["success"] is True
    assert collector.record_llm_call.call_args.kwargs["input_tokens"] == 12
    assert collector.record_llm_call.call_args.kwargs["output_tokens"] == 3


@pytest.mark.asyncio
async def test_truncated_response_records_its_billed_tokens_once():
    provider = _provider()
    provider._acompletion = AsyncMock(return_value=_response("partial", 100, 32_000, "length"))
    collector = MagicMock()

    with patch("hindsight_api.engine.providers.litellm_llm.get_metrics_collector", return_value=collector):
        with pytest.raises(OutputTooLongError):
            await provider.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    collector.record_llm_call.assert_called_once()
    assert collector.record_llm_call.call_args.kwargs["success"] is False
    assert collector.record_llm_call.call_args.kwargs["input_tokens"] == 100
    assert collector.record_llm_call.call_args.kwargs["output_tokens"] == 32_000


@pytest.mark.asyncio
async def test_invalid_json_retry_records_each_returned_response_once():
    provider = _provider()
    provider._acompletion = AsyncMock(
        side_effect=[_response("not json", 90, 32_000), _response('{"answer": 2}', 91, 8)]
    )
    collector = MagicMock()

    with patch("hindsight_api.engine.providers.litellm_llm.get_metrics_collector", return_value=collector):
        result = await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
            max_retries=1,
            initial_backoff=0,
        )

    assert result.content == _Answer(answer=2)
    assert collector.record_llm_call.call_count == 2
    assert [call.kwargs["success"] for call in collector.record_llm_call.call_args_list] == [False, True]
    assert [call.kwargs["output_tokens"] for call in collector.record_llm_call.call_args_list] == [32_000, 8]


@pytest.mark.asyncio
async def test_schema_rejection_records_returned_response():
    provider = _provider()
    provider._acompletion = AsyncMock(return_value=_response('{"answer": "wrong"}', 80, 12))
    collector = MagicMock()

    with patch("hindsight_api.engine.providers.litellm_llm.get_metrics_collector", return_value=collector):
        with pytest.raises(ValidationError):
            await provider.call(messages=[{"role": "user", "content": "hi"}], response_format=_Answer, max_retries=0)

    collector.record_llm_call.assert_called_once()
    assert collector.record_llm_call.call_args.kwargs["success"] is False
    assert collector.record_llm_call.call_args.kwargs["input_tokens"] == 80
    assert collector.record_llm_call.call_args.kwargs["output_tokens"] == 12


@pytest.mark.asyncio
async def test_invalid_tool_arguments_record_returned_response():
    provider = _provider()
    response = _response("", 75, 15, "tool_calls")
    response.choices[0].message.tool_calls = [
        SimpleNamespace(id="call_1", function=SimpleNamespace(name="lookup", arguments="not json"))
    ]
    provider._acompletion = AsyncMock(return_value=response)
    collector = MagicMock()

    with patch("hindsight_api.engine.providers.litellm_llm.get_metrics_collector", return_value=collector):
        with pytest.raises(json.JSONDecodeError):
            await provider.call_with_tools(messages=[{"role": "user", "content": "hi"}], tools=[], max_retries=0)

    collector.record_llm_call.assert_called_once()
    assert collector.record_llm_call.call_args.kwargs["success"] is False
    assert collector.record_llm_call.call_args.kwargs["input_tokens"] == 75
    assert collector.record_llm_call.call_args.kwargs["output_tokens"] == 15
