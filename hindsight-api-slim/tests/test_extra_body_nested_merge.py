"""Configured extra-body params merge into the objects Hindsight sets, key by key (#6640).

``HINDSIGHT_API_LLM_EXTRA_BODY={"output_config": {"effort": "low"}}`` is how an operator sets
effort on Claude. Hindsight puts the strict response schema in that same object
(``output_config.format``), and the SDKs merge ``extra_body`` one level deep only, so the
operator's object used to replace Hindsight's and the schema never reached the model. Same
for ``text`` / ``reasoning`` on the OpenAI Responses and Codex bodies. Each test checks the
body as it goes on the wire, after the SDK's own merge.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from anthropic import BadRequestError
from anthropic._base_client import _merge_mappings as anthropic_merge
from openai._base_client import _merge_mappings as openai_merge
from pydantic import BaseModel

from hindsight_api.engine.llm_interface import merge_extra_body
from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM
from hindsight_api.engine.providers.codex_llm import CodexLLM, _CodexStreamReply
from hindsight_api.engine.providers.openai_responses_llm import OpenAIResponsesLLM
from hindsight_api.engine.response_models import BatchRequest
from tests.codex_stream_stub import stub_codex_stream


class _Answer(BaseModel):
    answer: str


def test_merge_extra_body_deep_merges_dicts_and_operator_wins_leaves():
    params = {"output_config": {"format": {"type": "json_schema", "strict": True}}, "model": "m"}
    extra = {"output_config": {"effort": "low", "format": {"strict": False}}, "model": "x", "top_p": 0.9}

    assert merge_extra_body(params, extra) == {
        "output_config": {"effort": "low", "format": {"type": "json_schema", "strict": False}},
        "model": "x",
        "top_p": 0.9,
    }
    # The request is not touched; only the returned extra body carries the merge.
    assert params["output_config"] == {"format": {"type": "json_schema", "strict": True}}


def _anthropic(extra_body: dict) -> AnthropicLLM:
    with patch("anthropic.AsyncAnthropic") as client_cls:
        client_cls.return_value = MagicMock()
        provider = AnthropicLLM(
            provider="anthropic", api_key="k", base_url="", model="claude-sonnet-5-5", extra_body=extra_body
        )
    provider._client = MagicMock()
    return provider


def _text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(input_tokens=5, output_tokens=2, cache_read_input_tokens=0)
    response.stop_reason = "end_turn"
    return response


def _anthropic_wire(kwargs: dict) -> dict:
    sent = {k: v for k, v in kwargs.items() if k != "extra_body"}
    return anthropic_merge(sent, kwargs.get("extra_body") or {})


async def test_anthropic_effort_keeps_the_native_schema():
    provider = _anthropic({"output_config": {"effort": "low"}})
    provider._client.messages.create = AsyncMock(return_value=_text_response('{"answer": "a"}'))

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        await provider.call([{"role": "user", "content": "q"}], response_format=_Answer, strict_schema=True)

    output_config = _anthropic_wire(provider._client.messages.create.call_args.kwargs)["output_config"]
    assert output_config["effort"] == "low"
    assert output_config["format"]["type"] == "json_schema"
    assert "answer" in output_config["format"]["schema"]["properties"]


async def test_anthropic_gateway_fallback_drops_the_operator_output_config():
    """Once the endpoint refuses output_config, resending the operator's copy would refuse again."""
    provider = _anthropic({"output_config": {"effort": "low"}, "top_k": 5})
    response = httpx.Response(400, request=httpx.Request("POST", "https://gateway.example/v1/messages"))
    outcomes: list = [
        BadRequestError("output_config: Extra inputs are not permitted", response=response, body=None),
        _text_response('{"answer": "a"}'),
    ]
    requests: list[dict] = []

    async def create(**kwargs):
        requests.append(kwargs)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    provider._client.messages.create = create
    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        result = await provider.call(
            [{"role": "user", "content": "q"}], response_format=_Answer, strict_schema=True, max_retries=0
        )

    assert result.content.answer == "a"
    resend = _anthropic_wire(requests[1])
    assert "output_config" not in resend
    assert resend["top_k"] == 5


async def test_anthropic_batch_effort_keeps_the_native_schema():
    provider = _anthropic({"output_config": {"effort": "low"}})
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    request = BatchRequest(
        custom_id="c0",
        body={
            "messages": [{"role": "user", "content": "q"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": schema, "strict": True}},
        },
    )

    params = provider._translate_batch_body(request.body)

    assert params["output_config"]["effort"] == "low"
    assert params["output_config"]["format"]["schema"]["properties"] == schema["properties"]


async def test_anthropic_reply_with_a_code_block_inside_a_string_parses_whole():
    provider = _anthropic({})
    reply = '{"answer": "Example:\\n```json\\n[1, 2]\\n```"}'
    provider._client.messages.create = AsyncMock(return_value=_text_response(reply))

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        result = await provider.call([{"role": "user", "content": "q"}], response_format=_Answer)

    assert result.content.answer == "Example:\n```json\n[1, 2]\n```"


async def test_openai_responses_verbosity_keeps_the_schema_and_effort():
    llm = OpenAIResponsesLLM(
        provider="openai-responses",
        api_key="sk-test",
        base_url="",
        model="gpt-5.6",
        reasoning_effort="high",
        extra_body={"text": {"verbosity": "low"}, "reasoning": {"summary": "auto"}},
    )
    response = MagicMock()
    response.output_text = '{"answer": "a"}'
    response.output = []
    response.status = "completed"
    response.incomplete_details = None
    response.usage = MagicMock(input_tokens=1, output_tokens=1, input_tokens_details=None)
    create = AsyncMock(return_value=response)
    llm._client = MagicMock()
    llm._client.responses.create = create

    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call([{"role": "user", "content": "q"}], response_format=_Answer, strict_schema=True)

    kwargs = create.call_args.kwargs
    wire = openai_merge({k: v for k, v in kwargs.items() if k != "extra_body"}, kwargs["extra_body"])
    assert wire["text"]["verbosity"] == "low"
    assert wire["text"]["format"]["type"] == "json_schema"
    assert wire["reasoning"] == {"effort": "high", "summary": "auto"}


async def test_codex_verbosity_keeps_the_schema_and_effort():
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        llm = CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.6-luna",
            reasoning_effort="low",
            extra_body={"text": {"verbosity": "low"}, "reasoning": {"summary": "auto"}},
        )
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None

    with (
        stub_codex_stream(llm, response) as mock_stream,
        patch.object(
            llm,
            "_parse_sse_stream",
            new_callable=AsyncMock,
            return_value=_CodexStreamReply(text='{"answer": "a"}', usage=None),
        ),
    ):
        await llm.call([{"role": "user", "content": "q"}], response_format=_Answer, strict_schema=True, max_retries=0)

    body = mock_stream.call_args.kwargs["json"]
    assert body["text"]["verbosity"] == "low"
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["reasoning"]["summary"] == "auto"
    assert body["reasoning"]["effort"] == "low"
