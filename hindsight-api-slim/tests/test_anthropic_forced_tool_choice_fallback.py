"""A model that rejects a forced tool_choice still gets its tool calls (#5738).

Claude Opus 5.5 and Sonnet 5.5 answer a named or ``any`` tool_choice with a 400. Reflect forces
tools on its first steps and its closing ``done`` call, so without a fallback every reflect and
mental-model refresh failed on those models; with HINDSIGHT_API_LLM_STRICT_SCHEMA on, every
structured call (retain, consolidation) failed the same way.
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import httpx
import pytest
from anthropic import BadRequestError
from pydantic import BaseModel

from hindsight_api.engine.llm_interface import LLMToolChoice, LLMToolChoiceMode
from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

pytestmark = pytest.mark.asyncio

_TOOLS = [{"type": "function", "function": {"name": "done", "description": "d", "parameters": {"type": "object"}}}]
_NAMED = LLMToolChoice(mode=LLMToolChoiceMode.NAMED, function_name="done")
_REQUIRED = LLMToolChoice(mode=LLMToolChoiceMode.REQUIRED)


def _bad_request(message: str) -> BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return BadRequestError(message, response=response, body=None)


def _response():
    block = MagicMock()
    block.type = "tool_use"
    block.id = "t1"
    block.name = "done"
    block.input = {"answer": "A"}
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(input_tokens=10, output_tokens=2, cache_read_input_tokens=0)
    response.stop_reason = "tool_use"
    return response


@dataclass
class _Fake:
    provider: AnthropicLLM
    sent: list[dict] = field(default_factory=list)  # the tool_choice of each request, in order


def _provider(responses: list) -> _Fake:
    with patch("anthropic.AsyncAnthropic") as client_cls:
        client_cls.return_value = MagicMock()
        provider = AnthropicLLM(provider="anthropic", api_key="fake-key", base_url="", model="claude-opus-5-5")
    fake = _Fake(provider)

    async def create(**kwargs):
        fake.sent.append(dict(kwargs["tool_choice"]) if "tool_choice" in kwargs else {})
        outcome = responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    provider._client = MagicMock()
    provider._client.messages.create = create
    return fake


@pytest.mark.parametrize("choice", [_NAMED, _REQUIRED])
async def test_rejected_forced_choice_is_resent_as_auto_and_remembered(choice):
    rejection = _bad_request('tool_choice: type "tool" and "any" are not supported for this model.')
    fake = _provider([rejection, _response(), _response()])
    provider, sent = fake.provider, fake.sent

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        first = await provider.call_with_tools([{"role": "user", "content": "q"}], _TOOLS, tool_choice=choice)
        second = await provider.call_with_tools([{"role": "user", "content": "q"}], _TOOLS, tool_choice=choice)

    assert [tc.name for tc in first.tool_calls] == ["done"]
    assert [tc.name for tc in second.tool_calls] == ["done"]
    # The rejected request, its resend, then the next call asks with auto straight away.
    assert [c["type"] for c in sent] == [sent[0]["type"], "auto", "auto"]
    assert sent[0]["type"] in ("tool", "any")


async def test_other_bad_requests_are_not_masked():
    fake = _provider([_bad_request("messages: roles must alternate")])
    provider, sent = fake.provider, fake.sent

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        with pytest.raises(BadRequestError):
            await provider.call_with_tools(
                [{"role": "user", "content": "q"}], _TOOLS, tool_choice=_NAMED, max_retries=0
            )

    assert [c["type"] for c in sent] == ["tool"]


class _Decision(BaseModel):
    action: str


def _text_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(input_tokens=5, output_tokens=2, cache_read_input_tokens=0)
    response.stop_reason = "end_turn"
    return response


async def test_strict_structured_output_falls_back_to_schema_in_prompt():
    """With HINDSIGHT_API_LLM_STRICT_SCHEMA on, structured calls force a tool too; they fall back."""
    rejection = _bad_request('tool_choice: type "tool" and "any" are not supported for this model.')
    fake = _provider([rejection, _text_response('{"action": "keep"}'), _text_response('{"action": "skip"}')])
    requests: list[dict] = []
    create = fake.provider._client.messages.create

    async def capture(**kwargs):
        requests.append(kwargs)
        return await create(**kwargs)

    fake.provider._client.messages.create = capture

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        first = await fake.provider.call(
            [{"role": "user", "content": "decide"}], response_format=_Decision, strict_schema=True, max_retries=0
        )
        second = await fake.provider.call(
            [{"role": "user", "content": "decide"}], response_format=_Decision, strict_schema=True, max_retries=0
        )

    assert (first.content.action, second.content.action) == ("keep", "skip")
    assert requests[0]["tool_choice"]["type"] == "tool"
    # The resend and every later call carry the schema in the prompt and force nothing.
    for request in requests[1:]:
        assert "tools" not in request and "tool_choice" not in request
        assert "valid JSON matching this schema" in str(request["system"])
