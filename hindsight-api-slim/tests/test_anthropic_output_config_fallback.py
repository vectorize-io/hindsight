"""An endpoint that rejects ``output_config`` still gets structured output.

Strict structured output on Anthropic is ``output_config.format`` (native structured
output). A gateway in front of Claude that does not forward it answers with a 400; strict
calls then fall back to the schema in the prompt, once and for good, rather than paying the
400 on every call.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from anthropic import BadRequestError
from pydantic import BaseModel

from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

pytestmark = pytest.mark.asyncio


class _Decision(BaseModel):
    action: str


def _bad_request(message: str) -> BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://gateway.example/v1/messages"))
    return BadRequestError(message, response=response, body=None)


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
    with patch("anthropic.AsyncAnthropic") as client_cls:
        client_cls.return_value = MagicMock()
        provider = AnthropicLLM(provider="anthropic", api_key="k", base_url="", model="claude-sonnet-5-5")
    outcomes = [
        _bad_request("output_config: Extra inputs are not permitted"),
        _text_response('{"action": "keep"}'),
        _text_response('{"action": "skip"}'),
    ]
    requests: list[dict] = []

    async def create(**kwargs):
        requests.append(kwargs)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    provider._client = MagicMock()
    provider._client.messages.create = create

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        first = await provider.call(
            [{"role": "user", "content": "decide"}], response_format=_Decision, strict_schema=True, max_retries=0
        )
        second = await provider.call(
            [{"role": "user", "content": "decide"}], response_format=_Decision, strict_schema=True, max_retries=0
        )

    assert (first.content.action, second.content.action) == ("keep", "skip")
    assert "output_config" in requests[0]
    # The resend and every later call carry the schema in the prompt instead.
    for request in requests[1:]:
        assert "output_config" not in request and "tools" not in request
        assert "valid JSON matching this schema" in str(request["system"])
