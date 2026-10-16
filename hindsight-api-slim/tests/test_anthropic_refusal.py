"""Issue #5201: an Anthropic refusal (HTTP 200, stop_reason refusal/content_filter)
must raise ProviderContentPolicyError after one call, not retry as invalid JSON."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.llm_interface import ProviderContentPolicyError


class _Answer(BaseModel):
    text: str


def _make_provider():
    with patch("anthropic.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

        provider = AnthropicLLM(provider="anthropic", api_key="fake-key", base_url="", model="claude-sonnet-4-20250514")
    provider._client = MagicMock()
    return provider


def _refusal(stop_reason: str):
    resp = MagicMock()
    resp.content = []
    resp.usage = MagicMock(input_tokens=5, output_tokens=0, cache_read_input_tokens=0)
    resp.stop_reason = stop_reason
    return resp


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["refusal", "content_filter"])
async def test_call_refusal_is_not_retried(stop_reason):
    provider = _make_provider()
    provider._client.messages.create = AsyncMock(return_value=_refusal(stop_reason))
    with pytest.raises(ProviderContentPolicyError):
        await provider.call(
            messages=[{"role": "user", "content": "x"}],
            response_format=_Answer,
            scope="test",
            max_retries=3,
            initial_backoff=0,
        )
    assert provider._client.messages.create.await_count == 1


@pytest.mark.asyncio
async def test_call_with_tools_refusal_is_not_retried():
    provider = _make_provider()
    provider._client.messages.create = AsyncMock(return_value=_refusal("refusal"))
    with pytest.raises(ProviderContentPolicyError):
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            max_retries=3,
            initial_backoff=0,
        )
    assert provider._client.messages.create.await_count == 1
