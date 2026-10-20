"""A Gemini content block fails once, without retrying (#5821).

A block (``PROHIBITED_CONTENT``, ``SAFETY``, ...) is a function of the prompt:
every retry earns the same block. The provider must raise the permanent
``ProviderContentPolicyError`` on the first attempt, naming the reason, while an
empty response with no reason keeps retrying as before.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("google.genai")

from google.genai import types as genai_types  # noqa: E402

from hindsight_api.engine.llm_interface import ProviderContentPolicyError  # noqa: E402


def _make_gemini_provider():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.gemini_llm import GeminiLLM

        provider = GeminiLLM(provider="gemini", api_key="fake-api-key", base_url="", model="gemini-2.5-flash")
        provider._client = MagicMock()
        return provider


def _empty_response(finish_reason=None, prompt_block_reason=None):
    return SimpleNamespace(
        text=None,
        candidates=[SimpleNamespace(finish_reason=finish_reason, content=None)],
        prompt_feedback=SimpleNamespace(block_reason=prompt_block_reason),
        usage_metadata=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        _empty_response(finish_reason=genai_types.FinishReason.PROHIBITED_CONTENT),
        _empty_response(prompt_block_reason=genai_types.BlockedReason.PROHIBITED_CONTENT),
    ],
)
async def test_call_content_block_fails_once(response):
    provider = _make_gemini_provider()
    generate = AsyncMock(return_value=response)
    provider._client.aio.models.generate_content = generate

    with pytest.raises(ProviderContentPolicyError, match="PROHIBITED_CONTENT"):
        await provider.call(messages=[{"role": "user", "content": "hi"}], max_retries=3, initial_backoff=0.0)

    assert generate.call_count == 1


@pytest.mark.asyncio
async def test_call_with_tools_content_block_fails_once():
    provider = _make_gemini_provider()
    generate = AsyncMock(return_value=_empty_response(finish_reason=genai_types.FinishReason.SAFETY))
    provider._client.aio.models.generate_content = generate

    with pytest.raises(ProviderContentPolicyError, match="SAFETY"):
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}}],
            max_retries=3,
            initial_backoff=0.0,
        )

    assert generate.call_count == 1


@pytest.mark.asyncio
async def test_call_empty_response_without_reason_still_retries():
    provider = _make_gemini_provider()
    generate = AsyncMock(return_value=_empty_response())
    provider._client.aio.models.generate_content = generate

    with pytest.raises(RuntimeError, match="empty response after 4 attempts") as excinfo:
        await provider.call(messages=[{"role": "user", "content": "hi"}], max_retries=3, initial_backoff=0.0)

    assert not isinstance(excinfo.value, ProviderContentPolicyError)
    assert generate.call_count == 4
