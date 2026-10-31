"""Issue #5439: HINDSIGHT_API_LLM_OPENAI_SERVICE_TIER must reach chat completions.

The tier is a native top-level ``service_tier`` param, sent only for provider=openai,
on ``call()``.
"""

import types
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


def _response() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                finish_reason="stop",
                message=types.SimpleNamespace(content="ok", tool_calls=None, refusal=None),
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="gpt-4o-mini",
    )


def _make_llm(provider: str) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider=provider,
        api_key="sk-test",
        base_url="http://localhost:1234/v1" if provider == "lmstudio" else "",
        model="gpt-4o-mini",
        openai_service_tier="flex",
    )


async def _sent_params(llm: OpenAICompatibleLLM) -> dict:
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _response()
        await llm.call(messages=[{"role": "user", "content": "go"}], max_retries=0)
        return create.call_args.kwargs


@pytest.mark.asyncio
async def test_openai_sends_service_tier():
    params = await _sent_params(_make_llm("openai"))
    assert params["service_tier"] == "flex"
    assert "service_tier" not in params.get("extra_body", {})


@pytest.mark.asyncio
async def test_other_providers_do_not_send_openai_service_tier():
    params = await _sent_params(_make_llm("lmstudio"))
    assert "service_tier" not in params
