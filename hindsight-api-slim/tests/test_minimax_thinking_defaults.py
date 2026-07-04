from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


@pytest.fixture(autouse=True)
def clear_proxy_env(monkeypatch):
    for key in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "http_proxy", "https_proxy"):
        monkeypatch.delenv(key, raising=False)


def _make_minimax_llm(model: str = "MiniMax-M3", extra_body: dict | None = None) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="minimax",
        api_key="test-key",
        base_url=None,
        model=model,
        extra_body=extra_body,
    )


def _chat_response(content: str = "ok"):
    choice = SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content=content, tool_calls=None, refusal=None),
    )
    return SimpleNamespace(choices=[choice], usage=None, error=None)


def _tool_response():
    choice = SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content="done", tool_calls=None, refusal=None, reasoning_content=None),
    )
    return SimpleNamespace(choices=[choice], usage=None, error=None)


async def _capture_call_kwargs(llm: OpenAICompatibleLLM) -> dict:
    create = AsyncMock(return_value=_chat_response())
    llm._client.chat.completions.create = create
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.call(
            messages=[{"role": "user", "content": "ping"}],
            max_retries=0,
        )
    return create.call_args.kwargs


async def _capture_tool_call_kwargs(llm: OpenAICompatibleLLM) -> dict:
    create = AsyncMock(return_value=_tool_response())
    llm._client.chat.completions.create = create
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "ping"}],
            tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
            max_retries=0,
        )
    return create.call_args.kwargs


@pytest.mark.asyncio
async def test_minimax_m3_disables_thinking_by_default():
    sent = await _capture_call_kwargs(_make_minimax_llm())

    assert sent["extra_body"]["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_minimax_m3_preserves_explicit_thinking_config():
    sent = await _capture_call_kwargs(_make_minimax_llm(extra_body={"thinking": {"type": "adaptive"}}))

    assert sent["extra_body"]["thinking"] == {"type": "adaptive"}


@pytest.mark.asyncio
async def test_minimax_m3_tool_calls_disable_thinking_by_default():
    sent = await _capture_tool_call_kwargs(_make_minimax_llm())

    assert sent["extra_body"]["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_minimax_m2_does_not_claim_thinking_can_be_disabled():
    sent = await _capture_call_kwargs(_make_minimax_llm(model="MiniMax-M2.7"))

    assert "extra_body" not in sent
