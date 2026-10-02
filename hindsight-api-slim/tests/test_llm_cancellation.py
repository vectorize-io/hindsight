"""Cooperative cancellation at real wrapper/provider attempt boundaries."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from hindsight_api.cancellation import CancellationToken, OperationCancelledError
from hindsight_api.engine import llm_wrapper
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.response_models import LLMCallResult, LLMToolCallResult, TokenUsage


def _provider():
    return LLMProvider(
        provider="litellm",
        api_key="unused",
        base_url="http://localhost:0/v1",
        model="litellm_proxy/test-model",
        max_retries=3,
        initial_backoff=0,
        max_backoff=0,
    )


@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_cooperative_cancellation_is_not_a_retryable_provider_error(monkeypatch, method):
    provider = _provider()
    error = OperationCancelledError("request timeout")
    wire = AsyncMock(side_effect=error)
    monkeypatch.setattr(provider._provider_impl, "_acompletion", wire)
    kwargs = {"messages": []}
    if method == "call_with_tools":
        kwargs["tools"] = []
    with pytest.raises(OperationCancelledError) as raised:
        await getattr(provider, method)(**kwargs)
    assert raised.value is error
    wire.assert_awaited_once()
    await provider.cleanup()


@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_disconnect_during_queue_wait_releases_all_permits(monkeypatch, method):
    provider = _provider()
    token = CancellationToken()
    per_op = asyncio.Semaphore(1)
    global_sem = asyncio.Semaphore(0)
    entered_queue = asyncio.Event()

    class ObservedSemaphore:
        async def __aenter__(self):
            entered_queue.set()
            await global_sem.acquire()

        async def __aexit__(self, *args):
            global_sem.release()

    monkeypatch.setattr(llm_wrapper, "_global_llm_semaphore", ObservedSemaphore())
    monkeypatch.setattr(llm_wrapper, "_per_op_llm_semaphores", {"reflect": per_op})
    wire = AsyncMock()
    monkeypatch.setattr(provider._provider_impl, "_acompletion", wire)
    kwargs = {"messages": [], "scope": "reflect", "cancel_check": token.raise_if_cancelled}
    if method == "call_with_tools":
        kwargs["tools"] = []
    task = asyncio.create_task(getattr(provider, method)(**kwargs))
    await asyncio.wait_for(entered_queue.wait(), timeout=2)
    assert per_op.locked()
    token.cancel("client disconnected")
    global_sem.release()
    with pytest.raises(OperationCancelledError, match="client disconnected"):
        await asyncio.wait_for(task, timeout=2)
    wire.assert_not_awaited()
    assert not per_op.locked()
    assert not global_sem.locked()
    await provider.cleanup()


@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_non_attempt_gated_provider_checks_after_completion(monkeypatch, method):
    provider = LLMProvider(provider="mock", api_key="", base_url="", model="mock")
    token = CancellationToken()

    async def completed(**kwargs):
        token.cancel("client disconnected")
        if method == "call":
            return LLMCallResult(content="abandoned", usage=TokenUsage())
        return LLMToolCallResult(content="abandoned")

    monkeypatch.setattr(provider._provider_impl, method, completed)
    kwargs = {"messages": [], "cancel_check": token.raise_if_cancelled}
    if method == "call_with_tools":
        kwargs["tools"] = []
    with pytest.raises(OperationCancelledError, match="client disconnected"):
        await getattr(provider, method)(**kwargs)
    await provider.cleanup()


@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_disconnect_during_backoff_does_not_start_the_next_attempt(monkeypatch, method):
    provider = _provider()
    token = CancellationToken()
    wire = AsyncMock(side_effect=TimeoutError("completion timed out"))
    monkeypatch.setattr(provider._provider_impl, "_acompletion", wire)
    sleeps = []
    original_sleep = asyncio.sleep

    async def backoff(delay):
        sleeps.append(delay)
        token.cancel("client disconnected")
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", backoff)
    kwargs = {"messages": [], "cancel_check": token.raise_if_cancelled}
    if method == "call_with_tools":
        kwargs["tools"] = []
    with pytest.raises(OperationCancelledError, match="client disconnected"):
        await getattr(provider, method)(**kwargs)
    wire.assert_awaited_once()
    assert len(sleeps) == 1
    await provider.cleanup()


@pytest.mark.parametrize("provider_name", ["litellm", "openai", "anthropic", "gemini"])
@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_cancelled_wire_failure_is_terminal_across_attempt_gated_providers(monkeypatch, provider_name, method):
    token = CancellationToken()
    # Constructors create SDK clients, but only the stubbed wire method runs.
    provider = LLMProvider(
        provider=provider_name,
        api_key="unused",
        base_url="",
        model="test-model",
        max_retries=3,
        initial_backoff=0,
        max_backoff=0,
    )

    async def failed(**kwargs):
        token.cancel("client disconnected")
        raise TimeoutError("completion timed out")

    wire = AsyncMock(side_effect=failed)
    impl = provider._provider_impl
    if provider_name == "litellm":
        monkeypatch.setattr(impl, "_acompletion", wire)
    elif provider_name == "openai":
        monkeypatch.setattr(impl._client.chat.completions, "create", wire)
    elif provider_name == "anthropic":
        monkeypatch.setattr(impl._client.messages, "create", wire)
    else:
        monkeypatch.setattr(impl._client.aio.models, "generate_content", wire)
    kwargs = {"messages": [{"role": "user", "content": "test"}], "cancel_check": token.raise_if_cancelled}
    if method == "call_with_tools":
        kwargs["tools"] = [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}]
    with pytest.raises(OperationCancelledError, match="client disconnected"):
        await getattr(provider, method)(**kwargs)
    wire.assert_awaited_once()
    await provider.cleanup()
