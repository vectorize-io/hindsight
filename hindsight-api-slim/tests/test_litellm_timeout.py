"""
Regression test for the hard timeout on the LiteLLM provider.

A completion that never returns — a connection held open with no token
progress, or one straggler inside a concurrent ``asyncio.gather`` fan-out —
must not block forever. ``call`` / ``call_with_tools`` wrap the request in
``asyncio.wait_for`` so it is cancelled after ``timeout`` seconds and surfaced
as a retryable ``TimeoutError`` instead of pinning a worker slot and a
concurrency permit indefinitely.
"""

import asyncio
import time

import pytest

from hindsight_api.config import DEFAULT_LLM_TIMEOUT, ENV_LLM_TIMEOUT
from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM


def _make_provider(timeout: float | None) -> LiteLLMLLM:
    return LiteLLMLLM(
        provider="litellm",
        api_key="unused",
        base_url="http://localhost:0/v1",
        model="litellm_proxy/test-model",
        timeout=timeout,
    )


async def test_call_cancels_hung_completion(monkeypatch):
    """A hung ``_acompletion`` is cancelled per attempt and raises TimeoutError."""
    provider = _make_provider(timeout=0.1)
    calls = 0

    async def _hang(**kwargs):
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()  # never resolves

    monkeypatch.setattr(provider, "_acompletion", _hang)

    started = time.monotonic()
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=1,
            initial_backoff=0.01,
            max_backoff=0.01,
        )
    elapsed = time.monotonic() - started

    # max_retries=1 -> attempts 0 and 1, each bounded by the timeout.
    assert calls == 2
    # Bounded by ~2 * timeout + backoff — nowhere near hanging forever.
    assert elapsed < 2.0


async def test_call_with_tools_cancels_hung_completion(monkeypatch):
    provider = _make_provider(timeout=0.1)

    async def _hang(**kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(provider, "_acompletion", _hang)

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            max_retries=0,
            initial_backoff=0.01,
            max_backoff=0.01,
        )


async def test_unset_timeout_falls_back_to_default(monkeypatch):
    """``None`` must resolve to a finite default — never ``None``, which would
    make ``asyncio.wait_for`` wait forever and reintroduce the hang."""
    monkeypatch.delenv(ENV_LLM_TIMEOUT, raising=False)
    provider = _make_provider(timeout=None)
    assert provider.timeout == DEFAULT_LLM_TIMEOUT
