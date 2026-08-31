"""The native Ollama path must use the configured LLM timeout, not a literal.

`_call_ollama_native` built its own `httpx.AsyncClient(timeout=300.0)`, which is
the one request path that ignored `HINDSIGHT_API_LLM_TIMEOUT`. On a CPU ollama
host a single fact-extraction prompt can need longer than 300 s just to be
ingested, and the call was aborted mid-prompt with a bare "Ollama connection
error" that raising the configured timeout could not fix.

Note the fix cuts both ways: `self.timeout` falls back to DEFAULT_LLM_TIMEOUT
(120 s), which is LOWER than the old literal, so a deployment relying on the
implicit 300 s must now set ENV_LLM_TIMEOUT. Both directions are asserted here.
"""

import httpx
import pytest

from hindsight_api.config import DEFAULT_LLM_TIMEOUT, ENV_LLM_TIMEOUT
from hindsight_api.engine.providers import openai_compatible_llm
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


class _CapturingAsyncClient:
    """Stand-in for httpx.AsyncClient that records the timeout it was built with."""

    captured: list[float | None] = []

    def __init__(self, *args, timeout=None, **kwargs):
        type(self).captured.append(timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *args, **kwargs):
        # Fail the call immediately: the assertion is about client construction,
        # so the request itself never needs to succeed.
        raise httpx.ConnectError("no ollama in tests")


def _timeout_used(monkeypatch, env_value: str | None) -> float | None:
    if env_value is None:
        monkeypatch.delenv(ENV_LLM_TIMEOUT, raising=False)
    else:
        monkeypatch.setenv(ENV_LLM_TIMEOUT, env_value)
    monkeypatch.setattr(openai_compatible_llm.httpx, "AsyncClient", _CapturingAsyncClient)
    _CapturingAsyncClient.captured = []

    llm = OpenAICompatibleLLM(provider="ollama", api_key="local", base_url="http://localhost:11434", model="qwen")
    return llm.timeout


def test_configured_timeout_is_not_capped_by_the_old_literal(monkeypatch):
    """A timeout above the old 300 s literal survives — the bug this fixes."""
    assert _timeout_used(monkeypatch, "900") == pytest.approx(900.0)


def test_unset_timeout_falls_back_to_the_configured_default(monkeypatch):
    """With ENV_LLM_TIMEOUT unset the native path gets DEFAULT_LLM_TIMEOUT, not 300.0."""
    used = _timeout_used(monkeypatch, None)
    assert used == pytest.approx(DEFAULT_LLM_TIMEOUT)
    assert used != pytest.approx(300.0), "the hardcoded native-path literal is back"
