"""The native Ollama path must use the configured LLM timeout, not a literal.

`_call_ollama_native` used to build its own HTTP client with `timeout=300.0`, the
one request path that ignored `HINDSIGHT_API_LLM_TIMEOUT`. On a CPU ollama host a
single fact-extraction prompt can need longer than 300 s just to be ingested, and
the call was aborted mid-prompt with a bare "Ollama connection error" that raising
the configured timeout could not fix.

Note the fix cuts both ways: `self.timeout` falls back to DEFAULT_LLM_TIMEOUT
(120 s), which is LOWER than the old literal, so a deployment relying on the
implicit 300 s must now set ENV_LLM_TIMEOUT. Both directions are asserted here.
"""

import aiohttp
import pytest

from hindsight_api.config import DEFAULT_LLM_TIMEOUT, ENV_LLM_TIMEOUT, clear_config_cache
from hindsight_api.engine.providers import openai_compatible_llm
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


class _CapturingSession:
    """Stand-in for LoopLocalSession that records the timeout it was built with."""

    captured: list[aiohttp.ClientTimeout] = []

    def __init__(self, *, timeout: aiohttp.ClientTimeout, **kwargs):
        type(self).captured.append(timeout)


def _timeout_used(monkeypatch, env_value: str | None) -> float | None:
    """The per-read timeout of the session the native path sends through."""
    if env_value is None:
        monkeypatch.delenv(ENV_LLM_TIMEOUT, raising=False)
    else:
        monkeypatch.setenv(ENV_LLM_TIMEOUT, env_value)
    clear_config_cache()
    monkeypatch.setattr(openai_compatible_llm, "LoopLocalSession", _CapturingSession)
    _CapturingSession.captured = []

    try:
        OpenAICompatibleLLM(provider="ollama", api_key="local", base_url="http://localhost:11434", model="qwen")
    finally:
        clear_config_cache()
    assert len(_CapturingSession.captured) == 1
    return _CapturingSession.captured[0].sock_read


def test_configured_timeout_is_not_capped_by_the_old_literal(monkeypatch):
    """A timeout above the old 300 s literal survives — the bug this fixes."""
    assert _timeout_used(monkeypatch, "900") == pytest.approx(900.0)


def test_unset_timeout_falls_back_to_the_configured_default(monkeypatch):
    """With ENV_LLM_TIMEOUT unset the native path gets DEFAULT_LLM_TIMEOUT, not 300.0."""
    used = _timeout_used(monkeypatch, None)
    assert used == pytest.approx(DEFAULT_LLM_TIMEOUT)
    assert used != pytest.approx(300.0), "the hardcoded native-path literal is back"
