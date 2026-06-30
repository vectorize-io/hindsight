"""Plumbing tests for the per-operation / global LLM request defaults (issue #2452).

These assert the *wiring* — that a resolved timeout / retry policy actually reaches
the provider that uses it — not just that the env var parses into config (covered by
test_config_validation.py).

The values are threaded
``config -> MemoryEngine per-op resolve -> LLMProvider -> (create_llm_provider /
call())``. Before the fix the per-operation ``*_llm_timeout`` / ``*_llm_max_retries`` /
``*_llm_initial_backoff`` / ``*_llm_max_backoff`` fields (and even the global ``llm_*``)
were resolved into ``HindsightConfig`` but never reached the provider, so a configured
``HINDSIGHT_API_RETAIN_LLM_TIMEOUT`` silently used the global default
(``LiteLLM call exceeded timeout=120.0s``) and the per-op retry knobs were inert.
"""

import pytest

from hindsight_api.config import DEFAULT_LLM_TIMEOUT
from hindsight_api.engine.llm_wrapper import LLMConfig


def _mock_llm(**kwargs) -> LLMConfig:
    return LLMConfig(provider="mock", api_key="", base_url="", model="m", **kwargs)


def _spy_provider_call(monkeypatch, llm: LLMConfig) -> dict:
    """Replace the provider impl's call() with a kwargs-capturing stub."""
    captured: dict = {}

    async def fake_call(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(llm._provider_impl, "call", fake_call)
    return captured


def test_litellm_provider_impl_receives_timeout():
    """LLMConfig -> create_llm_provider -> LiteLLMLLM carries the resolved timeout."""
    llm = LLMConfig(provider="litellm", api_key="k", base_url="", model="gpt-4o-mini", timeout=300.0)
    assert llm.timeout == 300.0
    assert llm._provider_impl.timeout == 300.0


def test_openai_compatible_provider_impl_receives_timeout():
    """The OpenAI-compatible path (openai/groq/ollama/...) carries the timeout too."""
    llm = LLMConfig(provider="openai", api_key="k", base_url="", model="gpt-4o-mini", timeout=250.0)
    assert llm._provider_impl.timeout == 250.0


def test_timeout_none_falls_back_to_provider_default():
    """No timeout passed -> provider falls back to its env/DEFAULT_LLM_TIMEOUT default.

    Guards against a regression where threading the value would override the
    long-standing default for callers that never configured a timeout.
    """
    llm = LLMConfig(provider="litellm", api_key="k", base_url="", model="gpt-4o-mini")
    assert llm._provider_impl.timeout == DEFAULT_LLM_TIMEOUT


@pytest.fixture
def _clean_timeout_env(monkeypatch):
    """Mock provider + verification off, with all timeout env vars cleared."""
    from hindsight_api.config import clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_SKIP_LLM_VERIFICATION", "true")
    monkeypatch.setenv("HINDSIGHT_API_LAZY_RERANKER", "true")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "default-model")
    for op in ("LLM", "RETAIN_LLM", "REFLECT_LLM", "CONSOLIDATION_LLM"):
        for knob in ("TIMEOUT", "MAX_RETRIES", "INITIAL_BACKOFF", "MAX_BACKOFF"):
            monkeypatch.delenv(f"HINDSIGHT_API_{op}_{knob}", raising=False)
    clear_config_cache()
    yield
    clear_config_cache()


async def test_call_uses_instance_retry_defaults(monkeypatch):
    """call() falls back to the provider's configured retry policy when no per-call
    arg is given — this is what makes a per-op ``*_llm_max_retries`` take effect."""
    llm = _mock_llm(max_retries=7, initial_backoff=2.0, max_backoff=9.0)
    captured = _spy_provider_call(monkeypatch, llm)

    await llm.call(messages=[{"role": "user", "content": "hi"}], scope="x")

    assert captured["max_retries"] == 7
    assert captured["initial_backoff"] == 2.0
    assert captured["max_backoff"] == 9.0


async def test_call_explicit_arg_overrides_instance_default(monkeypatch):
    """An explicit per-call value still wins over the configured default."""
    llm = _mock_llm(max_retries=7)
    captured = _spy_provider_call(monkeypatch, llm)

    await llm.call(messages=[{"role": "user", "content": "hi"}], scope="x", max_retries=2)

    assert captured["max_retries"] == 2


async def test_call_falls_back_to_method_default_when_unconfigured(monkeypatch):
    """No instance config and no per-call arg -> the method's own fallback (10),
    so providers built outside MemoryEngine (from_env, tests) are unchanged."""
    llm = _mock_llm()
    captured = _spy_provider_call(monkeypatch, llm)

    await llm.call(messages=[{"role": "user", "content": "hi"}], scope="x")

    assert captured["max_retries"] == 10
    assert captured["initial_backoff"] == 1.0
    assert captured["max_backoff"] == 60.0


def test_memory_engine_threads_per_operation_timeout(monkeypatch, _clean_timeout_env):
    """Each per-operation override reaches its own LLM config; the rest fall back
    to the global ``llm_timeout``."""
    from hindsight_api import MemoryEngine
    from hindsight_api.config import clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_LLM_TIMEOUT", "100")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_LLM_TIMEOUT", "300")
    monkeypatch.setenv("HINDSIGHT_API_CONSOLIDATION_LLM_TIMEOUT", "450")
    # reflect intentionally unset -> inherits the global 100
    clear_config_cache()

    engine = MemoryEngine(skip_llm_verification=True, lazy_reranker=True)

    assert engine._llm_config.timeout == 100.0
    assert engine._retain_llm_config.timeout == 300.0
    assert engine._reflect_llm_config.timeout == 100.0
    assert engine._consolidation_llm_config.timeout == 450.0


def test_memory_engine_threads_per_operation_retry_policy(monkeypatch, _clean_timeout_env):
    """Per-op retry/backoff overrides reach their own config; unset ops fall back
    to the global ``llm_max_retries`` / ``llm_initial_backoff`` / ``llm_max_backoff``."""
    from hindsight_api import MemoryEngine
    from hindsight_api.config import clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_LLM_MAX_RETRIES", "4")
    monkeypatch.setenv("HINDSIGHT_API_LLM_INITIAL_BACKOFF", "0.5")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MAX_BACKOFF", "20")
    monkeypatch.setenv("HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("HINDSIGHT_API_CONSOLIDATION_LLM_MAX_BACKOFF", "99")
    clear_config_cache()

    engine = MemoryEngine(skip_llm_verification=True, lazy_reranker=True)

    # Global applies everywhere unless overridden.
    assert engine._llm_config.max_retries == 4
    assert engine._retain_llm_config.max_retries == 4
    # reflect overrides only max_retries; backoff inherits global.
    assert engine._reflect_llm_config.max_retries == 2
    assert engine._reflect_llm_config.initial_backoff == 0.5
    # consolidation overrides only max_backoff; retries inherit global.
    assert engine._consolidation_llm_config.max_retries == 4
    assert engine._consolidation_llm_config.max_backoff == 99.0


def test_memory_engine_per_op_defaults_to_global_default(_clean_timeout_env):
    """With nothing configured, every operation uses the documented global defaults."""
    from hindsight_api import MemoryEngine
    from hindsight_api.config import (
        DEFAULT_LLM_INITIAL_BACKOFF,
        DEFAULT_LLM_MAX_BACKOFF,
        DEFAULT_LLM_MAX_RETRIES,
    )

    engine = MemoryEngine(skip_llm_verification=True, lazy_reranker=True)

    for cfg in (
        engine._llm_config,
        engine._retain_llm_config,
        engine._reflect_llm_config,
        engine._consolidation_llm_config,
    ):
        assert cfg.timeout == DEFAULT_LLM_TIMEOUT
        assert cfg.max_retries == DEFAULT_LLM_MAX_RETRIES
        assert cfg.initial_backoff == DEFAULT_LLM_INITIAL_BACKOFF
        assert cfg.max_backoff == DEFAULT_LLM_MAX_BACKOFF
