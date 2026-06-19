"""Integration: MemoryEngine constructs FailoverLLMProvider when configured."""

import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Use mock LLM for both primary and failover; skip verification."""
    from hindsight_api.config import clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_SKIP_LLM_VERIFICATION", "true")
    monkeypatch.setenv("HINDSIGHT_API_LAZY_RERANKER", "true")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "primary-model")
    clear_config_cache()
    yield
    clear_config_cache()


def test_memory_engine_uses_bare_llmprovider_when_failover_unset(monkeypatch):
    """Backwards-compat: no failover env => no FailoverLLMProvider wrapper."""
    monkeypatch.delenv("HINDSIGHT_API_LLM_FAILOVER_PROVIDER", raising=False)
    from hindsight_api import MemoryEngine
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.failover_llm import FailoverLLMProvider
    from hindsight_api.engine.llm_wrapper import LLMProvider

    clear_config_cache()
    engine = MemoryEngine(skip_llm_verification=True, lazy_reranker=True)

    # Hot path: must be a bare LLMProvider, not a FailoverLLMProvider
    assert isinstance(engine._llm_config, LLMProvider)
    assert not isinstance(engine._llm_config, FailoverLLMProvider)
    assert isinstance(engine._retain_llm_config, LLMProvider)
    assert not isinstance(engine._retain_llm_config, FailoverLLMProvider)


def test_memory_engine_wraps_with_failover_when_set(monkeypatch):
    """When HINDSIGHT_API_LLM_FAILOVER_PROVIDER is set, all four LLM configs are wrapped."""
    monkeypatch.setenv("HINDSIGHT_API_LLM_FAILOVER_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_LLM_FAILOVER_MODEL", "failover-model")
    from hindsight_api import MemoryEngine
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.failover_llm import FailoverLLMProvider

    clear_config_cache()
    engine = MemoryEngine(skip_llm_verification=True, lazy_reranker=True)

    for attr in ("_llm_config", "_retain_llm_config", "_reflect_llm_config", "_consolidation_llm_config"):
        wrapper = getattr(engine, attr)
        assert isinstance(wrapper, FailoverLLMProvider), f"{attr} should be wrapped"
        assert wrapper._primary.model == "primary-model"
        assert wrapper._failover is not None
        assert wrapper._failover.model == "failover-model"


@pytest.mark.asyncio
async def test_memory_engine_call_falls_over_when_primary_raises(monkeypatch):
    """End-to-end: a call on _retain_llm_config falls over correctly."""
    monkeypatch.setenv("HINDSIGHT_API_LLM_FAILOVER_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_LLM_FAILOVER_MODEL", "failover-model")
    from hindsight_api import MemoryEngine
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.providers.mock_llm import MockLLM

    clear_config_cache()
    engine = MemoryEngine(skip_llm_verification=True, lazy_reranker=True)

    # Force the primary inside the composite to raise
    primary = engine._retain_llm_config._primary  # type: ignore[attr-defined]
    failover = engine._retain_llm_config._failover  # type: ignore[attr-defined]
    assert isinstance(primary._provider_impl, MockLLM)
    primary._provider_impl._mock_exception = RuntimeError("upstream 503")
    failover.set_mock_response("failover ok")

    result = await engine._retain_llm_config.call(messages=[{"role": "user", "content": "hi"}])
    assert result == "failover ok"
