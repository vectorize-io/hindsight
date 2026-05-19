"""Tests for the opencode-go OpenAI-compatible LLM provider."""

import pytest


def test_opencode_go_config_has_expected_default_model(monkeypatch):
    """HindsightConfig should default opencode-go to the DeepSeek v4 flash model."""
    from hindsight_api.config import PROVIDER_DEFAULT_MODELS, HindsightConfig, clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "opencode-go")
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL", raising=False)
    clear_config_cache()

    try:
        assert PROVIDER_DEFAULT_MODELS["opencode-go"] == "deepseek-v4-flash"
        config = HindsightConfig.from_env()
        assert config.llm_provider == "opencode-go"
        assert config.llm_model == "deepseek-v4-flash"
    finally:
        clear_config_cache()


def test_opencode_go_llm_provider_from_env_has_expected_default_model(monkeypatch):
    """LLMProvider.from_env should use the opencode-go provider default model."""
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.llm_wrapper import LLMProvider

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "opencode-go")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "test-key")
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_LLM_BASE_URL", raising=False)
    clear_config_cache()

    try:
        llm = LLMProvider.from_env()
        assert llm.provider == "opencode-go"
        assert llm.model == "deepseek-v4-flash"
        assert llm.base_url == "https://opencode.ai/zen/go/v1"
    finally:
        clear_config_cache()


def test_opencode_go_requires_api_key_like_zai():
    """opencode-go is a cloud provider and should require an API key."""
    from hindsight_api.engine.llm_wrapper import requires_api_key

    assert requires_api_key("opencode-go") is True


def test_opencode_go_uses_openai_compatible_provider_with_default_base_url():
    """The provider factory should route opencode-go to OpenAICompatibleLLM."""
    from hindsight_api.engine.llm_wrapper import LLMProvider
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    llm = LLMProvider(
        provider="opencode-go",
        api_key="test-key",
        base_url="",
        model="deepseek-v4-flash",
    )

    assert llm.provider == "opencode-go"
    assert llm.model == "deepseek-v4-flash"
    assert llm.base_url == "https://opencode.ai/zen/go/v1"
    assert not llm.base_url.endswith("/")
    assert isinstance(llm._provider_impl, OpenAICompatibleLLM)
    assert llm._provider_impl.base_url == "https://opencode.ai/zen/go/v1"


def test_opencode_go_rejects_missing_api_key():
    """opencode-go should fail fast without an API key, matching zai behavior."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    with pytest.raises(ValueError, match="API key is required for opencode-go"):
        LLMProvider(
            provider="opencode-go",
            api_key="",
            base_url="",
            model="deepseek-v4-flash",
        )
