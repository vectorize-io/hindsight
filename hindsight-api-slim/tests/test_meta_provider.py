"""Tests for the Meta Model API OpenAI-compatible LLM provider."""

import pytest


def test_meta_config_has_expected_default_model(monkeypatch):
    """HindsightConfig should default meta to Muse Spark 1.3."""
    from hindsight_api.config import PROVIDER_DEFAULT_MODELS, HindsightConfig, clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "meta")
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL", raising=False)
    clear_config_cache()

    try:
        assert PROVIDER_DEFAULT_MODELS["meta"] == "muse-spark-1.3"
        config = HindsightConfig.from_env()
        assert config.llm_provider == "meta"
        assert config.llm_model == "muse-spark-1.3"
    finally:
        clear_config_cache()


def test_meta_llm_provider_from_env_has_expected_default_model(monkeypatch):
    """LLMProvider.from_env should use the meta provider default model and base URL."""
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.llm_wrapper import LLMProvider

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "meta")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "test-key")
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_LLM_BASE_URL", raising=False)
    clear_config_cache()

    try:
        llm = LLMProvider.from_env()
        assert llm.provider == "meta"
        assert llm.model == "muse-spark-1.3"
        assert llm.base_url == "https://api.meta.ai/v1"
    finally:
        clear_config_cache()


def test_meta_requires_api_key():
    """Meta Model API is a cloud provider and should require an API key."""
    from hindsight_api.engine.llm_wrapper import requires_api_key

    assert requires_api_key("meta") is True


def test_meta_uses_openai_compatible_provider_with_default_base_url():
    """The provider factory should route meta to OpenAICompatibleLLM."""
    from hindsight_api.engine.llm_wrapper import LLMProvider
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    llm = LLMProvider(
        provider="meta",
        api_key="test-key",
        base_url="",
        model="muse-spark-1.3",
    )

    assert llm.provider == "meta"
    assert llm.model == "muse-spark-1.3"
    assert llm.base_url == "https://api.meta.ai/v1"
    assert not llm.base_url.endswith("/")
    assert isinstance(llm._provider_impl, OpenAICompatibleLLM)
    assert llm._provider_impl.base_url == "https://api.meta.ai/v1"


def test_meta_rejects_missing_api_key():
    """meta should fail fast without an API key, matching the other cloud providers."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    with pytest.raises(ValueError, match="API key is required for meta"):
        LLMProvider(
            provider="meta",
            api_key="",
            base_url="",
            model="muse-spark-1.3",
        )


def test_meta_uses_max_tokens_not_max_completion_tokens():
    """Meta's chat/completions endpoint documents ``max_tokens``, not the newer name.

    Muse Spark is a reasoning model, but it is not one of the OpenAI products the
    frozen ``_supports_reasoning_model`` list recognises, so the parameter name must
    come from the provider default rather than that name check.

    Verified live: Meta accepts ``max_completion_tokens`` too, so this is a choice
    between two working names rather than a correctness fix — we send the one the
    docs specify.
    """
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(
        provider="meta",
        api_key="test-key",
        base_url="",
        model="muse-spark-1.3",
    )

    assert llm._max_tokens_param_name() == "max_tokens"


def test_meta_sends_configured_reasoning_effort():
    """Muse Spark always reasons, so an operator-set effort must reach the request.

    ``reasoning_effort`` is only dropped for the OpenAI products that reject it
    outright (#3449); ``muse-spark-*`` is not one of them.
    """
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(
        provider="meta",
        api_key="test-key",
        base_url="",
        model="muse-spark-1.3",
        reasoning_effort="high",
    )

    assert llm._sends_reasoning_effort() is True
