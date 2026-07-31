"""Tests for lib/llm.py — provider/key resolution priority.

The documented order is: HINDSIGHT_API_LLM_* overrides, then plugin config
(`llmProvider` / `llmApiKeyEnv`), then auto-detection from a standard provider
env var, then external-API mode. The cases below pin the boundaries between
those tiers, which is where the resolution used to disagree with the docs.
"""

import pytest
from lib.llm import detect_llm_config


@pytest.fixture(autouse=True)
def _no_ambient_provider_keys(monkeypatch):
    """Strip anything the developer's own shell exports."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key in (
        "HINDSIGHT_API_LLM_PROVIDER",
        "HINDSIGHT_API_LLM_API_KEY",
        "HINDSIGHT_API_LLM_MODEL",
        "HINDSIGHT_API_LLM_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


class TestApiKeyOverrideWithConfiguredProvider:
    """`HINDSIGHT_API_LLM_API_KEY` is documented as the top-priority key source.

    It used to be read only alongside `HINDSIGHT_API_LLM_PROVIDER`, so pairing
    it with `llmProvider` in settings.json failed outright.
    """

    def test_override_key_satisfies_a_config_provider(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-override")

        result = detect_llm_config({"llmProvider": "openai"})

        assert result["provider"] == "openai"
        assert result["api_key"] == "sk-override"

    def test_override_key_wins_over_the_providers_standard_env_var(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-override")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-stale")

        result = detect_llm_config({"llmProvider": "openai"})

        assert result["api_key"] == "sk-override", "a stale OPENAI_API_KEY must not shadow the explicit override"

    def test_llm_api_key_env_is_still_honoured_without_an_override(self, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_KEY", "sk-custom")

        result = detect_llm_config({"llmProvider": "openai", "llmApiKeyEnv": "MY_CUSTOM_KEY"})

        assert result["api_key"] == "sk-custom"

    def test_a_config_provider_with_no_key_anywhere_still_fails(self):
        with pytest.raises(RuntimeError, match="no API key found"):
            detect_llm_config({"llmProvider": "openai"})


class TestModelOverrideWithConfiguredProvider:
    """`HINDSIGHT_API_LLM_MODEL` is documented as the top-priority model source.

    In the plugin-config branch a configured `llmModel` used to win over it, so
    the daemon ran a different model than the environment asked for. `base_url`
    in that same branch always honoured the override, which is what made the
    model's behaviour an inconsistency rather than a deliberate choice.
    """

    def test_override_model_wins_over_a_configured_model(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-override")
        monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "gpt-from-env")

        result = detect_llm_config({"llmProvider": "openai", "llmModel": "gpt-from-config"})

        assert result["model"] == "gpt-from-env"

    def test_configured_model_is_still_used_without_an_override(self, monkeypatch):
        """Control: the override only wins when it is actually set."""
        monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-override")

        result = detect_llm_config({"llmProvider": "openai", "llmModel": "gpt-from-config"})

        assert result["model"] == "gpt-from-config"
