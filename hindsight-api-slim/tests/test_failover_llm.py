"""Tests for the LLM failover provider feature."""

import pytest


@pytest.fixture
def clean_config(monkeypatch):
    """Clear cached config before/after each test so env changes take effect."""
    from hindsight_api.config import clear_config_cache

    # Ensure no pre-existing failover env vars leak in
    for key in (
        "HINDSIGHT_API_LLM_FAILOVER_PROVIDER",
        "HINDSIGHT_API_LLM_FAILOVER_API_KEY",
        "HINDSIGHT_API_LLM_FAILOVER_MODEL",
        "HINDSIGHT_API_LLM_FAILOVER_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    clear_config_cache()
    yield
    clear_config_cache()


def test_failover_fields_default_to_none(clean_config):
    """When no failover env vars are set, all four failover fields are None."""
    from hindsight_api.config import get_config

    config = get_config()
    assert config.llm_failover_provider is None
    assert config.llm_failover_api_key is None
    assert config.llm_failover_model is None
    assert config.llm_failover_base_url is None


def test_failover_fields_load_from_env(clean_config, monkeypatch):
    """Setting failover env vars populates the four fields."""
    from hindsight_api.config import clear_config_cache, get_config

    monkeypatch.setenv("HINDSIGHT_API_LLM_FAILOVER_PROVIDER", "anthropic")
    monkeypatch.setenv("HINDSIGHT_API_LLM_FAILOVER_API_KEY", "sk-test-failover")
    monkeypatch.setenv("HINDSIGHT_API_LLM_FAILOVER_MODEL", "claude-3-5-sonnet-latest")
    monkeypatch.setenv("HINDSIGHT_API_LLM_FAILOVER_BASE_URL", "https://api.anthropic.com")
    clear_config_cache()

    config = get_config()
    assert config.llm_failover_provider == "anthropic"
    assert config.llm_failover_api_key == "sk-test-failover"
    assert config.llm_failover_model == "claude-3-5-sonnet-latest"
    assert config.llm_failover_base_url == "https://api.anthropic.com"


def test_failover_credentials_marked_as_credential_fields():
    """The failover api_key and base_url must be in _CREDENTIAL_FIELDS so the API never echoes them."""
    from hindsight_api.config import HindsightConfig

    credential_fields = HindsightConfig.get_credential_fields()
    assert "llm_failover_api_key" in credential_fields
    assert "llm_failover_base_url" in credential_fields


def test_failover_fields_are_static_not_configurable():
    """Per spec, provider/model/credentials are server-level only — never per-bank configurable."""
    from hindsight_api.config import HindsightConfig

    configurable = HindsightConfig.get_configurable_fields()
    for field in (
        "llm_failover_provider",
        "llm_failover_api_key",
        "llm_failover_model",
        "llm_failover_base_url",
    ):
        assert field not in configurable, f"{field} must not be per-bank configurable"
