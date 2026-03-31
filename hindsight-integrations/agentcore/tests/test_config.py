import os
import pytest
from hindsight_agentcore.config import (
    configure,
    get_config,
    reset_config,
    DEFAULT_HINDSIGHT_API_URL,
)


class TestConfigure:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_returns_config_object(self):
        cfg = configure(hindsight_api_url="http://localhost:8888")
        assert cfg.hindsight_api_url == "http://localhost:8888"

    def test_get_config_returns_none_before_configure(self):
        assert get_config() is None

    def test_get_config_returns_config_after_configure(self):
        configure(hindsight_api_url="http://localhost:8888")
        assert get_config() is not None
        assert get_config().hindsight_api_url == "http://localhost:8888"

    def test_reset_config_clears_config(self):
        configure(hindsight_api_url="http://localhost:8888")
        reset_config()
        assert get_config() is None

    def test_defaults_to_hindsight_cloud(self):
        cfg = configure()
        assert cfg.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL

    def test_reads_api_url_from_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_URL", "http://from-env:8888")
        cfg = configure()
        assert cfg.hindsight_api_url == "http://from-env:8888"

    def test_explicit_url_overrides_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_URL", "http://from-env:8888")
        cfg = configure(hindsight_api_url="http://explicit:9000")
        assert cfg.hindsight_api_url == "http://explicit:9000"

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_KEY", "test-key-123")
        cfg = configure()
        assert cfg.api_key == "test-key-123"

    def test_reads_api_token_env_as_fallback(self, monkeypatch):
        monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
        monkeypatch.setenv("HINDSIGHT_API_TOKEN", "token-456")
        cfg = configure()
        assert cfg.api_key == "token-456"

    def test_recall_budget_default(self):
        cfg = configure()
        assert cfg.recall_budget == "mid"

    def test_retain_async_default_true(self):
        cfg = configure()
        assert cfg.retain_async is True

    def test_custom_recall_settings(self):
        cfg = configure(recall_budget="high", recall_max_tokens=3000)
        assert cfg.recall_budget == "high"
        assert cfg.recall_max_tokens == 3000
