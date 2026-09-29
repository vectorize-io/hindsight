"""Unit tests for hindsight_agno configuration."""

import os
from unittest.mock import patch

from hindsight_agno import configure, get_config, reset_config
from hindsight_agno.config import (
    DEFAULT_HINDSIGHT_API_URL,
    HINDSIGHT_API_KEY_ENV,
    HindsightAgnoConfig,
)


class TestDefaults:
    def test_default_api_url(self):
        assert DEFAULT_HINDSIGHT_API_URL == "https://api.hindsight.vectorize.io"

    def test_env_var_name(self):
        assert HINDSIGHT_API_KEY_ENV == "HINDSIGHT_API_KEY"


class TestHindsightAgnoConfigDataclass:
    def test_default_values(self):
        config = HindsightAgnoConfig()
        assert config.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL
        assert config.api_key is None
        assert config.budget == "mid"
        assert config.max_tokens == 4096
        assert config.tags is None
        assert config.recall_tags is None
        assert config.recall_tags_match == "any"
        assert config.verbose is False

    def test_custom_values(self):
        config = HindsightAgnoConfig(
            hindsight_api_url="http://custom:9999",
            api_key="key-123",
            budget="high",
            max_tokens=2048,
            tags=["t1"],
            recall_tags=["r1"],
            recall_tags_match="all",
            verbose=True,
        )
        assert config.hindsight_api_url == "http://custom:9999"
        assert config.api_key == "key-123"
        assert config.budget == "high"
        assert config.max_tokens == 2048
        assert config.tags == ["t1"]
        assert config.recall_tags == ["r1"]
        assert config.recall_tags_match == "all"
        assert config.verbose is True

    def test_is_mutable_dataclass(self):
        config = HindsightAgnoConfig()
        config.budget = "low"
        assert config.budget == "low"


class TestConfigure:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_configure_with_no_arguments(self):
        config = configure()
        assert config.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL
        assert config.api_key is None
        assert config.budget == "mid"
        assert config.max_tokens == 4096
        assert config.tags is None
        assert config.recall_tags is None
        assert config.recall_tags_match == "any"
        assert config.verbose is False

    def test_configure_reads_api_key_from_env(self):
        with patch.dict(os.environ, {HINDSIGHT_API_KEY_ENV: "test-key"}):
            config = configure()
        assert config.api_key == "test-key"

    def test_configure_explicit_overrides_env(self):
        with patch.dict(os.environ, {HINDSIGHT_API_KEY_ENV: "env-key"}):
            config = configure(api_key="explicit-key")
        assert config.api_key == "explicit-key"

    def test_configure_api_key_none_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            config = configure()
        assert config.api_key is None

    def test_configure_all_options(self):
        config = configure(
            hindsight_api_url="http://custom:8888",
            api_key="my-key",
            budget="high",
            max_tokens=2048,
            tags=["env:test"],
            recall_tags=["scope:global"],
            recall_tags_match="all",
            verbose=True,
        )
        assert config.hindsight_api_url == "http://custom:8888"
        assert config.api_key == "my-key"
        assert config.budget == "high"
        assert config.max_tokens == 2048
        assert config.tags == ["env:test"]
        assert config.recall_tags == ["scope:global"]
        assert config.recall_tags_match == "all"
        assert config.verbose is True

    def test_configure_returns_config_instance(self):
        config = configure()
        assert isinstance(config, HindsightAgnoConfig)

    def test_configure_replaces_previous_config(self):
        configure(budget="low")
        config1 = get_config()
        assert config1.budget == "low"

        configure(budget="high")
        config2 = get_config()
        assert config2.budget == "high"
        assert config1 is not config2

    def test_configure_url_defaults_when_none(self):
        config = configure(hindsight_api_url=None)
        assert config.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL


class TestGetConfig:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_returns_none_without_configure(self):
        assert get_config() is None

    def test_returns_config_after_configure(self):
        configure()
        config = get_config()
        assert config is not None
        assert isinstance(config, HindsightAgnoConfig)

    def test_returns_same_instance(self):
        configure()
        assert get_config() is get_config()


class TestResetConfig:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_reset_config(self):
        configure()
        assert get_config() is not None
        reset_config()
        assert get_config() is None

    def test_reset_is_idempotent(self):
        reset_config()
        reset_config()
        assert get_config() is None
