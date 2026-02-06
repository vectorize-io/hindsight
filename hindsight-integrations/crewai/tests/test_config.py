"""Unit tests for hindsight_crewai configuration."""

import os
from unittest.mock import patch

from hindsight_crewai import configure, get_config, reset_config
from hindsight_crewai.config import (
    DEFAULT_HINDSIGHT_API_URL,
    HINDSIGHT_API_KEY_ENV,
)


class TestDefaults:
    def test_default_api_url(self):
        assert DEFAULT_HINDSIGHT_API_URL == "https://api.hindsight.vectorize.io"

    def test_env_var_name(self):
        assert HINDSIGHT_API_KEY_ENV == "HINDSIGHT_API_KEY"


class TestConfigure:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_configure_with_no_arguments(self):
        config = configure()
        assert config.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL
        assert config.budget == "mid"
        assert config.max_tokens == 4096
        assert config.verbose is False

    def test_configure_reads_api_key_from_env(self):
        with patch.dict(os.environ, {HINDSIGHT_API_KEY_ENV: "test-key"}):
            config = configure()
        assert config.api_key == "test-key"

    def test_configure_explicit_overrides_env(self):
        with patch.dict(os.environ, {HINDSIGHT_API_KEY_ENV: "env-key"}):
            config = configure(api_key="explicit-key")
        assert config.api_key == "explicit-key"

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

    def test_get_config_returns_none_without_configure(self):
        assert get_config() is None

    def test_get_config_returns_config_after_configure(self):
        configure()
        assert get_config() is not None

    def test_reset_config(self):
        configure()
        assert get_config() is not None
        reset_config()
        assert get_config() is None
