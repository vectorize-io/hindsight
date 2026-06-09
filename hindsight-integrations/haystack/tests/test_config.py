"""Unit tests for Hindsight Haystack configuration."""

import os
from unittest.mock import patch

from hindsight_haystack import (
    HindsightHaystackConfig,
    configure,
    get_config,
    reset_config,
)


class TestConfigure:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_configure_returns_config(self):
        config = configure(hindsight_api_url="http://localhost:8888")
        assert isinstance(config, HindsightHaystackConfig)
        assert config.hindsight_api_url == "http://localhost:8888"

    def test_configure_sets_global_config(self):
        assert get_config() is None
        configure(hindsight_api_url="http://localhost:8888")
        assert get_config() is not None
        assert get_config().hindsight_api_url == "http://localhost:8888"

    def test_configure_defaults(self):
        config = configure()
        assert config.hindsight_api_url == "https://api.hindsight.vectorize.io"
        assert config.api_key is None
        assert config.budget == "mid"
        assert config.max_tokens == 4096
        assert config.tags is None
        assert config.recall_tags is None
        assert config.recall_tags_match == "any"
        assert config.context == "haystack"
        assert config.mission is None
        assert config.verbose is False

    def test_configure_with_all_params(self):
        config = configure(
            hindsight_api_url="http://test:9999",
            api_key="test-key",
            budget="high",
            max_tokens=2048,
            tags=["tag1"],
            recall_tags=["rtag1"],
            recall_tags_match="all",
            context="my-app",
            mission="test mission",
            verbose=True,
        )
        assert config.hindsight_api_url == "http://test:9999"
        assert config.api_key == "test-key"
        assert config.budget == "high"
        assert config.max_tokens == 2048
        assert config.tags == ["tag1"]
        assert config.recall_tags == ["rtag1"]
        assert config.recall_tags_match == "all"
        assert config.context == "my-app"
        assert config.mission == "test mission"
        assert config.verbose is True

    def test_reset_config(self):
        configure(hindsight_api_url="http://localhost:8888")
        assert get_config() is not None
        reset_config()
        assert get_config() is None

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"HINDSIGHT_API_KEY": "env-key"}):
            config = configure()
            assert config.api_key == "env-key"

    def test_explicit_api_key_overrides_env(self):
        with patch.dict(os.environ, {"HINDSIGHT_API_KEY": "env-key"}):
            config = configure(api_key="explicit-key")
            assert config.api_key == "explicit-key"

    def test_context_defaults_to_haystack(self):
        config = configure()
        assert config.context == "haystack"
