"""Unit tests for Hindsight Claude Agent SDK configuration."""

import os
from unittest.mock import patch

from hindsight_claude_agent_sdk import (
    HindsightClaudeAgentSDKConfig,
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
        assert isinstance(config, HindsightClaudeAgentSDKConfig)
        assert config.hindsight_api_url == "http://localhost:8888"

    def test_configure_defaults(self):
        config = configure()
        assert config.hindsight_api_url == "https://api.hindsight.vectorize.io"
        assert config.budget == "mid"
        assert config.max_tokens == 4096
        assert config.recall_tags_match == "any"

    def test_configure_sets_global(self):
        assert get_config() is None
        configure(hindsight_api_url="http://localhost:8888")
        assert get_config() is not None
        assert get_config().hindsight_api_url == "http://localhost:8888"

    def test_reset_clears_global(self):
        configure(hindsight_api_url="http://localhost:8888")
        reset_config()
        assert get_config() is None

    def test_configure_picks_up_env_var(self):
        with patch.dict(os.environ, {"HINDSIGHT_API_KEY": "test-key-123"}):
            config = configure()
            assert config.api_key == "test-key-123"

    def test_explicit_api_key_overrides_env(self):
        with patch.dict(os.environ, {"HINDSIGHT_API_KEY": "env-key"}):
            config = configure(api_key="explicit-key")
            assert config.api_key == "explicit-key"

    def test_configure_with_tags(self):
        config = configure(
            hindsight_api_url="http://localhost:8888",
            tags=["source:test"],
            recall_tags=["scope:user"],
            recall_tags_match="all",
        )
        assert config.tags == ["source:test"]
        assert config.recall_tags == ["scope:user"]
        assert config.recall_tags_match == "all"

    def test_configure_overrides_previous(self):
        configure(hindsight_api_url="http://first:8888")
        configure(hindsight_api_url="http://second:9999")
        assert get_config().hindsight_api_url == "http://second:9999"
