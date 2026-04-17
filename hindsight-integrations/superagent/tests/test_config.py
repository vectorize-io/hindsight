"""Unit tests for Hindsight-Superagent configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

from hindsight_superagent import (
    HindsightSuperagentConfig,
    configure,
    get_config,
    reset_config,
)


class TestConfigureLifecycle:
    def setup_method(self) -> None:
        reset_config()

    def teardown_method(self) -> None:
        reset_config()

    def test_get_config_returns_none_by_default(self) -> None:
        assert get_config() is None

    def test_configure_returns_config(self) -> None:
        config = configure(hindsight_api_url="http://localhost:8888")
        assert isinstance(config, HindsightSuperagentConfig)
        assert config.hindsight_api_url == "http://localhost:8888"

    def test_get_config_returns_configured_value(self) -> None:
        configure(hindsight_api_url="http://localhost:8888")
        config = get_config()
        assert config is not None
        assert config.hindsight_api_url == "http://localhost:8888"

    def test_reset_config_clears_state(self) -> None:
        configure(hindsight_api_url="http://localhost:8888")
        assert get_config() is not None
        reset_config()
        assert get_config() is None

    def test_configure_overwrites_previous(self) -> None:
        configure(hindsight_api_url="http://first:8888")
        configure(hindsight_api_url="http://second:8888")
        config = get_config()
        assert config is not None
        assert config.hindsight_api_url == "http://second:8888"


class TestConfigureDefaults:
    def setup_method(self) -> None:
        reset_config()

    def teardown_method(self) -> None:
        reset_config()

    def test_default_budget(self) -> None:
        config = configure()
        assert config.budget == "mid"

    def test_default_max_tokens(self) -> None:
        config = configure()
        assert config.max_tokens == 4096

    def test_default_tags_are_none(self) -> None:
        config = configure()
        assert config.tags is None

    def test_default_recall_tags_are_none(self) -> None:
        config = configure()
        assert config.recall_tags is None

    def test_default_recall_tags_match(self) -> None:
        config = configure()
        assert config.recall_tags_match == "any"

    def test_default_guard_model_is_none(self) -> None:
        config = configure()
        assert config.guard_model is None

    def test_default_redact_model_is_none(self) -> None:
        config = configure()
        assert config.redact_model is None

    def test_default_redact_rewrite_is_false(self) -> None:
        config = configure()
        assert config.redact_rewrite is False

    def test_default_guard_enabled(self) -> None:
        config = configure()
        assert config.enable_guard_on_retain is True
        assert config.enable_guard_on_recall is True
        assert config.enable_guard_on_reflect is True

    def test_default_redact_enabled(self) -> None:
        config = configure()
        assert config.enable_redact_on_retain is True

    def test_default_fallback_disabled(self) -> None:
        config = configure()
        assert config.enable_fallback is False
        assert config.fallback_timeout == 5.0

    def test_default_verbose_is_false(self) -> None:
        config = configure()
        assert config.verbose is False


class TestConfigureCustomValues:
    def setup_method(self) -> None:
        reset_config()

    def teardown_method(self) -> None:
        reset_config()

    def test_all_params_set(self) -> None:
        config = configure(
            hindsight_api_url="http://custom:9999",
            api_key="h-key",
            superagent_api_key="sa-key",
            budget="high",
            max_tokens=2048,
            tags=["env:test"],
            recall_tags=["scope:global"],
            recall_tags_match="all",
            guard_model="openai/gpt-4o",
            redact_model="openai/gpt-4o-mini",
            redact_entities=["emails", "ssns"],
            redact_rewrite=True,
            enable_guard_on_retain=False,
            enable_guard_on_recall=False,
            enable_guard_on_reflect=False,
            enable_redact_on_retain=False,
            enable_fallback=True,
            fallback_timeout=10.0,
            verbose=True,
        )
        assert config.hindsight_api_url == "http://custom:9999"
        assert config.api_key == "h-key"
        assert config.superagent_api_key == "sa-key"
        assert config.budget == "high"
        assert config.max_tokens == 2048
        assert config.tags == ["env:test"]
        assert config.recall_tags == ["scope:global"]
        assert config.recall_tags_match == "all"
        assert config.guard_model == "openai/gpt-4o"
        assert config.redact_model == "openai/gpt-4o-mini"
        assert config.redact_entities == ["emails", "ssns"]
        assert config.redact_rewrite is True
        assert config.enable_guard_on_retain is False
        assert config.enable_guard_on_recall is False
        assert config.enable_guard_on_reflect is False
        assert config.enable_redact_on_retain is False
        assert config.enable_fallback is True
        assert config.fallback_timeout == 10.0
        assert config.verbose is True


class TestConfigureEnvVars:
    def setup_method(self) -> None:
        reset_config()

    def teardown_method(self) -> None:
        reset_config()

    def test_api_key_from_env(self) -> None:
        with patch.dict(os.environ, {"HINDSIGHT_API_KEY": "env-h-key"}):
            config = configure()
            assert config.api_key == "env-h-key"

    def test_superagent_api_key_from_env(self) -> None:
        with patch.dict(os.environ, {"SUPERAGENT_API_KEY": "env-sa-key"}):
            config = configure()
            assert config.superagent_api_key == "env-sa-key"

    def test_explicit_key_overrides_env(self) -> None:
        with patch.dict(os.environ, {"HINDSIGHT_API_KEY": "env-key"}):
            config = configure(api_key="explicit-key")
            assert config.api_key == "explicit-key"

    def test_explicit_superagent_key_overrides_env(self) -> None:
        with patch.dict(os.environ, {"SUPERAGENT_API_KEY": "env-sa-key"}):
            config = configure(superagent_api_key="explicit-sa-key")
            assert config.superagent_api_key == "explicit-sa-key"
