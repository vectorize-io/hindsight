"""Tests for global configuration."""

from __future__ import annotations

import os

import pytest

from hindsight_google_adk import (
    HindsightAdkConfig,
    configure,
    get_config,
    reset_config,
)
from hindsight_google_adk.config import DEFAULT_HINDSIGHT_API_URL


@pytest.fixture(autouse=True)
def _reset_global_config():
    reset_config()
    yield
    reset_config()


class TestConfigure:
    def test_defaults(self):
        cfg = configure()
        assert isinstance(cfg, HindsightAdkConfig)
        assert cfg.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL
        assert cfg.budget == "mid"
        assert cfg.max_tokens == 4096
        assert cfg.context == "google-adk"
        assert cfg.bank_id_template == "{app_name}::{user_id}"

    def test_explicit_url_wins(self):
        cfg = configure(hindsight_api_url="https://internal.example.com")
        assert cfg.hindsight_api_url == "https://internal.example.com"

    def test_api_key_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_KEY", "from-env")
        cfg = configure(api_key="from-arg")
        assert cfg.api_key == "from-arg"

    def test_api_key_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_KEY", "from-env")
        cfg = configure()
        assert cfg.api_key == "from-env"

    def test_api_key_unset_is_none(self, monkeypatch):
        monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
        cfg = configure()
        assert cfg.api_key is None

    def test_custom_bank_id_template(self):
        cfg = configure(bank_id_template="agent::{user_id}")
        assert cfg.bank_id_template == "agent::{user_id}"


class TestGetReset:
    def test_get_before_configure_returns_none(self):
        assert get_config() is None

    def test_get_after_configure_returns_config(self):
        cfg = configure(budget="high")
        assert get_config() is cfg

    def test_reset_clears_global(self):
        configure(budget="high")
        reset_config()
        assert get_config() is None
