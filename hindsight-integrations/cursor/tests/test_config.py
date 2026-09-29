"""Tests for Cursor plugin configuration loading."""

import os
import json
import tempfile

import pytest

from lib.config import load_config, DEFAULTS


class TestLoadConfig:
    def test_returns_defaults_when_no_files(self, monkeypatch):
        monkeypatch.delenv("CURSOR_PLUGIN_ROOT", raising=False)
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        config = load_config()
        assert config["autoRecall"] is True
        assert config["autoRetain"] is True
        assert config["recallBudget"] == "mid"
        assert config["retainContext"] == "cursor"
        assert config["agentName"] == "cursor"

    def test_env_overrides_bool(self, monkeypatch):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_AUTO_RECALL", "false")
        config = load_config()
        assert config["autoRecall"] is False

    def test_env_overrides_int(self, monkeypatch):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_RECALL_MAX_TOKENS", "2048")
        config = load_config()
        assert config["recallMaxTokens"] == 2048

    def test_env_overrides_string(self, monkeypatch):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_API_URL", "http://example.com")
        config = load_config()
        assert config["hindsightApiUrl"] == "http://example.com"

    def test_settings_file_loaded(self, monkeypatch, tmp_path):
        settings = {"bankId": "test-bank", "debug": True}
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps(settings))
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", str(tmp_path))
        # Prevent real user config from interfering
        monkeypatch.setattr("os.path.expanduser", lambda _: str(tmp_path / "fakehome"))
        config = load_config()
        assert config["bankId"] == "test-bank"
        assert config["debug"] is True

    def test_user_config_overrides_plugin_settings(self, monkeypatch, tmp_path):
        # Plugin settings
        plugin_settings = {"bankId": "plugin-bank", "debug": False}
        plugin_dir = tmp_path / "plugin"
        plugin_dir.mkdir()
        (plugin_dir / "settings.json").write_text(json.dumps(plugin_settings))
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", str(plugin_dir))

        # User config
        user_dir = tmp_path / "home" / ".hindsight"
        user_dir.mkdir(parents=True)
        user_config = {"bankId": "user-bank"}
        (user_dir / "cursor.json").write_text(json.dumps(user_config))
        monkeypatch.setenv("HOME", str(tmp_path / "home"))

        config = load_config()
        assert config["bankId"] == "user-bank"

    def test_env_overrides_user_config(self, monkeypatch, tmp_path):
        user_dir = tmp_path / "home" / ".hindsight"
        user_dir.mkdir(parents=True)
        (user_dir / "cursor.json").write_text(json.dumps({"bankId": "user-bank"}))
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_BANK_ID", "env-bank")

        config = load_config()
        assert config["bankId"] == "env-bank"

    def test_default_agent_name_is_cursor(self, monkeypatch):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        config = load_config()
        assert config["agentName"] == "cursor"

    def test_default_retain_context_is_cursor(self, monkeypatch):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        config = load_config()
        assert config["retainContext"] == "cursor"
