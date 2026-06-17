"""Tests for lib/config.py — configuration loading and env overrides."""

import json
import os

import pytest

from lib.config import _cast_env, load_config


class TestCastEnv:
    def test_bool_true_values(self):
        for v in ("true", "True", "TRUE", "1", "yes"):
            assert _cast_env(v, bool) is True

    def test_bool_false_values(self):
        for v in ("false", "False", "0", "no"):
            assert _cast_env(v, bool) is False

    def test_int_cast(self):
        assert _cast_env("42", int) == 42

    def test_int_invalid_returns_none(self):
        assert _cast_env("notanint", int) is None

    def test_str_passthrough(self):
        assert _cast_env("hello", str) == "hello"


class TestLoadConfig:
    @pytest.fixture(autouse=True)
    def _isolate_config(self, tmp_path, monkeypatch):
        """Isolate from real user config and env vars."""
        monkeypatch.setenv("HOME", str(tmp_path))
        for k in list(os.environ):
            if k.startswith("HINDSIGHT_"):
                monkeypatch.delenv(k, raising=False)

    def test_defaults_include_cloud_url(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLUGIN_ROOT", str(tmp_path))
        cfg = load_config()
        assert cfg["hindsightApiUrl"] == "https://api.hindsight.vectorize.io"

    def test_defaults_applied_when_no_settings_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLUGIN_ROOT", str(tmp_path))
        cfg = load_config()
        assert cfg["autoRecall"] is True
        assert cfg["autoRetain"] is True
        assert cfg["recallBudget"] == "mid"
        assert cfg["retainEveryNTurns"] == 10
        assert cfg["agentName"] == "omo"
        assert cfg["retainContext"] == "omo"

    def test_settings_json_overrides_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLUGIN_ROOT", str(tmp_path))
        (tmp_path / "settings.json").write_text(json.dumps({"recallBudget": "high", "bankId": "my-bank"}))
        cfg = load_config()
        assert cfg["recallBudget"] == "high"
        assert cfg["bankId"] == "my-bank"

    def test_env_var_overrides_settings_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLUGIN_ROOT", str(tmp_path))
        (tmp_path / "settings.json").write_text(json.dumps({"recallBudget": "low"}))
        monkeypatch.setenv("HINDSIGHT_RECALL_BUDGET", "high")
        cfg = load_config()
        assert cfg["recallBudget"] == "high"

    def test_bool_env_var_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLUGIN_ROOT", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_AUTO_RECALL", "false")
        cfg = load_config()
        assert cfg["autoRecall"] is False

    def test_api_url_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLUGIN_ROOT", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_API_URL", "http://localhost:8888")
        cfg = load_config()
        assert cfg["hindsightApiUrl"] == "http://localhost:8888"

    def test_api_token_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLUGIN_ROOT", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_API_TOKEN", "hsk_test123")
        cfg = load_config()
        assert cfg["hindsightApiToken"] == "hsk_test123"

    def test_user_config_overrides_plugin_settings(self, tmp_path, monkeypatch):
        plugin_root = tmp_path / "plugin"
        plugin_root.mkdir()
        (plugin_root / "settings.json").write_text(json.dumps({"recallBudget": "low"}))
        user_cfg = tmp_path / ".hindsight" / "omo.json"
        user_cfg.parent.mkdir()
        user_cfg.write_text(json.dumps({"recallBudget": "high"}))

        monkeypatch.setenv("PLUGIN_ROOT", str(plugin_root))
        cfg = load_config()
        assert cfg["recallBudget"] == "high"

    def test_env_var_wins_over_user_config(self, tmp_path, monkeypatch):
        plugin_root = tmp_path / "plugin"
        plugin_root.mkdir()
        user_cfg_dir = tmp_path / ".hindsight"
        user_cfg_dir.mkdir()
        (user_cfg_dir / "omo.json").write_text(json.dumps({"recallBudget": "low"}))

        monkeypatch.setenv("PLUGIN_ROOT", str(plugin_root))
        monkeypatch.setenv("HINDSIGHT_RECALL_BUDGET", "high")
        cfg = load_config()
        assert cfg["recallBudget"] == "high"

    def test_invalid_settings_json_falls_back_to_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLUGIN_ROOT", str(tmp_path))
        (tmp_path / "settings.json").write_text("not valid json{{")
        cfg = load_config()
        assert cfg["recallBudget"] == "mid"
