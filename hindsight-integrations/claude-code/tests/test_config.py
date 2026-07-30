"""Tests for lib/config.py — configuration loading and env overrides."""

import json
import os

import pytest
from lib.config import _cast_env, load_config


class TestCastEnv:
    def test_bool_true_values(self):
        for v in ("true", "True", "TRUE", "1", "yes", "YES"):
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

    def test_defaults_applied_when_no_settings_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        # No settings.json in tmp_path
        cfg = load_config()
        assert cfg["autoRecall"] is True
        assert cfg["autoRetain"] is True
        assert cfg["recallBudget"] == "mid"
        assert cfg["retainEveryNTurns"] == 10

    def test_settings_json_overrides_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        (tmp_path / "settings.json").write_text(json.dumps({"recallBudget": "high", "bankId": "my-bank"}))
        cfg = load_config()
        assert cfg["recallBudget"] == "high"
        assert cfg["bankId"] == "my-bank"

    def test_env_var_overrides_settings_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        (tmp_path / "settings.json").write_text(json.dumps({"recallBudget": "low"}))
        monkeypatch.setenv("HINDSIGHT_RECALL_BUDGET", "high")
        cfg = load_config()
        assert cfg["recallBudget"] == "high"

    def test_bool_env_var_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_AUTO_RECALL", "false")
        cfg = load_config()
        assert cfg["autoRecall"] is False

    def test_int_env_var_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_API_PORT", "9999")
        cfg = load_config()
        assert cfg["apiPort"] == 9999

    def test_request_timeout_default_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        cfg = load_config()
        assert cfg["requestTimeoutSeconds"] is None

    def test_request_timeout_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_REQUEST_TIMEOUT_SECONDS", "60")
        cfg = load_config()
        assert cfg["requestTimeoutSeconds"] == 60

    def test_recall_tags_env_override_accepts_comma_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_RECALL_TAGS", "memory_type:rule, tech_stack:supabase")
        cfg = load_config()
        assert cfg["recallTags"] == ["memory_type:rule", "tech_stack:supabase"]

    def test_recall_tag_groups_env_override_accepts_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        monkeypatch.setenv(
            "HINDSIGHT_RECALL_TAG_GROUPS",
            '[{"op":"all","tags":["memory_type:rule","tech_stack:supabase"]}]',
        )
        cfg = load_config()
        assert cfg["recallTagGroups"] == [{"op": "all", "tags": ["memory_type:rule", "tech_stack:supabase"]}]

    def test_recall_additional_bank_filters_env_override_accepts_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        monkeypatch.setenv(
            "HINDSIGHT_RECALL_ADDITIONAL_BANK_FILTERS",
            '{"normative":{"recallTags":["memory_type:rule"],"recallTagsMatch":"all"}}',
        )
        cfg = load_config()
        assert cfg["recallAdditionalBankFilters"] == {
            "normative": {"recallTags": ["memory_type:rule"], "recallTagsMatch": "all"}
        }

    def test_invalid_settings_json_falls_back_to_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        (tmp_path / "settings.json").write_text("not valid json{{")
        cfg = load_config()
        assert cfg["recallBudget"] == "mid"  # default still applies

    def test_null_values_in_settings_json_not_applied(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        (tmp_path / "settings.json").write_text(json.dumps({"bankId": None, "recallBudget": "high"}))
        cfg = load_config()
        # None values in file should not override defaults
        assert cfg["bankId"] is None  # default is None, so ok
        assert cfg["recallBudget"] == "high"

    def test_api_url_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_API_URL", "http://myserver:8080")
        cfg = load_config()
        assert cfg["hindsightApiUrl"] == "http://myserver:8080"

    def test_user_config_overrides_plugin_settings(self, tmp_path, monkeypatch):
        plugin_root = tmp_path / "plugin"
        plugin_root.mkdir()

        # Plugin default ships with "low"
        (plugin_root / "settings.json").write_text(json.dumps({"recallBudget": "low"}))
        # User overrides to "high" via ~/.hindsight/claude-code.json
        user_cfg = tmp_path / ".hindsight" / "claude-code.json"
        user_cfg.parent.mkdir()
        user_cfg.write_text(json.dumps({"recallBudget": "high"}))

        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(plugin_root))
        monkeypatch.setenv("HOME", str(tmp_path))
        cfg = load_config()
        assert cfg["recallBudget"] == "high"

    def test_user_config_missing_falls_back_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        # HOME points to tmp_path where no .hindsight/claude-code.json exists
        monkeypatch.setenv("HOME", str(tmp_path))
        cfg = load_config()
        assert cfg["recallBudget"] == "mid"  # default

    def test_env_var_wins_over_user_config(self, tmp_path, monkeypatch):
        plugin_root = tmp_path / "plugin"
        plugin_root.mkdir()
        user_cfg_dir = tmp_path / ".hindsight"
        user_cfg_dir.mkdir()
        (user_cfg_dir / "claude-code.json").write_text(json.dumps({"recallBudget": "low"}))

        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(plugin_root))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_RECALL_BUDGET", "high")
        cfg = load_config()
        assert cfg["recallBudget"] == "high"


class TestProfilePresets:
    @pytest.fixture(autouse=True)
    def _isolate_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        for k in list(os.environ):
            if k.startswith("HINDSIGHT_"):
                monkeypatch.delenv(k, raising=False)

    def _write_user_config(self, tmp_path, data):
        user_cfg_dir = tmp_path / ".hindsight"
        user_cfg_dir.mkdir(exist_ok=True)
        (user_cfg_dir / "claude-code.json").write_text(json.dumps(data))

    def test_no_profile_keeps_chat_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        cfg = load_config()
        assert cfg["profile"] is None
        assert cfg["retainToolCalls"] is False
        assert cfg["dynamicBankId"] is False
        assert cfg["recallBudget"] == "mid"

    def test_coding_profile_applies_preset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        self._write_user_config(tmp_path, {"profile": "coding"})
        cfg = load_config()
        assert cfg["retainToolCalls"] is True
        assert cfg["dynamicBankId"] is True
        assert cfg["dynamicBankGranularity"] == ["agent", "project"]
        assert cfg["recallBudget"] == "low"
        assert "engineering" in cfg["retainMission"]

    def test_coding_profile_overrides_plugin_settings_json(self, tmp_path, monkeypatch):
        # The shipped settings.json (vendor defaults) says retainToolCalls
        # false — a profile must override vendor defaults.
        plugin_root = tmp_path / "plugin"
        plugin_root.mkdir()
        (plugin_root / "settings.json").write_text(json.dumps({"retainToolCalls": False, "recallBudget": "mid"}))
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(plugin_root))
        self._write_user_config(tmp_path, {"profile": "coding"})
        cfg = load_config()
        assert cfg["retainToolCalls"] is True
        assert cfg["recallBudget"] == "low"

    def test_explicit_user_config_beats_preset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        self._write_user_config(tmp_path, {"profile": "coding", "recallBudget": "high"})
        cfg = load_config()
        assert cfg["recallBudget"] == "high"  # user's explicit choice wins
        assert cfg["retainToolCalls"] is True  # rest of preset still applies

    def test_env_var_beats_preset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        self._write_user_config(tmp_path, {"profile": "coding"})
        monkeypatch.setenv("HINDSIGHT_RECALL_BUDGET", "high")
        cfg = load_config()
        assert cfg["recallBudget"] == "high"
        assert cfg["dynamicBankId"] is True

    def test_profile_via_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_PROFILE", "coding")
        cfg = load_config()
        assert cfg["retainToolCalls"] is True
        assert cfg["dynamicBankId"] is True

    def test_unknown_profile_warns_and_keeps_defaults(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        self._write_user_config(tmp_path, {"profile": "gardening"})
        cfg = load_config()
        assert cfg["retainToolCalls"] is False  # untouched defaults
        assert "Unknown profile" in capsys.readouterr().err
