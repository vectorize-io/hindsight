"""Tests for lib/config.py — configuration loading and env overrides.

Ported from the Claude Code plugin's test_config.py. The one structural
difference: Devin CLI doesn't set a `CLAUDE_PLUGIN_ROOT`-equivalent env var,
so `lib.config.plugin_root()` resolves its own location from `__file__`
instead — tests monkeypatch that function directly rather than an env var.
"""

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

    def test_dict_cast_rejects_a_json_array(self):
        """A dict setting must not accept a list — the caller will .get() on it.

        The list branch already requires a list; accepting either shape here
        turns a wrong env var into an AttributeError inside a hook rather than
        an ignored value.
        """
        assert _cast_env('["a", "b"]', dict) is None
        assert _cast_env('{"a": 1}', dict) == {"a": 1}


class TestLoadConfig:
    @pytest.fixture(autouse=True)
    def _isolate_config(self, tmp_path, monkeypatch):
        """Isolate from real user config and env vars."""
        monkeypatch.setenv("HOME", str(tmp_path))
        for k in list(os.environ):
            if k.startswith("HINDSIGHT_"):
                monkeypatch.delenv(k, raising=False)

    def _set_plugin_root(self, monkeypatch, path):
        monkeypatch.setattr("lib.config.plugin_root", lambda: str(path))

    def test_defaults_applied_when_no_settings_file(self, tmp_path, monkeypatch):
        self._set_plugin_root(monkeypatch, tmp_path)
        cfg = load_config()
        assert cfg["autoRecall"] is True
        assert cfg["autoRetain"] is True
        assert cfg["recallBudget"] == "mid"
        assert cfg["retainEveryNTurns"] == 10
        assert cfg["agentName"] == "devin-cli"

    def test_settings_json_overrides_defaults(self, tmp_path, monkeypatch):
        self._set_plugin_root(monkeypatch, tmp_path)
        (tmp_path / "settings.json").write_text(json.dumps({"recallBudget": "high", "bankId": "my-bank"}))
        cfg = load_config()
        assert cfg["recallBudget"] == "high"
        assert cfg["bankId"] == "my-bank"

    def test_env_var_overrides_settings_json(self, tmp_path, monkeypatch):
        self._set_plugin_root(monkeypatch, tmp_path)
        (tmp_path / "settings.json").write_text(json.dumps({"recallBudget": "low"}))
        monkeypatch.setenv("HINDSIGHT_RECALL_BUDGET", "high")
        cfg = load_config()
        assert cfg["recallBudget"] == "high"

    def test_bool_env_var_override(self, tmp_path, monkeypatch):
        self._set_plugin_root(monkeypatch, tmp_path)
        monkeypatch.setenv("HINDSIGHT_AUTO_RECALL", "false")
        cfg = load_config()
        assert cfg["autoRecall"] is False

    def test_int_env_var_override(self, tmp_path, monkeypatch):
        self._set_plugin_root(monkeypatch, tmp_path)
        monkeypatch.setenv("HINDSIGHT_API_PORT", "9999")
        cfg = load_config()
        assert cfg["apiPort"] == 9999

    def test_request_timeout_default_none(self, tmp_path, monkeypatch):
        self._set_plugin_root(monkeypatch, tmp_path)
        cfg = load_config()
        assert cfg["requestTimeoutSeconds"] is None

    def test_recall_tags_env_override_accepts_comma_list(self, tmp_path, monkeypatch):
        self._set_plugin_root(monkeypatch, tmp_path)
        monkeypatch.setenv("HINDSIGHT_RECALL_TAGS", "memory_type:rule, tech_stack:supabase")
        cfg = load_config()
        assert cfg["recallTags"] == ["memory_type:rule", "tech_stack:supabase"]

    def test_invalid_settings_json_falls_back_to_defaults(self, tmp_path, monkeypatch):
        self._set_plugin_root(monkeypatch, tmp_path)
        (tmp_path / "settings.json").write_text("not valid json{{")
        cfg = load_config()
        assert cfg["recallBudget"] == "mid"

    def test_null_values_in_settings_json_not_applied(self, tmp_path, monkeypatch):
        self._set_plugin_root(monkeypatch, tmp_path)
        (tmp_path / "settings.json").write_text(json.dumps({"bankId": None, "recallBudget": "high"}))
        cfg = load_config()
        assert cfg["bankId"] is None
        assert cfg["recallBudget"] == "high"

    def test_user_config_overrides_plugin_settings(self, tmp_path, monkeypatch):
        plugin_root = tmp_path / "plugin"
        plugin_root.mkdir()

        (plugin_root / "settings.json").write_text(json.dumps({"recallBudget": "low"}))
        user_cfg = tmp_path / ".hindsight" / "devin-cli.json"
        user_cfg.parent.mkdir()
        user_cfg.write_text(json.dumps({"recallBudget": "high"}))

        self._set_plugin_root(monkeypatch, plugin_root)
        monkeypatch.setenv("HOME", str(tmp_path))
        cfg = load_config()
        assert cfg["recallBudget"] == "high"

    def test_user_config_missing_falls_back_gracefully(self, tmp_path, monkeypatch):
        self._set_plugin_root(monkeypatch, tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        cfg = load_config()
        assert cfg["recallBudget"] == "mid"

    def test_env_var_wins_over_user_config(self, tmp_path, monkeypatch):
        plugin_root = tmp_path / "plugin"
        plugin_root.mkdir()
        user_cfg_dir = tmp_path / ".hindsight"
        user_cfg_dir.mkdir()
        (user_cfg_dir / "devin-cli.json").write_text(json.dumps({"recallBudget": "low"}))

        self._set_plugin_root(monkeypatch, plugin_root)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_RECALL_BUDGET", "high")
        cfg = load_config()
        assert cfg["recallBudget"] == "high"

    def test_plugin_root_resolves_from_file_location(self):
        from lib.config import plugin_root

        # scripts/lib/config.py -> plugin root is two levels up from scripts/
        assert os.path.basename(plugin_root()) == "devin-cli"


class TestMalformedSettingsFile:
    def test_a_non_object_settings_file_is_ignored_not_fatal(self, tmp_path, monkeypatch):
        """Valid JSON that is not an object used to raise AttributeError.

        `.items()` on a list escaped the (JSONDecodeError, OSError) handler and
        propagated out of load_config, stopping every hook from running.
        """
        settings = tmp_path / "settings.json"
        settings.write_text('["not", "an", "object"]')
        monkeypatch.setattr("lib.config.plugin_root", lambda: str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        from lib.config import load_config

        config = load_config()

        assert config["apiPort"] == 9078, "defaults should survive an unusable settings file"


class TestMalformedSettingsFallBackToDefaults:
    """A settings file is arbitrary user JSON, and wrong shapes crash hooks.

    A string where a list belongs iterates one character at a time; a list
    where a dict belongs raises AttributeError from .get(); a string where an
    int belongs raises TypeError from arithmetic. All of it happens inside a
    hook, so a single mistyped optional setting used to take recall or retain
    down entirely — a value added to *tune* the plugin silently switching it
    off. Each is reverted to its default instead.

    Checked centrally rather than per call site: DEFAULTS is the type table, so
    a new setting is covered the moment it has a default.
    """

    @pytest.fixture(autouse=True)
    def _isolate_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        for k in list(os.environ):
            if k.startswith("HINDSIGHT_"):
                monkeypatch.delenv(k, raising=False)
        monkeypatch.setattr("lib.config.plugin_root", lambda: str(tmp_path))
        self.root = tmp_path

    def _load(self, settings):
        (self.root / "settings.json").write_text(json.dumps(settings))
        return load_config()

    @pytest.mark.parametrize(
        ("key", "bad", "expected"),
        [
            ("directoryBankMap", ["a", "b"], {}),
            ("directoryBankMap", "some-bank", {}),
            ("recallAdditionalBanks", "one-bank", []),
            ("recallAdditionalBanks", 3, []),
            ("recallMinScores", [], {}),
            ("recallAdditionalBankFilters", "nope", {}),
            ("retainRoles", "user", ["user", "assistant"]),
            ("recallTypes", "observation", ["observation"]),
            ("retainEveryNTurns", "10", 10),
            ("apiPort", "9078", 9078),
        ],
    )
    def test_a_wrong_shape_reverts_to_the_default(self, key, bad, expected):
        assert self._load({key: bad})[key] == expected

    def test_a_bool_does_not_satisfy_an_int_setting(self):
        """bool subclasses int, so `true` would reach arithmetic as 1."""
        assert self._load({"retainEveryNTurns": True})["retainEveryNTurns"] == 10

    def test_a_correctly_shaped_value_is_kept(self):
        """Control: the check must reject shapes, not values."""
        cfg = self._load(
            {
                "directoryBankMap": {"/src": "work"},
                "recallAdditionalBanks": ["other"],
                "retainEveryNTurns": 3,
            }
        )
        assert cfg["directoryBankMap"] == {"/src": "work"}
        assert cfg["recallAdditionalBanks"] == ["other"]
        assert cfg["retainEveryNTurns"] == 3

    def test_a_settings_file_with_invalid_utf8_falls_back_to_defaults(self):
        """UnicodeDecodeError subclasses ValueError, not OSError.

        Without it in the except clause a single bad byte escaped load_config()
        and stopped every hook — the opposite of the fallback this provides.
        """
        (self.root / "settings.json").write_bytes(b'{"recallBudget": "\xff\xfe high"}')

        cfg = load_config()

        assert cfg["recallBudget"] == "mid"
