"""Tests for the hindsight-cursor CLI."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from hindsight_cursor.cli import (
    _PLUGIN_FILES,
    _hook_interpreter,
    _plugin_data_dir,
    _project_hooks_block,
    cmd_init,
    cmd_uninstall,
)


@pytest.fixture()
def fake_plugin_data(tmp_path):
    """A complete stand-in payload: every file the CLI declares it ships.

    Built from ``_PLUGIN_FILES`` rather than a hand-picked subset so a new
    entry in that list cannot silently go untested — and so the fixture keeps
    exercising the success path of ``_copy_plugin``, which now aborts on an
    incomplete payload.
    """
    data_dir = tmp_path / "plugin_data"
    for rel in _PLUGIN_FILES:
        path = data_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel}")
    (data_dir / ".cursor-plugin" / "plugin.json").write_text('{"name": "test"}')
    (data_dir / "hooks" / "hooks.json").write_text('{"version": 1}')
    (data_dir / "settings.json").write_text('{"bankId": "cursor"}')
    return data_dir


class _Args:
    """Minimal namespace for testing CLI commands."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestInit:
    def test_installs_plugin_files(self, tmp_path, fake_plugin_data):
        project = tmp_path / "my-project"
        project.mkdir()

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=False)
            )

        dest = project / ".cursor-plugin" / "hindsight-memory"
        assert dest.exists()
        assert (dest / ".cursor-plugin" / "plugin.json").exists()
        assert (dest / "hooks" / "hooks.json").exists()
        assert (dest / "settings.json").exists()
        assert (dest / "scripts" / "session_start.py").exists()
        assert (dest / "scripts" / "lib" / "rules_file.py").exists()
        assert (dest / "rules" / "hindsight-memory.mdc").exists()

    def test_refuses_overwrite_without_force(self, tmp_path, fake_plugin_data, capsys):
        project = tmp_path / "my-project"
        dest = project / ".cursor-plugin" / "hindsight-memory"
        dest.mkdir(parents=True)

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=False)
            )

        out = capsys.readouterr().out
        assert "already installed" in out
        assert "--force" in out

    def test_force_overwrites(self, tmp_path, fake_plugin_data):
        project = tmp_path / "my-project"
        dest = project / ".cursor-plugin" / "hindsight-memory"
        dest.mkdir(parents=True)
        (dest / "old-file.txt").write_text("old")

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(project=str(project), force=True, api_url=None, api_token=None, bank_id="cursor", no_mcp=False)
            )

        assert (dest / "hooks" / "hooks.json").exists()

    def test_creates_user_config(self, tmp_path, fake_plugin_data):
        project = tmp_path / "my-project"
        project.mkdir()
        config_dir = tmp_path / "hindsight-config"
        config_file = config_dir / "cursor.json"

        with (
            patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data),
            patch("hindsight_cursor.cli._USER_CONFIG_DIR", config_dir),
            patch("hindsight_cursor.cli._USER_CONFIG_FILE", config_file),
        ):
            cmd_init(
                _Args(
                    project=str(project),
                    force=False,
                    api_url="https://api.hindsight.vectorize.io",
                    api_token="tok_123",
                    bank_id="my-bank",
                    no_mcp=False,
                )
            )

        assert config_file.exists()
        cfg = json.loads(config_file.read_text())
        assert cfg["hindsightApiUrl"] == "https://api.hindsight.vectorize.io"
        assert cfg["hindsightApiToken"] == "tok_123"
        assert cfg["bankId"] == "my-bank"

    def test_skips_config_if_exists(self, tmp_path, fake_plugin_data, capsys):
        project = tmp_path / "my-project"
        project.mkdir()
        config_dir = tmp_path / "hindsight-config"
        config_file = config_dir / "cursor.json"
        config_dir.mkdir()
        config_file.write_text('{"bankId": "existing"}')

        with (
            patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data),
            patch("hindsight_cursor.cli._USER_CONFIG_DIR", config_dir),
            patch("hindsight_cursor.cli._USER_CONFIG_FILE", config_file),
        ):
            cmd_init(
                _Args(
                    project=str(project),
                    force=False,
                    api_url="http://x",
                    api_token=None,
                    bank_id="cursor",
                    no_mcp=False,
                )
            )

        # Original config should be untouched
        cfg = json.loads(config_file.read_text())
        assert cfg["bankId"] == "existing"
        assert "already exists" in capsys.readouterr().out

    def test_defaults_to_cwd(self, tmp_path, fake_plugin_data, monkeypatch):
        monkeypatch.chdir(tmp_path)

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(_Args(project=".", force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=False))

        assert (tmp_path / ".cursor-plugin" / "hindsight-memory" / "settings.json").exists()

    def test_creates_mcp_config(self, tmp_path, fake_plugin_data):
        project = tmp_path / "my-project"
        project.mkdir()

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(
                    project=str(project),
                    force=False,
                    api_url="https://api.hindsight.vectorize.io",
                    api_token="tok_123",
                    bank_id="my-bank",
                    no_mcp=False,
                )
            )

        mcp_file = project / ".cursor" / "mcp.json"
        assert mcp_file.exists()
        mcp = json.loads(mcp_file.read_text())
        assert "hindsight" in mcp["mcpServers"]
        assert mcp["mcpServers"]["hindsight"]["url"] == "https://api.hindsight.vectorize.io/mcp/my-bank/"
        assert mcp["mcpServers"]["hindsight"]["headers"]["Authorization"] == "Bearer tok_123"

    def test_skips_mcp_without_api_url(self, tmp_path, fake_plugin_data):
        project = tmp_path / "my-project"
        project.mkdir()

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=False)
            )

        mcp_file = project / ".cursor" / "mcp.json"
        assert not mcp_file.exists()

    def test_no_mcp_flag(self, tmp_path, fake_plugin_data):
        project = tmp_path / "my-project"
        project.mkdir()

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(
                    project=str(project),
                    force=False,
                    api_url="https://api.hindsight.vectorize.io",
                    api_token=None,
                    bank_id="cursor",
                    no_mcp=True,
                )
            )

        mcp_file = project / ".cursor" / "mcp.json"
        assert not mcp_file.exists()

    def test_merges_with_existing_mcp_config(self, tmp_path, fake_plugin_data):
        project = tmp_path / "my-project"
        project.mkdir()
        mcp_dir = project / ".cursor"
        mcp_dir.mkdir()
        (mcp_dir / "mcp.json").write_text(json.dumps({"mcpServers": {"other-server": {"url": "http://other"}}}))

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(
                    project=str(project),
                    force=False,
                    api_url="http://localhost:8888",
                    api_token=None,
                    bank_id="cursor",
                    no_mcp=False,
                )
            )

        mcp = json.loads((mcp_dir / "mcp.json").read_text())
        assert "other-server" in mcp["mcpServers"]
        assert "hindsight" in mcp["mcpServers"]

    def test_copies_rules_file(self, tmp_path, fake_plugin_data):
        """session_start imports lib.rules_file — init must ship it (#3864)."""
        project = tmp_path / "my-project"
        project.mkdir()

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=False)
            )

        assert (project / ".cursor-plugin" / "hindsight-memory" / "scripts" / "lib" / "rules_file.py").exists()

    def test_plugin_files_lists_rules_file(self):
        assert "scripts/lib/rules_file.py" in _PLUGIN_FILES

    def test_writes_project_hooks_json(self, tmp_path, fake_plugin_data):
        project = tmp_path / "my-project"
        project.mkdir()

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=False)
            )

        hooks_file = project / ".cursor" / "hooks.json"
        assert hooks_file.exists()
        hooks = json.loads(hooks_file.read_text())
        assert hooks["version"] == 1
        assert "sessionStart" in hooks["hooks"]
        assert "stop" in hooks["hooks"]
        session_cmd = hooks["hooks"]["sessionStart"][0]["command"]
        stop_cmd = hooks["hooks"]["stop"][0]["command"]
        assert "session_start.py" in session_cmd
        assert "retain.py" in stop_cmd
        assert ".cursor-plugin/hindsight-memory" in session_cmd
        assert "${CURSOR_PLUGIN_ROOT}" not in session_cmd

    def test_merges_with_existing_hooks_json(self, tmp_path, fake_plugin_data):
        project = tmp_path / "my-project"
        project.mkdir()
        cursor_dir = project / ".cursor"
        cursor_dir.mkdir()
        (cursor_dir / "hooks.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "hooks": {
                        "sessionStart": [{"command": "echo other-start", "timeout": 5}],
                        "stop": [{"command": "echo other-stop"}],
                    },
                }
            )
        )

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=False)
            )

        hooks = json.loads((cursor_dir / "hooks.json").read_text())
        session = hooks["hooks"]["sessionStart"]
        stop = hooks["hooks"]["stop"]
        assert any("echo other-start" in d.get("command", "") for d in session)
        assert any("hindsight-memory" in d.get("command", "") for d in session)
        assert any("echo other-stop" in d.get("command", "") for d in stop)
        assert any("retain.py" in d.get("command", "") for d in stop)

    def test_hooks_merge_is_idempotent(self, tmp_path, fake_plugin_data):
        project = tmp_path / "my-project"
        project.mkdir()

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=True)
            )
            cmd_init(
                _Args(project=str(project), force=True, api_url=None, api_token=None, bank_id="cursor", no_mcp=True)
            )

        hooks = json.loads((project / ".cursor" / "hooks.json").read_text())
        assert len(hooks["hooks"]["sessionStart"]) == 1
        assert len(hooks["hooks"]["stop"]) == 1


class TestUninstall:
    def test_removes_plugin(self, tmp_path, fake_plugin_data):
        project = tmp_path / "my-project"
        dest = project / ".cursor-plugin" / "hindsight-memory"
        dest.mkdir(parents=True)
        (dest / "something.txt").write_text("x")

        cmd_uninstall(_Args(project=str(project)))

        assert not dest.exists()

    def test_noop_if_not_installed(self, tmp_path, capsys):
        project = tmp_path / "my-project"
        project.mkdir()

        cmd_uninstall(_Args(project=str(project)))

        assert "not found" in capsys.readouterr().out

    def test_cleans_up_mcp_config(self, tmp_path):
        project = tmp_path / "my-project"
        dest = project / ".cursor-plugin" / "hindsight-memory"
        dest.mkdir(parents=True)
        (dest / "x.txt").write_text("x")

        mcp_dir = project / ".cursor"
        mcp_dir.mkdir()
        (mcp_dir / "mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "hindsight": {"url": "http://localhost:8888/mcp/cursor/"},
                        "other": {"url": "http://other"},
                    }
                }
            )
        )

        cmd_uninstall(_Args(project=str(project)))

        mcp = json.loads((mcp_dir / "mcp.json").read_text())
        assert "hindsight" not in mcp["mcpServers"]
        assert "other" in mcp["mcpServers"]

    def test_cleans_up_hooks_json(self, tmp_path):
        project = tmp_path / "my-project"
        dest = project / ".cursor-plugin" / "hindsight-memory"
        dest.mkdir(parents=True)
        (dest / "x.txt").write_text("x")

        cursor_dir = project / ".cursor"
        cursor_dir.mkdir()
        (cursor_dir / "hooks.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "hooks": {
                        "sessionStart": [
                            {"command": "echo other"},
                            {"command": "python3 .cursor-plugin/hindsight-memory/scripts/session_start.py"},
                        ],
                        "stop": [
                            {"command": "python3 .cursor-plugin/hindsight-memory/scripts/retain.py"},
                        ],
                    },
                }
            )
        )

        cmd_uninstall(_Args(project=str(project)))

        hooks = json.loads((cursor_dir / "hooks.json").read_text())
        assert hooks["hooks"]["sessionStart"] == [{"command": "echo other"}]
        assert "stop" not in hooks["hooks"]


class TestSessionEndHook:
    """`stop` is turn-window gated; sessionEnd is the flush that can't be."""

    def test_registers_retain_on_both_stop_and_session_end(self, tmp_path, fake_plugin_data):
        project = tmp_path / "proj"
        project.mkdir()

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=True)
            )

        hooks = json.loads((project / ".cursor" / "hooks.json").read_text())["hooks"]
        assert set(hooks) == {"sessionStart", "stop", "sessionEnd"}
        assert "retain.py" in hooks["stop"][0]["command"]
        assert "retain.py" in hooks["sessionEnd"][0]["command"]
        assert "session_start.py" in hooks["sessionStart"][0]["command"]

    def test_uninstall_strips_session_end_too(self, tmp_path, fake_plugin_data):
        project = tmp_path / "proj"
        project.mkdir()

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=True)
            )
        cmd_uninstall(_Args(project=str(project)))

        assert not (project / ".cursor" / "hooks.json").exists()


class TestHookInterpreter:
    """Windows has no `python3` on PATH -- a hardcoded one is a silent no-op."""

    def test_posix_uses_python3(self):
        with patch("hindsight_cursor.cli.sys.platform", "darwin"):
            assert _hook_interpreter() == "python3"

    def test_windows_uses_python(self):
        with patch("hindsight_cursor.cli.sys.platform", "win32"):
            assert _hook_interpreter() == "python"

    def test_commands_use_the_platform_interpreter(self):
        with patch("hindsight_cursor.cli.sys.platform", "win32"):
            hooks = _project_hooks_block()["hooks"]
        for definitions in hooks.values():
            for definition in definitions:
                assert definition["command"].startswith("python ")


class TestIncompletePayload:
    """A partial copy is never useful -- every hook would ImportError."""

    def test_init_aborts_when_bundled_files_are_missing(self, tmp_path, fake_plugin_data, capsys):
        (fake_plugin_data / "scripts" / "lib" / "rules_file.py").unlink()
        project = tmp_path / "proj"
        project.mkdir()

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            with pytest.raises(SystemExit) as exc:
                cmd_init(
                    _Args(
                        project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=True
                    )
                )

        assert exc.value.code == 1
        assert "scripts/lib/rules_file.py" in capsys.readouterr().err
        # No half-installed state, and no hooks pointing at scripts that are absent.
        assert not (project / ".cursor" / "hooks.json").exists()


class TestSessionRulesCleanup:
    """sessionStart writes an alwaysApply rules file -- uninstall must remove it."""

    def _install(self, project, fake_plugin_data):
        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(
                _Args(project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=True)
            )

    def test_removes_rules_file_and_gitignore_entry(self, tmp_path, fake_plugin_data):
        project = tmp_path / "proj"
        project.mkdir()
        self._install(project, fake_plugin_data)

        rules = project / ".cursor" / "rules" / "hindsight-session.mdc"
        rules.parent.mkdir(parents=True, exist_ok=True)
        rules.write_text("---\nalwaysApply: true\n---\nold memories\n")
        (project / ".gitignore").write_text(
            "node_modules/\n\n"
            "# Added by hindsight-cursor plugin — session rules are regenerated each session.\n"
            "/.cursor/rules/hindsight-session.mdc\n"
        )

        cmd_uninstall(_Args(project=str(project)))

        assert not rules.exists()
        assert (project / ".gitignore").read_text() == "node_modules/\n"

    def test_leaves_unrelated_gitignore_untouched(self, tmp_path, fake_plugin_data):
        project = tmp_path / "proj"
        project.mkdir()
        self._install(project, fake_plugin_data)
        (project / ".gitignore").write_text("node_modules/\ndist/\n")

        cmd_uninstall(_Args(project=str(project)))

        assert (project / ".gitignore").read_text() == "node_modules/\ndist/\n"

    def test_tolerates_missing_rules_file(self, tmp_path, fake_plugin_data):
        project = tmp_path / "proj"
        project.mkdir()
        self._install(project, fake_plugin_data)
        cmd_uninstall(_Args(project=str(project)))  # must not raise
