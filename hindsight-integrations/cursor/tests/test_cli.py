"""Tests for the hindsight-cursor CLI."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from hindsight_cursor.cli import _plugin_data_dir, cmd_init, cmd_uninstall


@pytest.fixture()
def fake_plugin_data(tmp_path):
    """Create a minimal set of plugin data files for testing."""
    data_dir = tmp_path / "plugin_data"
    # Create a few representative files
    (data_dir / ".cursor-plugin").mkdir(parents=True)
    (data_dir / ".cursor-plugin" / "plugin.json").write_text('{"name": "test"}')
    (data_dir / "hooks").mkdir()
    (data_dir / "hooks" / "hooks.json").write_text('{"version": 1}')
    (data_dir / "settings.json").write_text('{"bankId": "cursor"}')
    (data_dir / "scripts" / "lib").mkdir(parents=True)
    (data_dir / "scripts" / "lib" / "__init__.py").write_text("")
    (data_dir / "scripts" / "session_start.py").write_text("# session_start")
    (data_dir / "scripts" / "retain.py").write_text("# retain")
    (data_dir / "rules").mkdir()
    (data_dir / "rules" / "hindsight-memory.mdc").write_text("# rule")
    (data_dir / "skills" / "hindsight-recall").mkdir(parents=True)
    (data_dir / "skills" / "hindsight-recall" / "SKILL.md").write_text("# skill")
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
            cmd_init(_Args(project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=False))

        dest = project / ".cursor-plugin" / "hindsight-memory"
        assert dest.exists()
        assert (dest / ".cursor-plugin" / "plugin.json").exists()
        assert (dest / "hooks" / "hooks.json").exists()
        assert (dest / "settings.json").exists()
        assert (dest / "scripts" / "session_start.py").exists()
        assert (dest / "rules" / "hindsight-memory.mdc").exists()

    def test_refuses_overwrite_without_force(self, tmp_path, fake_plugin_data, capsys):
        project = tmp_path / "my-project"
        dest = project / ".cursor-plugin" / "hindsight-memory"
        dest.mkdir(parents=True)

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(_Args(project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=False))

        out = capsys.readouterr().out
        assert "already installed" in out
        assert "--force" in out

    def test_force_overwrites(self, tmp_path, fake_plugin_data):
        project = tmp_path / "my-project"
        dest = project / ".cursor-plugin" / "hindsight-memory"
        dest.mkdir(parents=True)
        (dest / "old-file.txt").write_text("old")

        with patch("hindsight_cursor.cli._plugin_data_dir", return_value=fake_plugin_data):
            cmd_init(_Args(project=str(project), force=True, api_url=None, api_token=None, bank_id="cursor", no_mcp=False))

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
            cmd_init(_Args(project=str(project), force=False, api_url="http://x", api_token=None, bank_id="cursor", no_mcp=False))

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
            cmd_init(_Args(project=str(project), force=False, api_url=None, api_token=None, bank_id="cursor", no_mcp=False))

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
        (mcp_dir / "mcp.json").write_text(json.dumps({
            "mcpServers": {"other-server": {"url": "http://other"}}
        }))

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
        (mcp_dir / "mcp.json").write_text(json.dumps({
            "mcpServers": {
                "hindsight": {"url": "http://localhost:8888/mcp/cursor/"},
                "other": {"url": "http://other"},
            }
        }))

        cmd_uninstall(_Args(project=str(project)))

        mcp = json.loads((mcp_dir / "mcp.json").read_text())
        assert "hindsight" not in mcp["mcpServers"]
        assert "other" in mcp["mcpServers"]
