"""Installer: copies hooks (executable, with shebang) + lib + settings, and the CLI."""

import json
import os
from pathlib import Path

import pytest
from hindsight_cline import get_hooks_dir, install_hooks, write_user_config
from hindsight_cline.cli import main

HOOK_FILES = ["TaskStart", "UserPromptSubmit", "TaskComplete", "TaskCancel"]


def test_install_copies_executable_hooks_with_shebang(tmp_path):
    hooks_dir = tmp_path / ".clinerules" / "hooks"
    install_hooks(hooks_dir)

    for name in HOOK_FILES:
        f = hooks_dir / name
        assert f.exists(), f"{name} not installed"
        assert os.access(f, os.X_OK), f"{name} is not executable"
        assert f.read_text().splitlines()[0] == "#!/usr/bin/env python3"


def test_install_copies_lib_and_settings(tmp_path):
    hooks_dir = tmp_path / ".clinerules" / "hooks"
    install_hooks(hooks_dir)
    assert (hooks_dir / "lib" / "client.py").exists()
    assert (hooks_dir / "lib" / "hooks_impl.py").exists()
    assert (hooks_dir / "settings.json").exists()


def test_write_user_config_records_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_user_config("https://api.example.com/", "secret-key")
    cfg = json.loads((tmp_path / ".hindsight" / "cline.json").read_text())
    assert cfg["hindsightApiUrl"] == "https://api.example.com"  # trailing slash stripped
    assert cfg["hindsightApiToken"] == "secret-key"


def test_write_user_config_noop_without_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_user_config("", "")
    assert not (tmp_path / ".hindsight" / "cline.json").exists()


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_install_into_project(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = main(["install", "--project-dir", str(tmp_path), "--api-url", "https://x.example", "--api-token", "k"])
    assert rc == 0
    hooks_dir = tmp_path / ".clinerules" / "hooks"
    assert all((hooks_dir / name).exists() for name in HOOK_FILES)
    assert (hooks_dir / "settings.json").exists()
    cfg = json.loads((tmp_path / ".hindsight" / "cline.json").read_text())
    assert cfg["hindsightApiToken"] == "k"


def test_cli_uninstall_removes_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    main(["install", "--project-dir", str(tmp_path)])
    hooks_dir = tmp_path / ".clinerules" / "hooks"
    assert (hooks_dir / "TaskStart").exists()

    rc = main(["uninstall", "--project-dir", str(tmp_path)])
    assert rc == 0
    assert not (hooks_dir / "TaskStart").exists()
    assert not (hooks_dir / "lib").exists()


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


# ── Cline CLI hook layout (#2711) ─────────────────────────────────────────────


def test_get_hooks_dir_targets_are_client_specific():
    project = Path("/proj")
    # Extension (default) — the VS Code paths.
    assert get_hooks_dir(project, global_install=False) == project / ".clinerules" / "hooks"
    assert get_hooks_dir(project, global_install=True) == Path.home() / "Documents" / "Cline" / "Rules" / "Hooks"
    # CLI — the paths the cline CLI actually reads.
    assert get_hooks_dir(project, global_install=False, cli=True) == project / ".cline" / "hooks"
    assert get_hooks_dir(project, global_install=True, cli=True) == Path.home() / ".cline" / "hooks"


def test_cli_flag_installs_into_dot_cline_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = main(["install", "--cli", "--project-dir", str(tmp_path)])
    assert rc == 0
    cli_hooks = tmp_path / ".cline" / "hooks"
    assert all((cli_hooks / name).exists() for name in HOOK_FILES)
    assert (cli_hooks / "lib" / "hooks_impl.py").exists()
    assert (cli_hooks / "settings.json").exists()
    # The extension path must NOT be written when targeting the CLI.
    assert not (tmp_path / ".clinerules" / "hooks").exists()


def test_cli_global_flag_installs_into_home_dot_cline_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = main(["install", "--cli", "--global"])
    assert rc == 0
    assert (tmp_path / ".cline" / "hooks" / "TaskStart").exists()


def test_cli_uninstall_removes_dot_cline_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    main(["install", "--cli", "--project-dir", str(tmp_path)])
    cli_hooks = tmp_path / ".cline" / "hooks"
    assert (cli_hooks / "TaskStart").exists()

    rc = main(["uninstall", "--cli", "--project-dir", str(tmp_path)])
    assert rc == 0
    assert not (cli_hooks / "TaskStart").exists()
    assert not (cli_hooks / "lib").exists()
