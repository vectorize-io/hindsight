"""Installer: copies hooks (executable, with shebang) + lib + settings."""

import json
import os
import sys
from pathlib import Path

# install.py lives at the integration root (one above hooks/).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import install  # noqa: E402

HOOK_FILES = ["TaskStart", "UserPromptSubmit", "TaskComplete", "TaskCancel"]


def test_install_copies_executable_hooks_with_shebang(tmp_path):
    hooks_dir = tmp_path / ".clinerules" / "hooks"
    install.install_hooks(hooks_dir)

    for name in HOOK_FILES:
        f = hooks_dir / name
        assert f.exists(), f"{name} not installed"
        assert os.access(f, os.X_OK), f"{name} is not executable"
        assert f.read_text().splitlines()[0] == "#!/usr/bin/env python3"


def test_install_copies_lib_and_settings(tmp_path):
    hooks_dir = tmp_path / ".clinerules" / "hooks"
    install.install_hooks(hooks_dir)
    assert (hooks_dir / "lib" / "client.py").exists()
    assert (hooks_dir / "lib" / "hooks_impl.py").exists()
    assert (hooks_dir / "settings.json").exists()


def test_write_user_config_records_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    install.write_user_config("https://api.example.com/", "secret-key")
    cfg = json.loads((tmp_path / ".hindsight" / "cline.json").read_text())
    assert cfg["hindsightApiUrl"] == "https://api.example.com"  # trailing slash stripped
    assert cfg["hindsightApiToken"] == "secret-key"


def test_write_user_config_noop_without_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    install.write_user_config("", "")
    assert not (tmp_path / ".hindsight" / "cline.json").exists()


def test_hooks_dir_paths():
    project = install.get_hooks_dir(Path("/proj"), global_install=False)
    assert project == Path("/proj") / ".clinerules" / "hooks"
    glob = install.get_hooks_dir(Path("/proj"), global_install=True)
    assert glob.parts[-3:] == ("Cline", "Rules", "Hooks")
