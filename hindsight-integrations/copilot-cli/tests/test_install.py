"""Tests for install.py — GitHub Copilot CLI Hindsight integration installer."""

import json
from pathlib import Path

import pytest

from hindsight_copilot_cli import install


@pytest.fixture()
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate HOME (and cwd, for repo-scope tests) so install/uninstall touch only tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("COPILOT_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# render_hooks_block
# ---------------------------------------------------------------------------


def test_render_hooks_substitutes_scripts_dir() -> None:
    block = install.render_hooks_block(Path("/opt/hooks/scripts"))
    commands = json.dumps(block)
    assert install.SCRIPTS_PLACEHOLDER not in commands
    assert "/opt/hooks/scripts" in commands


def test_render_hooks_covers_all_events() -> None:
    block = install.render_hooks_block(Path("/opt/hooks/scripts"))
    assert set(block["hooks"]) == {
        "sessionStart",
        "subagentStart",
        "agentStop",
        "sessionEnd",
    }


# ---------------------------------------------------------------------------
# run_install — user scope (default)
# ---------------------------------------------------------------------------


def test_install_copies_scripts_and_lib(fake_home: Path) -> None:
    install.run_install()
    scripts = fake_home / ".copilot" / "hindsight-copilot-cli" / "scripts"
    assert (scripts / "session_start.py").exists()
    assert (scripts / "agent_stop.py").exists()
    assert (scripts / "lib" / "client.py").exists()


def test_install_writes_settings_with_version(fake_home: Path) -> None:
    install.run_install()
    settings_path = fake_home / ".copilot" / "hindsight-copilot-cli" / "settings.json"
    settings = json.loads(settings_path.read_text())
    # Version is stamped from package metadata, not the shipped template.
    assert "version" in settings
    assert settings["bankId"] == "copilot-cli"


def test_install_registers_hooks_at_user_scope(fake_home: Path) -> None:
    install.run_install()
    registry_path = fake_home / ".copilot" / "hooks" / "hindsight-copilot-cli.json"
    assert registry_path.exists()
    registry = json.loads(registry_path.read_text())
    assert "sessionStart" in registry["hooks"]
    assert "subagentStart" in registry["hooks"]
    cmd = registry["hooks"]["sessionStart"][0]["bash"]
    assert str(fake_home) in cmd  # absolute path to the installed script


def test_install_respects_copilot_home_env(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom_home = fake_home / "custom-copilot-home"
    monkeypatch.setenv("COPILOT_HOME", str(custom_home))
    install.run_install()
    assert (custom_home / "hooks" / "hindsight-copilot-cli.json").exists()
    assert (custom_home / "hindsight-copilot-cli" / "scripts" / "session_start.py").exists()


def test_install_seeds_user_config(fake_home: Path) -> None:
    install.run_install(api_url="https://api.example.com", api_token="hsk_x")
    cfg = json.loads((fake_home / ".hindsight" / "copilot-cli.json").read_text())
    assert cfg["hindsightApiUrl"] == "https://api.example.com"
    assert cfg["hindsightApiToken"] == "hsk_x"


def test_install_preserves_existing_user_config(fake_home: Path) -> None:
    user_config = fake_home / ".hindsight" / "copilot-cli.json"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(json.dumps({"hindsightApiToken": "keep-me"}))

    install.run_install(api_url="https://override.example.com")

    cfg = json.loads(user_config.read_text())
    assert cfg == {"hindsightApiToken": "keep-me"}  # untouched


def test_reinstall_overwrites_registry_without_duplicating(fake_home: Path) -> None:
    install.run_install()
    install.run_install()
    registry = json.loads((fake_home / ".copilot" / "hooks" / "hindsight-copilot-cli.json").read_text())
    assert len(registry["hooks"]["sessionStart"]) == 1


# ---------------------------------------------------------------------------
# run_install — repo scope
# ---------------------------------------------------------------------------


def test_install_repo_scope_writes_github_hooks(fake_home: Path) -> None:
    install.run_install(scope="repo")
    registry_path = fake_home / ".github" / "hooks" / "hindsight-copilot-cli.json"
    assert registry_path.exists()
    registry = json.loads(registry_path.read_text())
    assert "agentStop" in registry["hooks"]


def test_install_repo_scope_does_not_write_user_registry(fake_home: Path) -> None:
    install.run_install(scope="repo")
    assert not (fake_home / ".copilot" / "hooks" / "hindsight-copilot-cli.json").exists()


def test_install_repo_scope_still_installs_scripts_locally(fake_home: Path) -> None:
    """Scripts always live under ~/.copilot — only the registration differs by scope."""
    install.run_install(scope="repo")
    scripts = fake_home / ".copilot" / "hindsight-copilot-cli" / "scripts"
    assert (scripts / "agent_stop.py").exists()


def test_install_invalid_scope_raises(fake_home: Path) -> None:
    with pytest.raises(ValueError):
        install.run_install(scope="bogus")


# ---------------------------------------------------------------------------
# uninstall
# ---------------------------------------------------------------------------


def test_uninstall_removes_user_registry_and_scripts(fake_home: Path) -> None:
    install.run_install()
    install.run_uninstall()

    assert not (fake_home / ".copilot" / "hooks" / "hindsight-copilot-cli.json").exists()
    assert not (fake_home / ".copilot" / "hindsight-copilot-cli").exists()


def test_uninstall_preserves_user_config(fake_home: Path) -> None:
    install.run_install(api_url="https://api.example.com")
    install.run_uninstall()
    assert (fake_home / ".hindsight" / "copilot-cli.json").exists()


def test_uninstall_repo_scope_removes_only_repo_registry(fake_home: Path) -> None:
    install.run_install(scope="repo")
    install.run_uninstall(scope="repo")
    assert not (fake_home / ".github" / "hooks" / "hindsight-copilot-cli.json").exists()


def test_uninstall_invalid_scope_raises(fake_home: Path) -> None:
    with pytest.raises(ValueError):
        install.run_uninstall(scope="bogus")


def test_uninstall_nonexistent_is_a_no_op(fake_home: Path) -> None:
    # Should not raise even though nothing was ever installed.
    install.run_uninstall()
