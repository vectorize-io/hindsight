"""Tests for install.py — Cursor CLI Hindsight integration installer."""

import json
from pathlib import Path

import pytest

from hindsight_cursor_cli import install


@pytest.fixture()
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate HOME so install/uninstall touch only tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
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
        "beforeSubmitPrompt",
        "stop",
        "sessionEnd",
    }


# ---------------------------------------------------------------------------
# run_install — full install
# ---------------------------------------------------------------------------


def test_install_copies_scripts_and_lib(fake_home: Path) -> None:
    install.run_install()
    scripts = fake_home / ".cursor" / "hooks" / "cursor-cli" / "scripts"
    assert (scripts / "recall.py").exists()
    assert (scripts / "lib" / "client.py").exists()


def test_install_writes_settings_with_version(fake_home: Path) -> None:
    install.run_install()
    settings_path = fake_home / ".cursor" / "hooks" / "cursor-cli" / "settings.json"
    settings = json.loads(settings_path.read_text())
    # Version is stamped from package metadata, not the shipped template.
    assert "version" in settings
    assert settings["bankId"] == "cursor-cli"


def test_install_registers_hooks(fake_home: Path) -> None:
    install.run_install()
    registry = json.loads((fake_home / ".cursor" / "hooks.json").read_text())
    assert "beforeSubmitPrompt" in registry["hooks"]
    cmd = registry["hooks"]["beforeSubmitPrompt"][0]["command"]
    assert str(fake_home) in cmd  # absolute path to the installed script


def test_install_seeds_user_config(fake_home: Path) -> None:
    install.run_install(api_url="https://api.example.com", api_token="hsk_x")
    cfg = json.loads((fake_home / ".hindsight" / "cursor-cli.json").read_text())
    assert cfg["hindsightApiUrl"] == "https://api.example.com"
    assert cfg["hindsightApiToken"] == "hsk_x"


def test_install_preserves_existing_user_config(fake_home: Path) -> None:
    user_config = fake_home / ".hindsight" / "cursor-cli.json"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(json.dumps({"hindsightApiToken": "keep-me"}))

    install.run_install(api_url="https://override.example.com")

    cfg = json.loads(user_config.read_text())
    assert cfg == {"hindsightApiToken": "keep-me"}  # untouched


# ---------------------------------------------------------------------------
# merge — preserve foreign hooks, stay idempotent
# ---------------------------------------------------------------------------


def test_merge_preserves_foreign_hooks(fake_home: Path) -> None:
    hooks_json = fake_home / ".cursor" / "hooks.json"
    hooks_json.parent.mkdir(parents=True)
    hooks_json.write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {"stop": [{"command": "echo other", "timeout": 1}]},
            }
        )
    )

    install.run_install()

    registry = json.loads(hooks_json.read_text())
    stop_cmds = [d["command"] for d in registry["hooks"]["stop"]]
    assert "echo other" in stop_cmds  # foreign hook preserved
    assert any("retain.py" in c for c in stop_cmds)  # ours added


def test_reinstall_does_not_duplicate(fake_home: Path) -> None:
    install.run_install()
    install.run_install()
    registry = json.loads((fake_home / ".cursor" / "hooks.json").read_text())
    hindsight_stop = [d for d in registry["hooks"]["stop"] if "cursor-cli" in json.dumps(d)]
    assert len(hindsight_stop) == 1


# ---------------------------------------------------------------------------
# uninstall
# ---------------------------------------------------------------------------


def test_uninstall_removes_scripts_and_strips_hooks(fake_home: Path) -> None:
    install.run_install()
    install.run_uninstall()

    assert not (fake_home / ".cursor" / "hooks" / "cursor-cli").exists()
    registry = json.loads((fake_home / ".cursor" / "hooks.json").read_text())
    assert all("cursor-cli" not in json.dumps(d) for defs in registry["hooks"].values() for d in defs)


def test_uninstall_preserves_user_config(fake_home: Path) -> None:
    install.run_install(api_url="https://api.example.com")
    install.run_uninstall()
    assert (fake_home / ".hindsight" / "cursor-cli.json").exists()


def test_uninstall_preserves_foreign_hooks(fake_home: Path) -> None:
    hooks_json = fake_home / ".cursor" / "hooks.json"
    hooks_json.parent.mkdir(parents=True)
    hooks_json.write_text(json.dumps({"version": 1, "hooks": {"stop": [{"command": "echo other"}]}}))
    install.run_install()
    install.run_uninstall()

    registry = json.loads(hooks_json.read_text())
    assert registry["hooks"]["stop"] == [{"command": "echo other"}]
