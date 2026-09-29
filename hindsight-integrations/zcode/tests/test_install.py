"""Tests for install.py — ZCode Hindsight integration installer."""

import json
from pathlib import Path

import pytest

from hindsight_zcode import install


@pytest.fixture()
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate HOME so install/uninstall touch only tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def _config(home: Path) -> dict:
    return json.loads((home / ".zcode" / "cli" / "config.json").read_text())


def _events(config: dict) -> dict:
    return config["hooks"]["events"]


def _hindsight_defs(config: dict, event: str) -> list:
    """Hindsight hook definitions registered for an event."""
    return [d for d in _events(config).get(event, []) if "hooks/hindsight" in json.dumps(d)]


# ---------------------------------------------------------------------------
# render_hooks_events — ZCode-native hook schema
# ---------------------------------------------------------------------------


def test_render_hooks_substitutes_scripts_dir() -> None:
    events = install.render_hooks_events(Path("/opt/hooks/scripts"))
    commands = json.dumps(events)
    assert install.SCRIPTS_PLACEHOLDER not in commands
    assert "/opt/hooks/scripts" in commands


def test_render_hooks_covers_wired_events() -> None:
    events = install.render_hooks_events(Path("/opt/hooks/scripts"))
    # ZCode has no SessionEnd — retain rides Stop.
    assert set(events) == {"SessionStart", "UserPromptSubmit", "Stop"}


def test_render_hooks_uses_process_schema() -> None:
    events = install.render_hooks_events(Path("/opt/hooks/scripts"))
    inner = events["UserPromptSubmit"][0]["hooks"][0]
    # ZCode-native shape: process command + args[] + timeoutMs.
    assert inner["type"] == "process"
    assert inner["command"] == "python3"
    assert any("recall.py" in a for a in inner["args"])
    assert inner["timeoutMs"] == 12000


def test_render_hooks_no_async_key() -> None:
    events = install.render_hooks_events(Path("/opt/hooks/scripts"))
    inner = events["Stop"][0]["hooks"][0]
    # async has no effect in ZCode; hooks run inline.
    assert "async" not in inner
    assert inner["timeoutMs"] == 15000


# ---------------------------------------------------------------------------
# run_install — full install
# ---------------------------------------------------------------------------


def test_install_copies_scripts_and_lib(fake_home: Path) -> None:
    install.run_install()
    scripts = fake_home / ".zcode" / "hooks" / "hindsight" / "scripts"
    assert (scripts / "recall.py").exists()
    assert (scripts / "lib" / "client.py").exists()


def test_install_writes_settings_with_version(fake_home: Path) -> None:
    install.run_install()
    settings_path = fake_home / ".zcode" / "hooks" / "hindsight" / "settings.json"
    settings = json.loads(settings_path.read_text())
    # Version is stamped from package metadata, not the shipped template.
    assert "version" in settings
    assert settings["bankId"] == "zcode"


def test_install_registers_hooks_in_zcode_config(fake_home: Path) -> None:
    install.run_install()
    config = _config(fake_home)
    # Config hooks are disabled by default — install must force them on.
    assert config["hooks"]["enabled"] is True
    assert config["hooks"]["maxOutputBytes"] == install.MAX_OUTPUT_BYTES
    events = _events(config)
    assert "UserPromptSubmit" in events
    inner = events["UserPromptSubmit"][0]["hooks"][0]
    script_arg = inner["args"][0]
    assert str(fake_home) in script_arg  # absolute path to the installed script
    assert "hooks/hindsight" in script_arg


def test_install_seeds_user_config(fake_home: Path) -> None:
    install.run_install(api_url="https://api.example.com", api_token="hsk_x")
    cfg = json.loads((fake_home / ".hindsight" / "zcode.json").read_text())
    assert cfg["hindsightApiUrl"] == "https://api.example.com"
    assert cfg["hindsightApiToken"] == "hsk_x"


def test_install_preserves_existing_user_config(fake_home: Path) -> None:
    user_config = fake_home / ".hindsight" / "zcode.json"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(json.dumps({"hindsightApiToken": "keep-me"}))

    install.run_install(api_url="https://override.example.com")

    cfg = json.loads(user_config.read_text())
    assert cfg == {"hindsightApiToken": "keep-me"}  # untouched


# ---------------------------------------------------------------------------
# merge — preserve foreign config keys/hooks, stay idempotent
# ---------------------------------------------------------------------------


def test_merge_preserves_foreign_config_keys(fake_home: Path) -> None:
    config_json = fake_home / ".zcode" / "cli" / "config.json"
    config_json.parent.mkdir(parents=True)
    config_json.write_text(json.dumps({"model": "glm-4.6", "theme": "dark"}))

    install.run_install()

    config = _config(fake_home)
    # Non-hooks keys are preserved untouched.
    assert config["model"] == "glm-4.6"
    assert config["theme"] == "dark"
    assert "UserPromptSubmit" in _events(config)


def test_merge_preserves_foreign_hooks(fake_home: Path) -> None:
    config_json = fake_home / ".zcode" / "cli" / "config.json"
    config_json.parent.mkdir(parents=True)
    config_json.write_text(
        json.dumps(
            {
                "hooks": {
                    "enabled": True,
                    "maxOutputBytes": 4096,
                    "events": {
                        "Stop": [{"hooks": [{"type": "process", "command": "echo", "args": ["other"]}]}],
                    },
                }
            }
        )
    )

    install.run_install()

    config = _config(fake_home)
    stop_cmds = json.dumps(_events(config)["Stop"])
    assert "other" in stop_cmds  # foreign hook preserved
    assert "retain.py" in stop_cmds  # ours added
    # A pre-existing maxOutputBytes is respected, not clobbered.
    assert config["hooks"]["maxOutputBytes"] == 4096


def test_reinstall_does_not_duplicate(fake_home: Path) -> None:
    install.run_install()
    install.run_install()
    config = _config(fake_home)
    assert len(_hindsight_defs(config, "Stop")) == 1


# ---------------------------------------------------------------------------
# uninstall
# ---------------------------------------------------------------------------


def test_uninstall_removes_scripts_and_strips_hooks(fake_home: Path) -> None:
    install.run_install()
    install.run_uninstall()

    assert not (fake_home / ".zcode" / "hooks" / "hindsight").exists()
    config = _config(fake_home)
    events = config["hooks"].get("events", {})
    assert all("hooks/hindsight" not in json.dumps(d) for defs in events.values() for d in defs)


def test_uninstall_preserves_user_config(fake_home: Path) -> None:
    install.run_install(api_url="https://api.example.com")
    install.run_uninstall()
    assert (fake_home / ".hindsight" / "zcode.json").exists()


def test_uninstall_preserves_foreign_hooks(fake_home: Path) -> None:
    config_json = fake_home / ".zcode" / "cli" / "config.json"
    config_json.parent.mkdir(parents=True)
    foreign_stop = [{"hooks": [{"type": "process", "command": "echo", "args": ["other"]}]}]
    config_json.write_text(json.dumps({"hooks": {"events": {"Stop": foreign_stop}}}))
    install.run_install()
    install.run_uninstall()

    config = _config(fake_home)
    assert _events(config)["Stop"] == foreign_stop
