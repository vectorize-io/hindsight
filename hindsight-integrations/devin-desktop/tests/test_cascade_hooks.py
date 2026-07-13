"""Tests for the Cascade visibility banner hook (hooks.json)."""

import json

from hindsight_devin_desktop.cascade_hooks import (
    EVENT,
    apply_banner,
    banner_command,
    default_hooks_path,
    is_installed,
    remove_banner,
)


def test_default_path():
    p = default_hooks_path()
    assert p.name == "hooks.json"
    assert p.parent.name == "windsurf"
    assert p.parent.parent.name == ".codeium"


def test_banner_command_targets_module():
    assert "hindsight_devin_desktop.hook banner" in banner_command()


def test_creates_with_show_output(tmp_path):
    p = tmp_path / "hooks.json"
    result = apply_banner(p)
    assert result.action == "created"
    entry = json.loads(p.read_text())["hooks"][EVENT][0]
    assert entry["show_output"] is True
    assert "hindsight_devin_desktop.hook banner" in entry["command"]
    assert "command" in entry and "powershell" in entry  # cross-platform
    assert is_installed(p)


def test_idempotent(tmp_path):
    p = tmp_path / "hooks.json"
    apply_banner(p)
    assert apply_banner(p).action == "unchanged"
    assert len(json.loads(p.read_text())["hooks"][EVENT]) == 1


def test_preserves_user_hooks(tmp_path):
    p = tmp_path / "hooks.json"
    p.write_text(json.dumps({"hooks": {EVENT: [{"command": "echo hi", "show_output": True}], "pre_write_code": []}}))
    apply_banner(p)
    data = json.loads(p.read_text())
    cmds = [e["command"] for e in data["hooks"][EVENT]]
    assert "echo hi" in cmds  # user's kept
    assert any("hindsight_devin_desktop.hook banner" in c for c in cmds)  # ours added
    assert "pre_write_code" in data["hooks"]  # untouched


def test_non_json_returns_manual(tmp_path):
    p = tmp_path / "hooks.json"
    p.write_text("{ not json")
    assert apply_banner(p).action == "manual"
    assert p.read_text() == "{ not json"  # untouched


def test_remove_keeps_user_hooks(tmp_path):
    p = tmp_path / "hooks.json"
    p.write_text(json.dumps({"hooks": {EVENT: [{"command": "echo hi"}]}}))
    apply_banner(p)
    remove_banner(p)
    cmds = [e["command"] for e in json.loads(p.read_text())["hooks"][EVENT]]
    assert "echo hi" in cmds
    assert not any("hindsight_devin_desktop.hook banner" in c for c in cmds)
    assert not is_installed(p)
