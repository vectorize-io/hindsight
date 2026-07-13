"""Tests for the Devin Local SessionStart auto-recall hook."""

import json

from hindsight_devin_desktop import devin_local, hook


def _config(tmp_path, token="hsk_x", bank="devin-desktop"):
    p = tmp_path / "config.json"
    p.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "hindsight": {
                        "url": "http://localhost:8888/mcp/",
                        "transport": "http",
                        "headers": {"Authorization": f"Bearer {token}", "X-Bank-Id": bank},
                    }
                }
            }
        )
    )
    return p


def test_extract_connection():
    data = json.loads(_config_text())
    url, token, bank = hook._extract_connection(data)
    assert url == "http://localhost:8888/mcp/"
    assert token == "hsk_x"
    assert bank == "devin-desktop"


def _config_text():
    return json.dumps(
        {
            "mcpServers": {
                "hindsight": {
                    "url": "http://localhost:8888/mcp/",
                    "headers": {"Authorization": "Bearer hsk_x", "X-Bank-Id": "devin-desktop"},
                }
            }
        }
    )


def test_extract_connection_missing():
    assert hook._extract_connection({}) == (None, None, None)


def test_context_loaded_both_sections():
    out = hook._context_loaded("proj-bank", ["uses Postgres 16"], "global-bank", ["prefers tabs"])
    assert "proj-bank" in out and "uses Postgres 16" in out
    assert "global-bank" in out and "prefers tabs" in out
    assert "PRELOADED" in out  # names the session-start preload
    assert "telling the user" in out.lower()  # instructs the model to surface it
    assert "preloaded 2 memories" in out.lower()  # 1 project + 1 user


def test_context_loaded_project_only():
    out = hook._context_loaded("proj-bank", ["uses Postgres 16"], "global-bank", [])
    assert "uses Postgres 16" in out
    assert "About the user" not in out


def test_cmd_recall_no_config_emits_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(devin_local, "default_config_path", lambda: tmp_path / "missing.json")
    assert hook.cmd_recall() == 0
    assert capsys.readouterr().out == ""


def test_cmd_recall_emits_additional_context(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(devin_local, "default_config_path", lambda: _config(tmp_path))
    monkeypatch.setenv("DEVIN_PROJECT_DIR", str(tmp_path))
    # Avoid any network: fake recall returns bank-specific memories.
    monkeypatch.setattr(hook, "_recall", lambda url, token, bank: ["mem in " + bank])
    assert hook.cmd_recall() == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "mem in devin-desktop" in payload["hookSpecificOutput"]["additionalContext"]


def test_cmd_recall_empty_reports_status(tmp_path, monkeypatch, capsys):
    # No silent failures: an empty bank still reports "empty", not nothing.
    monkeypatch.setattr(devin_local, "default_config_path", lambda: _config(tmp_path))
    monkeypatch.setenv("DEVIN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(hook, "_recall", lambda url, token, bank: [])
    assert hook.cmd_recall() == 0
    ctx = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "nothing stored" in ctx.lower() and "session start" in ctx.lower()


def test_cmd_recall_error_reports_status(tmp_path, monkeypatch, capsys):
    # No silent failures: a recall exception surfaces a warning, still exits 0.
    monkeypatch.setattr(devin_local, "default_config_path", lambda: _config(tmp_path))
    monkeypatch.setenv("DEVIN_PROJECT_DIR", str(tmp_path))

    def boom(url, token, bank):
        raise ConnectionError("down")

    monkeypatch.setattr(hook, "_recall", boom)
    assert hook.cmd_recall() == 0
    ctx = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "unavailable" in ctx.lower()


def test_recall_local_only_skips_global(tmp_path, monkeypatch):
    monkeypatch.setattr(devin_local, "default_config_path", lambda: _config(tmp_path))
    monkeypatch.setenv("DEVIN_PROJECT_DIR", str(tmp_path))
    banks = []
    monkeypatch.setattr(hook, "_recall", lambda url, token, bank: banks.append(bank) or [])
    hook.cmd_recall(local_only=True)
    # only the derived project bank is recalled, not the shared "devin-desktop" bank
    assert "devin-desktop" not in banks
    assert any(b.startswith("devin-desktop-") for b in banks)


def test_retain_nudge_local_only_one_bank(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(devin_local, "default_config_path", lambda: _config(tmp_path))
    monkeypatch.setenv("DEVIN_PROJECT_DIR", str(tmp_path))
    hook.cmd_retain_nudge({"stop_hook_active": False}, local_only=True)
    reason = json.loads(capsys.readouterr().out)["reason"]
    assert "USER facts" not in reason  # not the two-tier routing phrasing
    assert "retain" in reason.lower()


def test_retain_nudge_blocks_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(devin_local, "default_config_path", lambda: _config(tmp_path))
    monkeypatch.setenv("DEVIN_PROJECT_DIR", str(tmp_path))
    # First stop: block with a retain instruction.
    assert hook.cmd_retain_nudge({"stop_hook_active": False}) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "block"
    assert "retain" in payload["reason"].lower()


def test_retain_nudge_no_loop_when_active(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(devin_local, "default_config_path", lambda: _config(tmp_path))
    # Already nudged this turn -> allow stop, emit nothing (no loop).
    assert hook.cmd_retain_nudge({"stop_hook_active": True}) == 0
    assert capsys.readouterr().out == ""


def test_banner_prints_for_hindsight_tool(capsys):
    hook.cmd_banner({"tool_info": {"mcp_server_name": "hindsight", "mcp_tool_name": "recall"}})
    out = capsys.readouterr().out
    assert "Hindsight" in out and "recall" in out


def test_banner_silent_for_other_server(capsys):
    hook.cmd_banner({"tool_info": {"mcp_server_name": "github", "mcp_tool_name": "list_commits"}})
    assert capsys.readouterr().out == ""


def test_main_recall_never_raises(tmp_path, monkeypatch):
    # Even if recall blows up, the hook must exit 0 (never break a session).
    monkeypatch.setattr(devin_local, "default_config_path", lambda: _config(tmp_path))
    monkeypatch.setenv("DEVIN_PROJECT_DIR", str(tmp_path))

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(hook, "_recall", boom)
    assert hook.main(["recall"]) == 0
