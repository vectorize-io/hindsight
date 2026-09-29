"""Tests for lib/bank.py — bank ID derivation for the Cursor CLI integration."""

import os

from lib.bank import derive_bank_id


def _cfg(**overrides):
    base = {
        "dynamicBankId": False,
        "bankId": "cursor-cli",
        "bankIdPrefix": "",
        "agentName": "cursor-cli",
        "dynamicBankGranularity": ["agent", "project"],
        "bankMission": "",
        "retainMission": None,
    }
    base.update(overrides)
    return base


def _hook(session_id="sess-1", cwd="/home/user/myproject", workspace_roots=None):
    return {
        "session_id": session_id,
        "cwd": cwd,
        "workspace_roots": workspace_roots or [cwd],
    }


class TestDeriveBankIdStatic:
    def test_static_default_bank(self):
        assert derive_bank_id(_hook(), _cfg()) == "cursor-cli"

    def test_static_custom_bank_id(self):
        cfg = _cfg(bankId="my-agent")
        assert derive_bank_id(_hook(), cfg) == "my-agent"

    def test_static_with_prefix(self):
        cfg = _cfg(bankId="bot", bankIdPrefix="prod")
        assert derive_bank_id(_hook(), cfg) == "prod-bot"

    def test_static_prefix_without_bankid_uses_default(self):
        cfg = _cfg(bankId=None, bankIdPrefix="dev")
        assert derive_bank_id(_hook(), cfg) == "dev-cursor-cli"


class TestDeriveBankIdDynamic:
    def test_dynamic_agent_project(self, monkeypatch):
        monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
        cfg = _cfg(dynamicBankId=True, agentName="mybot", dynamicBankGranularity=["agent", "project"])
        result = derive_bank_id(_hook(cwd="/home/user/hindsight"), cfg)
        assert result == "mybot::hindsight"

    def test_dynamic_uses_cursor_project_dir_env(self, monkeypatch):
        monkeypatch.setenv("CURSOR_PROJECT_DIR", "/work/myapp")
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        result = derive_bank_id(_hook(cwd="/should/not/use"), cfg)
        assert "myapp" in result

    def test_dynamic_uses_workspace_roots_when_no_env(self, monkeypatch):
        monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        result = derive_bank_id(
            _hook(cwd="/x", workspace_roots=["/work/anotherapp"]),
            cfg,
        )
        assert "anotherapp" in result

    def test_dynamic_preserves_raw_special_chars(self, monkeypatch):
        monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        result = derive_bank_id(_hook(cwd="/home/user/my project"), cfg)
        assert "my project" in result
        assert "%" not in result

    def test_dynamic_preserves_raw_utf8(self, monkeypatch):
        monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        result = derive_bank_id(_hook(cwd="/home/user/мой проект"), cfg)
        assert "мой проект" in result
        assert "%" not in result

    def test_dynamic_session_field(self, monkeypatch):
        monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["session"])
        result = derive_bank_id(_hook(session_id="abc-123"), cfg)
        assert "abc-123" in result

    def test_dynamic_conversation_id_also_works(self, monkeypatch):
        monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["session"])
        hook = {"conversation_id": "conv-xyz", "cwd": "/x"}
        result = derive_bank_id(hook, cfg)
        assert "conv-xyz" in result

    def test_dynamic_with_prefix(self, monkeypatch):
        monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["agent"], bankIdPrefix="v2")
        result = derive_bank_id(_hook(), cfg)
        assert result.startswith("v2-")

    def test_dynamic_user_from_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_USER_ID", "user-456")
        monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["user"])
        result = derive_bank_id(_hook(), cfg)
        assert "user-456" in result

    def test_dynamic_missing_env_uses_default(self, monkeypatch):
        monkeypatch.delenv("HINDSIGHT_USER_ID", raising=False)
        monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["user"])
        result = derive_bank_id(_hook(), cfg)
        assert "anonymous" in result

    def test_dynamic_empty_cwd_uses_unknown(self, monkeypatch):
        monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        result = derive_bank_id({"session_id": "s", "cwd": ""}, cfg)
        assert "unknown" in result

    def test_dynamic_git_project_alias(self, monkeypatch):
        """gitProject falls back to the same project resolver (kept for codex-compat)."""
        monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["gitProject"])
        result = derive_bank_id(_hook(cwd="/work/sharedrepo"), cfg)
        assert "sharedrepo" in result
