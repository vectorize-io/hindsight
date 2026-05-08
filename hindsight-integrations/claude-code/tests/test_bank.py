"""Tests for lib/bank.py — bank ID derivation and mission management."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from lib.bank import _resolve_project_name, derive_bank_id, ensure_bank_mission


def _cfg(**overrides):
    base = {
        "dynamicBankId": False,
        "bankId": "claude-code",
        "bankIdPrefix": "",
        "agentName": "claude-code",
        "dynamicBankGranularity": ["agent", "project"],
        "bankMission": "",
        "retainMission": None,
        "resolveWorktrees": True,
        "directoryBankMap": {},
    }
    base.update(overrides)
    return base


def _hook(session_id="sess-1", cwd="/home/user/myproject"):
    return {"session_id": session_id, "cwd": cwd}


class TestDeriveBankIdStatic:
    def test_static_default_bank(self):
        assert derive_bank_id(_hook(), _cfg()) == "claude-code"

    def test_static_custom_bank_id(self):
        cfg = _cfg(bankId="my-agent")
        assert derive_bank_id(_hook(), cfg) == "my-agent"

    def test_static_with_prefix(self):
        cfg = _cfg(bankId="bot", bankIdPrefix="prod")
        assert derive_bank_id(_hook(), cfg) == "prod-bot"

    def test_static_prefix_without_bankid_uses_default(self):
        cfg = _cfg(bankId=None, bankIdPrefix="dev")
        assert derive_bank_id(_hook(), cfg) == "dev-claude-code"


class TestDeriveBankIdDynamic:
    def test_dynamic_agent_project(self):
        cfg = _cfg(dynamicBankId=True, agentName="mybot", dynamicBankGranularity=["agent", "project"])
        result = derive_bank_id(_hook(cwd="/home/user/hindsight"), cfg)
        assert result == "mybot::hindsight"

    def test_dynamic_preserves_raw_special_chars(self):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        result = derive_bank_id(_hook(cwd="/home/user/my project"), cfg)
        assert "my project" in result
        assert "%" not in result

    def test_dynamic_preserves_raw_utf8(self):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        result = derive_bank_id(_hook(cwd="/home/user/мой проект"), cfg)
        assert "мой проект" in result
        assert "%" not in result

    def test_dynamic_session_field(self):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["session"])
        result = derive_bank_id(_hook(session_id="abc-123"), cfg)
        assert "abc-123" in result

    def test_dynamic_with_prefix(self):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["agent"], bankIdPrefix="v2")
        result = derive_bank_id(_hook(), cfg)
        assert result.startswith("v2-")

    def test_dynamic_channel_from_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_CHANNEL_ID", "telegram-123")
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["channel"])
        result = derive_bank_id(_hook(), cfg)
        assert "telegram-123" in result

    def test_dynamic_user_from_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_USER_ID", "user-456")
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["user"])
        result = derive_bank_id(_hook(), cfg)
        assert "user-456" in result

    def test_dynamic_missing_env_uses_defaults(self, monkeypatch):
        monkeypatch.delenv("HINDSIGHT_CHANNEL_ID", raising=False)
        monkeypatch.delenv("HINDSIGHT_USER_ID", raising=False)
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["channel", "user"])
        result = derive_bank_id(_hook(), cfg)
        assert "default" in result
        assert "anonymous" in result

    def test_dynamic_empty_cwd_uses_unknown(self):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        result = derive_bank_id({"session_id": "s", "cwd": ""}, cfg)
        assert "unknown" in result

    @patch("lib.bank.subprocess.run")
    def test_dynamic_worktree_resolves_to_main_repo(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "/home/user/myproject/.git\n"
        mock_run.return_value = mock_result

        cfg = _cfg(dynamicBankId=True, agentName="bot", dynamicBankGranularity=["agent", "project"])
        # Working in a worktree, but should resolve to the main repo name
        result = derive_bank_id(_hook(cwd="/home/user/myproject-wt1"), cfg)
        assert result == "bot::myproject"


class TestResolveProjectName:
    """Tests for git worktree resolution in project name derivation."""

    def _mock_git(self, stdout, returncode=0):
        """Create a mock for subprocess.run that simulates git output."""
        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout
        return result

    @patch("lib.bank.subprocess.run")
    def test_regular_repo(self, mock_run):
        mock_run.return_value = self._mock_git("/home/user/myproject/.git\n")
        assert _resolve_project_name("/home/user/myproject", _cfg()) == "myproject"

    @patch("lib.bank.subprocess.run")
    def test_worktree_resolves_to_main_repo(self, mock_run):
        # Worktree at /home/user/myproject-wt1, main repo at /home/user/myproject
        mock_run.return_value = self._mock_git("/home/user/myproject/.git\n")
        assert _resolve_project_name("/home/user/myproject-wt1", _cfg()) == "myproject"

    @patch("lib.bank.subprocess.run")
    def test_worktree_different_location(self, mock_run):
        # Worktree at /tmp/worktrees/feature-x, main repo at /home/user/hindsight
        mock_run.return_value = self._mock_git("/home/user/hindsight/.git\n")
        assert _resolve_project_name("/tmp/worktrees/feature-x", _cfg()) == "hindsight"

    @patch("lib.bank.subprocess.run")
    def test_disabled_falls_back_to_basename(self, mock_run):
        cfg = _cfg(resolveWorktrees=False)
        assert _resolve_project_name("/home/user/myproject-wt1", cfg) == "myproject-wt1"
        mock_run.assert_not_called()

    @patch("lib.bank.subprocess.run")
    def test_git_not_available(self, mock_run):
        mock_run.side_effect = OSError("git not found")
        assert _resolve_project_name("/home/user/myproject", _cfg()) == "myproject"

    @patch("lib.bank.subprocess.run")
    def test_not_a_git_repo(self, mock_run):
        mock_run.return_value = self._mock_git("", returncode=128)
        assert _resolve_project_name("/home/user/plaindir", _cfg()) == "plaindir"

    @patch("lib.bank.subprocess.run")
    def test_git_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=5)
        assert _resolve_project_name("/home/user/myproject", _cfg()) == "myproject"

    def test_empty_cwd(self):
        assert _resolve_project_name("", _cfg()) == "unknown"


class TestDirectoryBankMap:
    """Tests for explicit directory-to-bank mapping."""

    def test_exact_match(self):
        cfg = _cfg(directoryBankMap={"/home/user/myproject": "custom-bank"})
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "custom-bank"

    def test_match_with_trailing_slash(self):
        cfg = _cfg(directoryBankMap={"/home/user/myproject/": "custom-bank"})
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "custom-bank"

    def test_no_match_falls_through_to_static(self):
        cfg = _cfg(directoryBankMap={"/home/user/other": "other-bank"}, bankId="default-bank")
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "default-bank"

    def test_no_match_falls_through_to_dynamic(self):
        cfg = _cfg(
            directoryBankMap={"/home/user/other": "other-bank"},
            dynamicBankId=True,
            agentName="bot",
            dynamicBankGranularity=["agent"],
            resolveWorktrees=False,
        )
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "bot"

    def test_with_prefix(self):
        cfg = _cfg(
            directoryBankMap={"/home/user/myproject": "custom-bank"},
            bankIdPrefix="prod",
        )
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "prod-custom-bank"

    def test_overrides_dynamic_mode(self):
        cfg = _cfg(
            directoryBankMap={"/home/user/myproject": "explicit-bank"},
            dynamicBankId=True,
            agentName="bot",
            dynamicBankGranularity=["agent", "project"],
        )
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "explicit-bank"

    def test_empty_map_ignored(self):
        cfg = _cfg(directoryBankMap={}, bankId="default-bank")
        result = derive_bank_id(_hook(), cfg)
        assert result == "default-bank"

    def test_empty_cwd_skips_map(self):
        cfg = _cfg(directoryBankMap={"/some/path": "mapped-bank"}, bankId="fallback")
        result = derive_bank_id({"session_id": "s", "cwd": ""}, cfg)
        assert result == "fallback"

    def test_multiple_entries(self):
        cfg = _cfg(directoryBankMap={
            "/home/user/project-a": "bank-a",
            "/home/user/project-b": "bank-b",
        })
        assert derive_bank_id(_hook(cwd="/home/user/project-a"), cfg) == "bank-a"
        assert derive_bank_id(_hook(cwd="/home/user/project-b"), cfg) == "bank-b"


class TestEnsureBankMission:
    def test_sets_mission_on_first_call(self, state_dir):
        client = MagicMock()
        cfg = _cfg(bankMission="You are a helpful assistant.", bankId="test-bank")
        ensure_bank_mission(client, "test-bank", cfg)
        client.set_bank_mission.assert_called_once_with(
            "test-bank", "You are a helpful assistant.", retain_mission=None, timeout=10
        )

    def test_skips_if_already_set(self, state_dir):
        client = MagicMock()
        cfg = _cfg(bankMission="mission text")
        ensure_bank_mission(client, "bank-a", cfg)
        ensure_bank_mission(client, "bank-a", cfg)  # second call
        assert client.set_bank_mission.call_count == 1

    def test_skips_if_mission_empty(self, state_dir):
        client = MagicMock()
        cfg = _cfg(bankMission="")
        ensure_bank_mission(client, "bank-b", cfg)
        client.set_bank_mission.assert_not_called()

    def test_includes_retain_mission_if_set(self, state_dir):
        client = MagicMock()
        cfg = _cfg(bankMission="reflect mission", retainMission="retain mission")
        ensure_bank_mission(client, "bank-c", cfg)
        client.set_bank_mission.assert_called_once_with(
            "bank-c", "reflect mission", retain_mission="retain mission", timeout=10
        )

    def test_graceful_on_api_error(self, state_dir):
        client = MagicMock()
        client.set_bank_mission.side_effect = RuntimeError("server down")
        cfg = _cfg(bankMission="mission")
        # Should not raise
        ensure_bank_mission(client, "bank-d", cfg)

    def test_different_banks_each_set_once(self, state_dir):
        client = MagicMock()
        cfg = _cfg(bankMission="mission")
        ensure_bank_mission(client, "bank-x", cfg)
        ensure_bank_mission(client, "bank-y", cfg)
        assert client.set_bank_mission.call_count == 2
