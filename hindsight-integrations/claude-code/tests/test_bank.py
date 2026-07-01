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

    def test_windows_drive_letter_case_insensitive_match(self):
        # On Windows, the cwd's drive-letter (and path) case depends on the
        # launcher: PowerShell and git-bash hand children an UPPERCASE drive
        # while the VS Code extension spawn reports lowercase. The map match
        # must not depend on which launcher started the session. normcase is
        # a no-op on POSIX, so this test only exercises the Windows behavior
        # when run there; the POSIX-case-sensitivity test below pins the
        # complementary guarantee.
        import os as _os

        if _os.path.normcase("A") == _os.path.normcase("a"):  # case-insensitive FS semantics
            cfg = _cfg(directoryBankMap={r"c:\Users\dev\proj": "custom-bank"})
            result = derive_bank_id(_hook(cwd=r"C:\Users\dev\proj"), cfg)
            assert result == "custom-bank"

    def test_posix_paths_stay_case_sensitive(self):
        import os as _os

        if _os.path.normcase("A") != _os.path.normcase("a"):  # POSIX semantics
            cfg = _cfg(directoryBankMap={"/home/User/myproject": "custom-bank"}, bankId="default-bank")
            result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
            assert result == "default-bank"

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
    def _client(self, overrides=None):
        """MagicMock client whose bank has the given config overrides.

        `overrides=None` simulates a bank that doesn't exist yet (GET 404 →
        empty dict from get_bank_config).
        """
        client = MagicMock()
        client.get_bank_config.return_value = {"overrides": overrides} if overrides is not None else {}
        return client

    def test_seeds_reflect_mission_on_new_bank(self, state_dir):
        client = self._client()
        cfg = _cfg(bankMission="You are a helpful assistant.", bankId="test-bank")
        ensure_bank_mission(client, "test-bank", cfg)
        client.get_bank_config.assert_called_once_with("test-bank", timeout=10)
        client.set_bank_mission.assert_called_once_with(
            "test-bank", reflect_mission="You are a helpful assistant.", retain_mission=None, timeout=10
        )

    def test_seeds_both_missions_on_new_bank(self, state_dir):
        client = self._client()
        cfg = _cfg(bankMission="reflect mission", retainMission="retain mission")
        ensure_bank_mission(client, "bank-c", cfg)
        client.set_bank_mission.assert_called_once_with(
            "bank-c", reflect_mission="reflect mission", retain_mission="retain mission", timeout=10
        )

    def test_does_not_overwrite_existing_server_missions(self, state_dir):
        # Bank already has both missions authored out-of-band (control plane).
        client = self._client(
            overrides={"reflect_mission": "You are an artist", "retain_mission": "Extract paintings"}
        )
        cfg = _cfg(bankMission="Claude Code default", retainMission="tech default")
        ensure_bank_mission(client, "bank-existing", cfg)
        client.set_bank_mission.assert_not_called()

    def test_fills_only_the_unset_field(self, state_dir):
        # Bank has a reflect mission but no retain mission; plugin fills the gap.
        client = self._client(overrides={"reflect_mission": "You are an artist"})
        cfg = _cfg(bankMission="Claude Code default", retainMission="tech default")
        ensure_bank_mission(client, "bank-partial", cfg)
        client.set_bank_mission.assert_called_once_with(
            "bank-partial", reflect_mission=None, retain_mission="tech default", timeout=10
        )

    def test_retain_only_config_seeds_retain(self, state_dir):
        # bankMission empty but retainMission set — must still seed retain.
        client = self._client()
        cfg = _cfg(bankMission="", retainMission="retain only")
        ensure_bank_mission(client, "bank-retain", cfg)
        client.set_bank_mission.assert_called_once_with(
            "bank-retain", reflect_mission=None, retain_mission="retain only", timeout=10
        )

    def test_skips_if_already_reconciled_locally(self, state_dir):
        client = self._client()
        cfg = _cfg(bankMission="mission text")
        ensure_bank_mission(client, "bank-a", cfg)
        ensure_bank_mission(client, "bank-a", cfg)  # second call → fast path
        assert client.get_bank_config.call_count == 1
        assert client.set_bank_mission.call_count == 1

    def test_skips_if_no_missions_configured(self, state_dir):
        client = self._client()
        cfg = _cfg(bankMission="", retainMission=None)
        ensure_bank_mission(client, "bank-b", cfg)
        client.get_bank_config.assert_not_called()
        client.set_bank_mission.assert_not_called()

    def test_graceful_on_api_error_and_retries(self, state_dir):
        client = MagicMock()
        client.get_bank_config.side_effect = RuntimeError("server down")
        cfg = _cfg(bankMission="mission")
        # Should not raise, and must not mark the bank reconciled (so it retries).
        ensure_bank_mission(client, "bank-d", cfg)
        client.get_bank_config.side_effect = None
        client.get_bank_config.return_value = {}
        ensure_bank_mission(client, "bank-d", cfg)
        client.set_bank_mission.assert_called_once()

    def test_different_banks_each_seeded_once(self, state_dir):
        client = self._client()
        cfg = _cfg(bankMission="mission")
        ensure_bank_mission(client, "bank-x", cfg)
        ensure_bank_mission(client, "bank-y", cfg)
        assert client.set_bank_mission.call_count == 2

    @pytest.mark.skipif(not hasattr(__import__("os"), "symlink"), reason="symlinks not supported")
    def test_directorybankmap_matches_symlinked_cwd(self, tmp_path):
        import os
        real = os.path.realpath(tmp_path / "proj")
        os.makedirs(real)
        link = str(tmp_path / "proj-link")
        os.symlink(real, link)
        cfg = _cfg(directoryBankMap={real: "myproj"}, bankId="fallback")
        assert derive_bank_id({"cwd": real, "session_id": "s"}, cfg) == "myproj"
        assert derive_bank_id({"cwd": link, "session_id": "s"}, cfg) == "myproj"
