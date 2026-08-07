"""Tests for lib/bank.py — bank ID derivation."""

import os
import subprocess

import pytest

from lib.bank import derive_bank_id


def _cfg(**overrides):
    base = {
        "dynamicBankId": False,
        "bankId": "codex",
        "bankIdPrefix": "",
        "agentName": "codex",
        "dynamicBankGranularity": ["agent", "project"],
        "bankMission": "",
        "retainMission": None,
    }
    base.update(overrides)
    return base


def _hook(session_id="sess-1", cwd="/home/user/myproject"):
    return {"session_id": session_id, "cwd": cwd}


class TestDeriveBankIdStatic:
    def test_static_default_bank(self):
        assert derive_bank_id(_hook(), _cfg()) == "codex"

    def test_static_custom_bank_id(self):
        cfg = _cfg(bankId="my-agent")
        assert derive_bank_id(_hook(), cfg) == "my-agent"

    def test_static_with_prefix(self):
        cfg = _cfg(bankId="bot", bankIdPrefix="prod")
        assert derive_bank_id(_hook(), cfg) == "prod-bot"

    def test_static_prefix_without_bankid_uses_default(self):
        cfg = _cfg(bankId=None, bankIdPrefix="dev")
        assert derive_bank_id(_hook(), cfg) == "dev-codex"


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

    def test_dynamic_user_from_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_USER_ID", "user-456")
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["user"])
        result = derive_bank_id(_hook(), cfg)
        assert "user-456" in result

    def test_dynamic_missing_env_uses_default(self, monkeypatch):
        monkeypatch.delenv("HINDSIGHT_USER_ID", raising=False)
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["user"])
        result = derive_bank_id(_hook(), cfg)
        assert "anonymous" in result

    def test_dynamic_empty_cwd_uses_unknown(self):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        result = derive_bank_id({"session_id": "s", "cwd": ""}, cfg)
        assert "unknown" in result


class TestWorktreeResolution:
    @pytest.fixture
    def repo_with_worktree(self, tmp_path):
        """A real git repo at mainrepo/ with a linked worktree at wt/."""
        repo = tmp_path / "mainrepo"
        repo.mkdir()
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
        def run(*args, cwd=repo):
            subprocess.run(args, cwd=cwd, env=env, check=True, capture_output=True)
        run("git", "init", "-q")
        run("git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init")
        wt = tmp_path / "wt"
        run("git", "worktree", "add", "-q", str(wt))
        return repo, wt

    def test_worktree_resolves_to_main_repo_basename(self, repo_with_worktree):
        _, wt = repo_with_worktree
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        assert derive_bank_id(_hook(cwd=str(wt)), cfg) == "mainrepo"

    def test_main_checkout_uses_own_basename(self, repo_with_worktree):
        repo, _ = repo_with_worktree
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        assert derive_bank_id(_hook(cwd=str(repo)), cfg) == "mainrepo"

    def test_resolve_worktrees_false_uses_literal_basename(self, repo_with_worktree):
        _, wt = repo_with_worktree
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"], resolveWorktrees=False)
        assert derive_bank_id(_hook(cwd=str(wt)), cfg) == "wt"

    def test_non_git_directory_uses_basename(self, tmp_path):
        d = tmp_path / "plaindir"
        d.mkdir()
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        assert derive_bank_id(_hook(cwd=str(d)), cfg) == "plaindir"

    def test_nonexistent_cwd_falls_back_to_basename(self):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        assert derive_bank_id(_hook(cwd="/nonexistent/somewhere/proj"), cfg) == "proj"
