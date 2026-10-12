"""Tests for Cursor plugin bank ID derivation."""

import os

import pytest

from lib.bank import DEFAULT_BANK_NAME, derive_bank_id, workspace_root


class TestDeriveBankId:
    def test_static_mode_default(self):
        config = {"dynamicBankId": False, "bankId": None, "bankIdPrefix": ""}
        result = derive_bank_id({}, config)
        assert result == DEFAULT_BANK_NAME
        assert result == "cursor"

    def test_static_mode_custom(self):
        config = {"dynamicBankId": False, "bankId": "my-project", "bankIdPrefix": ""}
        result = derive_bank_id({}, config)
        assert result == "my-project"

    def test_static_mode_with_prefix(self):
        config = {"dynamicBankId": False, "bankId": "my-project", "bankIdPrefix": "org"}
        result = derive_bank_id({}, config)
        assert result == "org-my-project"

    def test_dynamic_mode_project(self):
        config = {
            "dynamicBankId": True,
            "dynamicBankGranularity": ["project"],
            "bankIdPrefix": "",
            "agentName": "cursor",
        }
        hook_input = {"cwd": "/home/user/my-project"}
        result = derive_bank_id(hook_input, config)
        assert result == "my-project"

    def test_dynamic_mode_agent_project(self):
        config = {
            "dynamicBankId": True,
            "dynamicBankGranularity": ["agent", "project"],
            "bankIdPrefix": "",
            "agentName": "cursor",
        }
        hook_input = {"cwd": "/home/user/my-project"}
        result = derive_bank_id(hook_input, config)
        assert result == "cursor::my-project"

    def test_dynamic_mode_session(self):
        config = {
            "dynamicBankId": True,
            "dynamicBankGranularity": ["session"],
            "bankIdPrefix": "",
            "agentName": "cursor",
        }
        hook_input = {"conversation_id": "cursor-session-123"}
        result = derive_bank_id(hook_input, config)
        assert result == "cursor-session-123"

    def test_dynamic_mode_with_prefix(self):
        config = {
            "dynamicBankId": True,
            "dynamicBankGranularity": ["agent"],
            "bankIdPrefix": "company",
            "agentName": "cursor",
        }
        result = derive_bank_id({}, config)
        assert result == "company-cursor"

    def test_dynamic_mode_unknown_field_warns(self, capsys):
        config = {
            "dynamicBankId": True,
            "dynamicBankGranularity": ["agent", "invalid_field"],
            "bankIdPrefix": "",
            "agentName": "cursor",
        }
        derive_bank_id({}, config)
        captured = capsys.readouterr()
        assert "Unknown dynamicBankGranularity field" in captured.err

    def test_dynamic_mode_no_cwd(self):
        config = {
            "dynamicBankId": True,
            "dynamicBankGranularity": ["project"],
            "bankIdPrefix": "",
            "agentName": "cursor",
        }
        result = derive_bank_id({}, config)
        assert result == "unknown"

    def test_dynamic_mode_project_from_workspace_roots(self):
        """sessionStart sends workspace_roots and no cwd; Windows roots look like /C:/..."""
        config = {"dynamicBankId": True, "dynamicBankGranularity": ["project"], "agentName": "cursor"}
        assert derive_bank_id({"workspace_roots": ["/home/u/research"]}, config) == "research"
        assert derive_bank_id({"workspace_roots": ["/C:/Users/u/research"]}, config) == "research"
        # cwd still wins when present
        assert derive_bank_id({"cwd": "/a/other", "workspace_roots": ["/b/research"]}, config) == "other"

    def test_workspace_root_strips_windows_leading_slash(self):
        assert workspace_root({"workspace_roots": ["/C:/Users/u/research"]}) == "C:/Users/u/research"
        assert workspace_root({"workspace_roots": ["/home/u/research"]}) == "/home/u/research"
        assert workspace_root({}) == ""

    def test_dynamic_mode_channel_user(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_CHANNEL_ID", "slack-general")
        monkeypatch.setenv("HINDSIGHT_USER_ID", "user123")
        config = {
            "dynamicBankId": True,
            "dynamicBankGranularity": ["channel", "user"],
            "bankIdPrefix": "",
            "agentName": "cursor",
        }
        result = derive_bank_id({}, config)
        assert result == "slack-general::user123"
