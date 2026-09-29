"""Tests for Cursor plugin bank ID derivation."""

import os

import pytest

from lib.bank import derive_bank_id, DEFAULT_BANK_NAME


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
