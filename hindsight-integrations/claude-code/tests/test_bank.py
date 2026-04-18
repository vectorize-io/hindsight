"""Tests for lib/bank.py — bank ID derivation and mission management."""

import json
from unittest.mock import MagicMock

import pytest

from lib.bank import derive_bank_id, ensure_bank_mission


def _cfg(**overrides):
    base = {
        "dynamicBankId": False,
        "bankId": "claude-code",
        "bankIdPrefix": "",
        "agentName": "claude-code",
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
