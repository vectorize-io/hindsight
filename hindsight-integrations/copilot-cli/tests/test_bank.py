"""Tests for lib/bank.py — bank ID derivation for the Copilot CLI integration."""

from lib.bank import derive_bank_id


def _cfg(**overrides):
    base = {
        "dynamicBankId": False,
        "bankId": "copilot-cli",
        "bankIdPrefix": "",
        "agentName": "copilot-cli",
        "dynamicBankGranularity": ["agent", "project"],
        "bankMission": "",
        "retainMission": None,
    }
    base.update(overrides)
    return base


def _hook(session_id="sess-1", cwd="/home/user/myproject"):
    return {
        "sessionId": session_id,
        "cwd": cwd,
    }


class TestDeriveBankIdStatic:
    def test_static_default_bank(self):
        assert derive_bank_id(_hook(), _cfg()) == "copilot-cli"

    def test_static_custom_bank_id(self):
        cfg = _cfg(bankId="my-agent")
        assert derive_bank_id(_hook(), cfg) == "my-agent"

    def test_static_with_prefix(self):
        cfg = _cfg(bankId="bot", bankIdPrefix="prod")
        assert derive_bank_id(_hook(), cfg) == "prod-bot"

    def test_static_prefix_without_bankid_uses_default(self):
        cfg = _cfg(bankId=None, bankIdPrefix="dev")
        assert derive_bank_id(_hook(), cfg) == "dev-copilot-cli"


class TestDeriveBankIdDynamic:
    def test_dynamic_agent_project(self):
        cfg = _cfg(dynamicBankId=True, agentName="mybot", dynamicBankGranularity=["agent", "project"])
        result = derive_bank_id(_hook(cwd="/home/user/hindsight"), cfg)
        assert result == "mybot::hindsight"

    def test_dynamic_uses_cwd_for_project(self):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        result = derive_bank_id(_hook(cwd="/work/myapp"), cfg)
        assert "myapp" in result

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

    def test_dynamic_session_id_snake_case_fallback(self):
        """Defensive: also accepts snake_case session_id (VS Code compatible payloads)."""
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["session"])
        hook = {"session_id": "sess-snake", "cwd": "/x"}
        result = derive_bank_id(hook, cfg)
        assert "sess-snake" in result

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
        result = derive_bank_id({"sessionId": "s", "cwd": ""}, cfg)
        assert "unknown" in result

    def test_dynamic_unknown_granularity_field_warns_but_continues(self, capsys):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["gitProject"])
        result = derive_bank_id(_hook(cwd="/work/sharedrepo"), cfg)
        # gitProject isn't a valid field for copilot-cli (no VALID_FIELDS entry) —
        # falls back to "unknown" for that segment, and prints a warning to stderr.
        assert "unknown" in result
        captured = capsys.readouterr()
        assert "Unknown dynamicBankGranularity field" in captured.err
