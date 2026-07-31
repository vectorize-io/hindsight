"""Tests for lib/bank.py — bank ID derivation and mission management.

Ported from the Claude Code plugin's test_bank.py. `_hook()` still sets
`cwd` on the hook input dict (Devin CLI hooks don't send it, but bank.py's
`_resolve_cwd()` falls back to it when `DEVIN_PROJECT_DIR` is unset, which is
what these tests exercise — every test here explicitly clears the env var).
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from lib.bank import _resolve_project_name, derive_bank_id, ensure_bank_mission


@pytest.fixture(autouse=True)
def _no_devin_project_dir(monkeypatch):
    monkeypatch.delenv("DEVIN_PROJECT_DIR", raising=False)


def _cfg(**overrides):
    base = {
        "dynamicBankId": False,
        "bankId": "devin-cli",
        "bankIdPrefix": "",
        "agentName": "devin-cli",
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
        assert derive_bank_id(_hook(), _cfg()) == "devin-cli"

    def test_static_custom_bank_id(self):
        cfg = _cfg(bankId="my-agent")
        assert derive_bank_id(_hook(), cfg) == "my-agent"

    def test_static_with_prefix(self):
        cfg = _cfg(bankId="bot", bankIdPrefix="prod")
        assert derive_bank_id(_hook(), cfg) == "prod-bot"

    def test_static_prefix_without_bankid_uses_default(self):
        cfg = _cfg(bankId=None, bankIdPrefix="dev")
        assert derive_bank_id(_hook(), cfg) == "dev-devin-cli"


class TestDeriveBankIdDynamic:
    def test_dynamic_agent_project(self):
        cfg = _cfg(dynamicBankId=True, agentName="mybot", dynamicBankGranularity=["agent", "project"])
        result = derive_bank_id(_hook(cwd="/home/user/hindsight"), cfg)
        assert result == "mybot::hindsight"

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

    def test_devin_project_dir_env_takes_priority_over_hook_cwd(self, monkeypatch):
        # Devin CLI never sends `cwd` on hook stdin — DEVIN_PROJECT_DIR is the
        # real source of truth and must win when both are present.
        monkeypatch.setenv("DEVIN_PROJECT_DIR", "/home/user/real-project")
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"], resolveWorktrees=False)
        result = derive_bank_id(_hook(cwd="/home/user/stale-project"), cfg)
        assert result == "real-project"

    @patch("lib.bank.subprocess.run")
    def test_dynamic_worktree_resolves_to_main_repo(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "/home/user/myproject/.git\n"
        mock_run.return_value = mock_result

        cfg = _cfg(dynamicBankId=True, agentName="bot", dynamicBankGranularity=["agent", "project"])
        result = derive_bank_id(_hook(cwd="/home/user/myproject-wt1"), cfg)
        assert result == "bot::myproject"


class TestResolveProjectName:
    """Tests for git worktree resolution in project name derivation."""

    def _mock_git(self, stdout, returncode=0):
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
        mock_run.return_value = self._mock_git("/home/user/myproject/.git\n")
        assert _resolve_project_name("/home/user/myproject-wt1", _cfg()) == "myproject"

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
    def test_exact_match(self):
        cfg = _cfg(directoryBankMap={"/home/user/myproject": "custom-bank"})
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "custom-bank"

    def test_no_match_falls_through_to_static(self):
        cfg = _cfg(directoryBankMap={"/home/user/other": "other-bank"}, bankId="default-bank")
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "default-bank"

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
        ensure_bank_mission(client, "bank-a", cfg)
        assert client.set_bank_mission.call_count == 1

    def test_skips_if_mission_empty(self, state_dir):
        client = MagicMock()
        cfg = _cfg(bankMission="")
        ensure_bank_mission(client, "bank-b", cfg)
        client.set_bank_mission.assert_not_called()

    def test_graceful_on_api_error(self, state_dir):
        client = MagicMock()
        client.set_bank_mission.side_effect = RuntimeError("server down")
        cfg = _cfg(bankMission="mission")
        ensure_bank_mission(client, "bank-d", cfg)


class TestBankMissionStateFailureIsReportedHonestly:
    """A state-write failure must not be reported as an API failure.

    The mission call and the record of it used to share one `except
    Exception`, so an unwritable state directory surfaced as "Could not set
    bank mission" — when the mission had in fact been set and only the record
    of it was lost. The two are now separate.

    It is still not raised. This runs on the recall and retain hook paths, and
    a hook that has already done its work must not abort because a diagnostic
    write failed; that exact regression was fixed once already in recall.py.
    """

    def _unwritable_state(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        (tmp_path / "state").mkdir(parents=True, exist_ok=True)
        (tmp_path / "state").chmod(0o500)

    def test_an_unwritable_state_dir_does_not_abort_the_hook(self, tmp_path, monkeypatch):
        client = MagicMock()
        self._unwritable_state(tmp_path, monkeypatch)

        ensure_bank_mission(client, "bank-x", _cfg(bankMission="mission"))

        client.set_bank_mission.assert_called_once()

    def test_the_failure_is_not_blamed_on_the_api_call(self, tmp_path, monkeypatch):
        client = MagicMock()
        self._unwritable_state(tmp_path, monkeypatch)
        seen = []

        ensure_bank_mission(client, "bank-x", _cfg(bankMission="mission"), debug_fn=seen.append)

        assert seen, "expected a debug line explaining what failed"
        joined = " ".join(seen)
        assert "not recorded" in joined
        assert "Could not set bank mission" not in joined, (
            "the mission was set — reporting it as an API failure sends debugging the wrong way"
        )

    def test_an_api_failure_is_still_reported_as_one(self, tmp_path, monkeypatch):
        """Control: splitting the handlers must not lose the original message."""
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        client = MagicMock()
        client.set_bank_mission.side_effect = RuntimeError("server down")
        seen = []

        ensure_bank_mission(client, "bank-y", _cfg(bankMission="mission"), debug_fn=seen.append)

        assert any("Could not set bank mission" in line for line in seen)


class TestMalformedDirectoryBankMapDoesNotBreakBothHooks:
    """derive_bank_id runs in recall *and* retain, so a crash here blocks all work.

    `config.get("directoryBankMap") or {}` guards only falsy values, so a
    truthy non-mapping — a JSON list, a bare string — still reached .items().
    Guarded in the function as well as centrally at config load, because this
    takes a caller-supplied config rather than always one from load_config().
    """

    @pytest.mark.parametrize("bad", [["/src", "work"], "work", 7])
    def test_a_non_mapping_falls_back_to_normal_resolution(self, bad, monkeypatch):
        monkeypatch.setenv("DEVIN_PROJECT_DIR", "/home/user/myproject")

        result = derive_bank_id({}, _cfg(directoryBankMap=bad, bankId="fallback-bank"))

        assert result == "fallback-bank"

    def test_a_real_map_still_wins(self, monkeypatch, tmp_path):
        """Control: the guard must not disable the feature it protects."""
        monkeypatch.setenv("DEVIN_PROJECT_DIR", str(tmp_path))

        result = derive_bank_id({}, _cfg(directoryBankMap={str(tmp_path): "mapped-bank"}, bankId="fallback-bank"))

        assert result == "mapped-bank"


class TestMalformedGranularityFieldsDoNotBreakBothHooks:
    """`dynamicBankGranularity` is checked centrally as a list — not element by element.

    Both the validity check and the value lookup are hashed, so an unhashable
    element raises TypeError rather than being reported as an unknown field,
    and takes recall and retain down with it.
    """

    @pytest.mark.parametrize("bad_field", [[], {}, ["nested"], {"a": 1}])
    def test_an_unhashable_field_resolves_to_unknown_instead_of_raising(self, bad_field, monkeypatch):
        monkeypatch.setenv("DEVIN_PROJECT_DIR", "/home/user/myproject")

        result = derive_bank_id({}, _cfg(dynamicBankGranularity=["agent", bad_field], dynamicBankId=True))

        assert "unknown" in result

    @pytest.mark.parametrize("bad_field", [7, None, True])
    def test_a_non_string_scalar_field_also_resolves_to_unknown(self, bad_field, monkeypatch):
        monkeypatch.setenv("DEVIN_PROJECT_DIR", "/home/user/myproject")

        result = derive_bank_id({}, _cfg(dynamicBankGranularity=["agent", bad_field], dynamicBankId=True))

        assert "unknown" in result

    def test_valid_fields_still_resolve(self, monkeypatch):
        """Control: the guard must not disable the feature it protects."""
        monkeypatch.setenv("DEVIN_PROJECT_DIR", "/home/user/myproject")

        result = derive_bank_id({}, _cfg(dynamicBankGranularity=["agent"], agentName="devin-cli", dynamicBankId=True))

        assert "unknown" not in result


class TestMalformedDirectoryBankMapValuesDoNotMisrouteMemory:
    """The map's *values* are user-authored too, and they become the bank id.

    Only the map itself was type-checked, so a non-string value was returned
    verbatim as the bank id — or, with a bankIdPrefix set, f-string-formatted
    into one, making the literal bank `p-['wrong']`. Both hooks then read and
    wrote a bank the user never named, which is a silent data-routing error
    rather than a crash: nothing surfaces until memories go missing.
    """

    @pytest.mark.parametrize("bad", [["wrong"], {"a": 1}, 7, None, True, ""])
    def test_a_non_string_value_falls_through_to_the_next_branch(self, tmp_path, bad):
        result = derive_bank_id(
            {"cwd": str(tmp_path)},
            _cfg(directoryBankMap={str(tmp_path): bad}, bankId="fallback-bank"),
        )

        assert result == "fallback-bank"

    def test_a_non_string_value_is_not_formatted_into_a_prefixed_bank(self, tmp_path):
        result = derive_bank_id(
            {"cwd": str(tmp_path)},
            _cfg(directoryBankMap={str(tmp_path): ["wrong"]}, bankId="fallback-bank", bankIdPrefix="p"),
        )

        assert result == "p-fallback-bank"
        assert "wrong" not in result

    def test_a_valid_entry_still_wins_alongside_a_malformed_one(self, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        result = derive_bank_id(
            {"cwd": str(tmp_path)},
            _cfg(directoryBankMap={str(other): 7, str(tmp_path): "mapped-bank"}, bankId="fallback-bank"),
        )

        assert result == "mapped-bank"
