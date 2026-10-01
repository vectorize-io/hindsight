"""Codex retain request regressions for explicit strategy scope and source roles."""

import io
import json

import pytest
from conftest import FakeHTTPResponse, make_hook_input, make_transcript_file
from test_hooks import _run_hook


class TestRetainStrategyHook:
    @pytest.mark.parametrize("hook", ["retain", "recall", "session_start"])
    @pytest.mark.parametrize("strategy", ["agent-session", "custom-session"])
    def test_all_hooks_preserve_preprovisioned_bank_config(self, monkeypatch, tmp_path, hook, strategy):
        transcript_path = make_transcript_file(tmp_path, [{"role": "user", "content": "Use a short response."}])
        requests = []

        def fake_open(req, timeout=None):
            requests.append(req)
            return FakeHTTPResponse({"results": []})

        _run_hook(
            hook,
            make_hook_input(transcript_path=transcript_path),
            monkeypatch,
            tmp_path,
            urlopen_side_effect=fake_open,
            user_config={
                "retainStrategy": strategy,
                "bankMission": "Legacy reflect mission",
                "retainMission": "Legacy bank-wide retain mission",
                "upgradeNotice": False,
            },
        )
        assert [req.method for req in requests] == ([] if hook == "session_start" else ["POST"])
        assert not (tmp_path / ".hindsight/codex/state/bank_missions.json").exists()

    @pytest.mark.parametrize("include_tool_calls", [True, False])
    def test_explicit_agent_session_preserves_roles_and_qualifications_without_bank_patch(
        self,
        monkeypatch,
        tmp_path,
        include_tool_calls,
    ):
        messages = [
            {"role": "user", "content": 'Quoted video speaker: "I work in academia." Identity unknown.'},
            {"role": "assistant", "content": "I recommend the X CLI. PR #134 is open, not merged or deployed."},
            {"role": "user", "content": "That is your recommendation. I have not adopted it."},
            {"role": "assistant", "content": "I reported tests passed, but no independent result is attached."},
        ]
        transcript_path = make_transcript_file(tmp_path, messages, codex_format=True)
        requests = []

        def fake_open(req, timeout=None):
            requests.append(req)
            return FakeHTTPResponse({"operation_id": "op-session"})

        _run_hook(
            "retain",
            make_hook_input(transcript_path=transcript_path),
            monkeypatch,
            tmp_path,
            urlopen_side_effect=fake_open,
            user_config={
                "bankId": "shared-bank",
                "retainStrategy": "agent-session",
                "retainContext": "A Codex source transcript",
                "retainToolCalls": include_tool_calls,
                # Legacy bankMission must not mutate an already provisioned shared bank.
                "bankMission": "Legacy reflect mission",
                "retainMission": "Legacy bank-wide retain mission",
            },
        )

        assert len(requests) == 1
        req = requests[0]
        assert req.method == "POST"
        assert req.full_url == "http://fake:9077/v1/default/banks/shared-bank/memories"
        item = json.loads(req.data)["items"][0]
        assert item["strategy"] == "agent-session"
        assert item["context"].startswith("A Codex source transcript\n\n")
        for guard in [
            "Outer message roles",
            "unknown or ambiguous speakers",
            "not the user's preference",
            "proposed, requested, open, merged, installed and deployed",
            "agent-reported",
            "tool-call envelope",
            "do not prove success",
            "event time",
            "lead-agent and subagent",
        ]:
            assert guard in item["context"]
        if include_tool_calls:
            assert json.loads(item["content"]) == [
                {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}
                for message in messages
            ]
        else:
            for message in messages:
                assert f"[role: {message['role']}]\n{message['content']}" in item["content"]
        assert not (tmp_path / ".hindsight/codex/state/bank_missions.json").exists()

    @pytest.mark.parametrize("strategy", [None, "custom-session"])
    def test_session_guard_is_only_added_for_explicit_agent_session(self, monkeypatch, tmp_path, strategy):
        transcript_path = make_transcript_file(
            tmp_path,
            [
                {"role": "user", "content": "Use a short response."},
                {"role": "assistant", "content": "I will use a short response."},
            ],
        )
        requests = []

        def fake_open(req, timeout=None):
            requests.append(req)
            return FakeHTTPResponse({"operation_id": "op-session"})

        _run_hook(
            "retain",
            make_hook_input(transcript_path=transcript_path),
            monkeypatch,
            tmp_path,
            urlopen_side_effect=fake_open,
            user_config={"retainStrategy": strategy, "retainContext": "custom provenance", "bankMission": ""},
        )
        item = json.loads(requests[-1].data)["items"][0]
        assert item["context"] == "custom provenance"
        assert item.get("strategy") == strategy
        if strategy is None:
            assert "strategy" not in item

    @pytest.mark.parametrize("strategy", [42, True, [], {}])
    def test_invalid_strategy_skips_before_any_api_call(self, monkeypatch, tmp_path, strategy):
        def unexpected_request(*a, **kw):
            pytest.fail("An invalid strategy must not retain generically or patch a bank")

        transcript_path = make_transcript_file(
            tmp_path,
            [
                {"role": "user", "content": "Use a short response."},
            ],
        )
        _run_hook(
            "retain",
            make_hook_input(transcript_path=transcript_path),
            monkeypatch,
            tmp_path,
            urlopen_side_effect=unexpected_request,
            user_config={"retainStrategy": strategy},
        )

    @pytest.mark.parametrize("strategy", [None, "", "  "])
    def test_legacy_default_still_sets_configured_bank_mission(self, monkeypatch, tmp_path, strategy):
        transcript_path = make_transcript_file(
            tmp_path,
            [
                {"role": "user", "content": "Use a short response."},
            ],
        )
        requests = []

        def fake_open(req, timeout=None):
            requests.append(req)
            return FakeHTTPResponse({"operation_id": "op-session"})

        _run_hook(
            "retain",
            make_hook_input(transcript_path=transcript_path),
            monkeypatch,
            tmp_path,
            urlopen_side_effect=fake_open,
            user_config={
                "bankMission": "Explicit legacy mission",
                "retainMission": "Legacy extraction",
                "retainStrategy": strategy,
            },
        )
        assert [req.method for req in requests] == ["PATCH", "POST"]
        assert json.loads(requests[0].data) == {
            "updates": {"reflect_mission": "Explicit legacy mission", "retain_mission": "Legacy extraction"},
        }
        assert "strategy" not in json.loads(requests[1].data)["items"][0]

    def test_strategy_rejection_does_not_fall_back_to_generic_retain(self, monkeypatch, tmp_path):
        import urllib.error

        transcript_path = make_transcript_file(
            tmp_path,
            [
                {"role": "user", "content": "Use a short response."},
            ],
        )
        requests = []

        def reject_strategy(req, timeout=None):
            requests.append(req)
            raise urllib.error.HTTPError(req.full_url, 400, "Unknown strategy", {}, io.BytesIO(b"Unknown strategy"))

        output = _run_hook(
            "retain",
            make_hook_input(transcript_path=transcript_path),
            monkeypatch,
            tmp_path,
            urlopen_side_effect=reject_strategy,
            user_config={"retainStrategy": "agent-session"},
        )
        assert output == ""
        assert len(requests) == 1
        assert json.loads(requests[0].data)["items"][0]["strategy"] == "agent-session"
