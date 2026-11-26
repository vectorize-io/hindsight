"""Reading one Hermes turn out of its stream-json output — the eval's only deterministic step."""

from __future__ import annotations

import json

from hindsight_system_evals.hermes import _parse_turn


def _stream(*events: dict) -> str:
    return "diagnostic line on stdout\n" + "".join(json.dumps(e) + "\n" for e in events)


def test_reply_tools_and_session_come_from_the_events() -> None:
    turn = _parse_turn(
        "hi",
        _stream(
            {"type": "system", "subtype": "init", "session_id": ""},
            {"type": "tool_use", "name": "hindsight_recall", "input": {"query": "food"}},
            {"type": "tool_result", "name": "hindsight_recall", "output": "..."},
            {"type": "text", "text": "partial"},
            {"type": "result", "session_id": "local-abc", "exit_code": 0, "text": "Welcome to Lisbon!"},
        ),
    )
    assert turn.reply == "Welcome to Lisbon!"
    assert turn.tools == ["hindsight_recall"]
    assert turn.calls[0].input == '{"query": "food"}'
    assert turn.session_id == "local-abc"
    assert turn.error == ""


def test_a_failed_or_missing_result_is_an_error() -> None:
    failed = _parse_turn("hi", _stream({"type": "result", "exit_code": 1, "text": "", "error": "no credentials"}))
    assert failed.error == "no credentials"
    assert _parse_turn("hi", "Traceback ...\n").error == "no result event in Hermes' output"
