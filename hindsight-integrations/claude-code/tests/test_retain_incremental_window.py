"""Full-session retention must send the whole unprocessed suffix, not just its last turn.

`plan_retention()` selects `[start_index, len(all_messages))` as the unprocessed
range. That range was then narrowed a second time by last-turn slicing on every
retain after the first, while `commit_retention()` still advanced the cursor past
the entire range — so whatever the second window dropped was skipped permanently.

Claude Code makes this far worse than it looks: tool results are stored as
`role="user"` messages, so the "last turn" scan usually lands on a tool result
rather than the human prompt. With the resolved default `retainToolCalls: false`
that message extracts to empty and the whole retain is skipped.

Measured on a real 1,500-message transcript, formatting the same 30-message
unprocessed suffix: last-turn 0 chars (retain skipped) vs full-window 3,697 chars
across 6 messages.

Every test here MUST drive a second retain, because the defect only appears once a
retention cursor exists — on a fresh session `start_index == 0` and the buggy
expression evaluates to True by accident.
"""

import json

from conftest import FakeHTTPResponse, make_hook_input, make_transcript_file
from test_hooks import _run_hook

TOOL_RESULT = {"role": "user", "content": [{"type": "tool_result", "content": "exit 0"}]}

SEED = [
    {"role": "user", "content": "seed question establishing the retention cursor"},
    {"role": "assistant", "content": "seed answer establishing the retention cursor"},
]


def _retain(monkeypatch, tmp_path, messages):
    """Run the retain hook over `messages`, returning the posted body (or None)."""
    transcript = make_transcript_file(tmp_path, messages)
    captured = {}

    def capture(req, timeout=None):
        if "/memories" in req.full_url and "/recall" not in req.full_url:
            captured["body"] = json.loads(req.data.decode())
        return FakeHTTPResponse({"status": "accepted"})

    _run_hook(
        "retain",
        make_hook_input(transcript_path=transcript),
        monkeypatch,
        tmp_path,
        urlopen_side_effect=capture,
    )
    return captured.get("body")


def _second_retain(monkeypatch, tmp_path, new_messages):
    """Establish a cursor with SEED, then retain SEED + new_messages.

    Returns the body posted by the SECOND retain — the incremental path.
    """
    first = _retain(monkeypatch, tmp_path, SEED)
    assert first is not None, "seed retain did not fire; cursor was never established"
    return _retain(monkeypatch, tmp_path, SEED + new_messages)


class TestIncrementalRetainWindow:
    def test_incremental_retain_sends_all_new_turns(self, monkeypatch, tmp_path):
        # Intention: with several turns accumulated since the cursor, last-turn
        # slicing keeps only the newest while the cursor advances past the rest.
        # Expected: every new turn appears in the second payload.
        new = []
        for n in ("alpha", "bravo", "charlie"):
            new.append({"role": "user", "content": f"question {n}"})
            new.append({"role": "assistant", "content": f"answer {n}"})
        body = _second_retain(monkeypatch, tmp_path, new)
        assert body is not None, "incremental retain did not fire"
        content = body["items"][0]["content"]
        for n in ("alpha", "bravo", "charlie"):
            assert f"question {n}" in content, f"turn {n} was dropped by re-windowing"
            assert f"answer {n}" in content

    def test_tool_result_tail_does_not_swallow_the_window(self, monkeypatch, tmp_path):
        # Intention: THE BUG. A tool result is a role="user" message, so the last-turn
        # scan anchors on it; with tool calls excluded it strips to empty and the whole
        # incremental retain is skipped while the cursor still advances.
        # Expected: the human prompt and assistant reply before it are still sent.
        new = [
            {"role": "user", "content": "why did the flame sensor trip"},
            {"role": "assistant", "content": "the 74HC14 ground went marginal"},
            TOOL_RESULT,
        ]
        body = _second_retain(monkeypatch, tmp_path, new)
        assert body is not None, "retain skipped — the trailing tool result swallowed the window"
        content = body["items"][0]["content"]
        assert "why did the flame sensor trip" in content
        assert "74HC14 ground went marginal" in content

    def test_interleaved_tool_results_keep_all_assistant_text(self, monkeypatch, tmp_path):
        # Intention: a realistic agent turn interleaves assistant reasoning with many
        # tool results. Anchoring on the last tool result keeps only the tail.
        # Expected: all assistant reasoning in the window survives.
        new = [
            {"role": "user", "content": "run the OTA gate"},
            {"role": "assistant", "content": "checking the manifest signature"},
            TOOL_RESULT,
            {"role": "assistant", "content": "signature verified against the pinned key"},
            TOOL_RESULT,
            {"role": "assistant", "content": "gate passed, firmware is publishable"},
            TOOL_RESULT,
        ]
        body = _second_retain(monkeypatch, tmp_path, new)
        assert body is not None
        content = body["items"][0]["content"]
        assert "checking the manifest signature" in content
        assert "signature verified against the pinned key" in content
        assert "gate passed, firmware is publishable" in content

    def test_incremental_retain_excludes_already_retained_messages(self, monkeypatch, tmp_path):
        # Intention: don't over-correct into re-sending. The cursor, not the window,
        # is what prevents duplicates.
        # Expected: the second payload carries only new content, not the seed.
        new = [
            {"role": "user", "content": "brand new question"},
            {"role": "assistant", "content": "brand new answer"},
        ]
        body = _second_retain(monkeypatch, tmp_path, new)
        assert body is not None
        content = body["items"][0]["content"]
        assert "brand new question" in content
        assert "seed question establishing the retention cursor" not in content

    def test_tool_only_window_sends_nothing(self, monkeypatch, tmp_path):
        # Intention: when the entire new window is tool results and tool calls are
        # excluded, there is genuinely nothing retainable — don't invent a payload.
        # Expected: no API call for the second retain.
        body = _second_retain(monkeypatch, tmp_path, [TOOL_RESULT, TOOL_RESULT])
        assert body is None, "expected no retain for a window with no retainable content"
