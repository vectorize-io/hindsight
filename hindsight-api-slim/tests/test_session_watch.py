"""The memory watcher's rules on session events: which steps make a moment, and when a note may go out.

Pure functions, no database: see test_sessions.py for the session store and the worker steps.
"""

import pytest

from hindsight_api.engine.sessions import watch as W


def _feed(state: W.WatchState, *events: tuple[str, str, dict]) -> None:
    for i, (kind, role, payload) in enumerate(events, start=1):
        step = W.step_of(kind, role, payload)
        if step is not None:
            W.read_step(state, step, i)


@pytest.fixture(autouse=True)
def _host_never_busy(monkeypatch):
    """Checks skip on a loaded host (watch.host_busy); the tests must not depend on this machine's load."""
    monkeypatch.setattr(W, "BUSY_LOAD_PER_CPU", float("inf"))


def test_a_request_is_a_task_moment_and_becomes_the_goal():
    s = W.WatchState()
    _feed(s, ("message", "user", {"text": "Fix the rounding in pay/rounding.py"}))
    assert s.pending is not None and s.pending.trigger == "task" and s.pending.seq == 1
    assert s.goal == "Fix the rounding in pay/rounding.py"


def test_a_system_reminder_or_own_note_in_the_user_turn_is_not_a_request():
    s = W.WatchState()
    _feed(
        s,
        ("message", "user", {"text": "<system-reminder>stuff</system-reminder>"}),
        ("message", "user", {"text": "🧠 Note from this project's memory (past sessions): keep 7"}),
        ("message", "user", {"text": "🧠 Note #12 from this project's memory (past sessions and commits): keep 7"}),
    )
    assert s.pending is None and s.goal == ""


def test_a_plan_does_not_push_out_a_pending_request():
    s = W.WatchState()
    _feed(
        s,
        ("message", "user", {"text": "Fix the bug"}),
        ("message", "assistant", {"text": "I'll look at the rounding module first and then change the mode used."}),
    )
    assert s.pending is not None and s.pending.trigger == "task"


def test_reads_and_test_runs_are_context_but_a_changing_command_is_a_moment():
    s = W.WatchState()
    _feed(
        s,
        ("tool_call", "assistant", {"tool": "Bash", "call_id": "a", "input": {"command": "grep -rn round pay/"}}),
        ("tool_call", "assistant", {"tool": "Bash", "call_id": "b", "input": {"command": "python -m pytest -q"}}),
    )
    assert s.pending is None
    assert s.recent[-1].startswith("agent bash:")
    _feed(s, ("tool_call", "assistant", {"tool": "Bash", "call_id": "c", "input": {"command": "rm -rf build"}}))
    assert s.pending is not None and s.pending.trigger == "command" and s.pending.text == "rm -rf build"


def test_an_edit_is_a_moment_when_its_result_comes_back():
    s = W.WatchState()
    _feed(s, ("tool_call", "assistant", {"tool": "Edit", "call_id": "e1", "input": {"file_path": "pay/rounding.py"}}))
    assert s.pending is None  # the call alone changed nothing yet
    _feed(s, ("tool_result", "tool", {"call_id": "e1", "text": "updated"}))
    assert s.pending is not None and s.pending.trigger == "edit"
    assert s.recent[-1] == "agent edited pay/rounding.py"


def test_a_failure_is_a_moment_but_a_missing_tool_is_not():
    s = W.WatchState()
    _feed(
        s,
        ("tool_call", "assistant", {"tool": "Bash", "call_id": "x", "input": {"command": "make build"}}),
        ("tool_result", "tool", {"call_id": "x", "text": "python: command not found", "is_error": True}),
    )
    assert s.pending is not None and s.pending.trigger == "command"  # the command itself, not the failure
    s.pending = None
    _feed(
        s,
        ("tool_call", "assistant", {"tool": "Bash", "call_id": "y", "input": {"command": "make build"}}),
    )
    s.pending = None
    _feed(s, ("tool_result", "tool", {"call_id": "y", "text": "Traceback (most recent call last):\n boom"}))
    assert s.pending is not None and s.pending.trigger == "failure"


def test_a_state_event_is_the_change_so_far_and_makes_no_moment():
    s = W.WatchState()
    _feed(s, ("state", "system", {"data": {"diff": "+++ b/pay/rounding.py\n+x = 1"}}))
    assert s.diff.startswith("+++ b/pay") and s.pending is None and not s.recent


def test_take_moment_carries_the_steps_and_change_it_is_judged_against():
    s = W.WatchState()
    _feed(
        s,
        ("message", "user", {"text": "Fix the bug"}),
        ("state", "system", {"data": {"diff": "diff"}}),
    )
    m = W.take_moment(s)
    assert m is not None and m.trigger == "task" and m.diff == "diff" and s.pending is None
    assert W.take_moment(s) is None


def test_the_same_situation_is_judged_once_and_notes_stop_at_the_limit_of_a_request():
    s = W.WatchState()
    m = W.Moment(trigger="plan", text="I will set the retry to 10", recent=[], diff="")
    assert W.skip_reason(s, m) is None
    assert W.skip_reason(s, m) == "already judged"
    edit = W.Moment(trigger="edit", text="", recent=[], diff="d")
    rules = ["Retries stay at 7.", "Never log email addresses.", "Dates are stored in UTC."]
    for i, rule in enumerate(rules[: W.MAX_NOTES]):
        assert W.accept_note(s, edit, rule, [f"m{i}"])
    other = W.Moment(trigger="plan", text="something else entirely", recent=[], diff="")
    assert W.skip_reason(s, other) == "note limit reached"
    # The limit is per request: a new request from the user opens it again.
    W.read_step(s, W.Step(kind="ask", text="now fix the login page"))
    assert W.skip_reason(s, other) is None


def test_a_memory_is_said_once_but_a_heads_up_does_not_use_it_up():
    s = W.WatchState()
    task = W.Moment(trigger="task", text="Fix it", recent=[], diff="")
    edit = W.Moment(trigger="edit", text="", recent=[], diff="d")
    assert W.accept_note(s, task, "Before you change x: keep 7", ["m1"])
    assert s.evidence_sent == []
    assert W.accept_note(s, edit, "You set 10, the rule is 7", ["m1"])
    assert not W.accept_note(s, edit, "Again: the rule is 7", ["m1"])
    assert not W.accept_note(s, edit, "   ", ["m2"])
    assert s.sent == ["Before you change x: keep 7", "You set 10, the rule is 7"]


def test_a_conflict_restating_an_earlier_one_is_not_sent_again_under_new_memories():
    """Beta: one new article was flagged three times within minutes, each note from a memory the
    session's own saves had just rewritten (new ids, so the id guard let it through)."""
    s = W.WatchState()
    edit = W.Moment(trigger="edit", text="Write content/articles/voice-agent-memory.md", recent=[], diff="")
    first = 'The rule is "instead of generating articles automatically." Adding voice-agent-memory.md contradicts it.'
    again = 'Conflict: the rule is to improve the strongest remaining pages "instead of generating articles automatically." Adding voice-agent-memory.md goes against it.'
    other = (
        "Change content/articles/hermes-agent-plugins.md: make v0.22.4 a standalone sentence with its source and date."
    )
    assert W.accept_note(s, edit, first, ["m1"])
    assert not W.accept_note(s, edit, again, ["m2"])  # the same rule, a new memory id
    assert W.accept_note(s, edit, other, ["m3"])  # a different rule still goes
    # A heads-up is not compared: the same rule after it, as a conflict, is worth saying.
    t = W.WatchState()
    task = W.Moment(trigger="task", text="Fix x", recent=[], diff="")
    assert W.accept_note(t, task, "Before you change x: keep a 15-folder baseline", ["h1"])
    assert W.accept_note(t, edit, "Keep a 15-folder baseline: the plan drops it", ["h2"])


def test_a_request_is_searched_by_its_sentences_naming_code_then_its_paragraphs():
    m = W.Moment(
        trigger="task",
        text="You are a maintainer.\n\nFix `round_cents` so half cents round down. The tests show it.\n\n"
        "Work efficiently and do not modify the tests in this repository please.",
        recent=[],
        diff="",
    )
    queries = W.search_queries(m)
    assert queries[0] == "Fix `round_cents` so half cents round down."
    assert any(q.startswith("Work efficiently") for q in queries)


def test_a_command_is_searched_by_its_start():
    long = "kubectl get sts hindsight-analytics -o jsonpath='{.spec}' && " + "echo x; " * 300
    m = W.Moment(trigger="command", text=long, recent=[], diff="")
    assert W.search_queries(m) == [long[: W.COMMAND_QUERY_CAP]]


def test_an_edit_is_searched_by_its_diff_in_words():
    m = W.Moment(trigger="edit", text="", recent=[], diff="+++ b/pay/r.py\n@@ def f():\n-x = 1\n+x = 2\n")
    assert W.search_queries(m) == ["Change to pay/r.py (def f():):\n-x = 1\n+x = 2"]


def test_the_situation_speaks_as_the_agent_and_lists_what_it_was_told():
    m = W.Moment(trigger="command", text="rm -rf build", recent=["agent runs: rm -rf build"], diff="+x")
    text = W.situation("Fix it", m, ["keep 7"])
    assert text.startswith("MY REQUEST:\nFix it")
    assert "WHAT I JUST DID (command):\nrm -rf build" in text
    assert "+x" not in text  # a command is judged on its own; each edit was judged by its diff
    assert text.endswith("ALREADY TOLD ME:\nkeep 7")


def test_the_quote_prefers_the_sentence_with_the_note_value():
    source = "The current setting is 10 retries for now. We decided to keep the retry count at 7 for ledger reasons."
    assert W.best_sentence(source, "Keep the retry count at 7").startswith("We decided")


def test_a_saved_chat_reads_as_its_words():
    assert W.readable('{"role": "user", "content": "keep 7"}\nplain') == "keep 7\nplain"
    assert W.readable('[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]') == "a\nb"


def test_the_end_of_a_turn_drops_a_pending_closing_message_but_not_an_edit():
    s = W.WatchState()
    _feed(
        s,
        ("message", "assistant", {"text": "All tests pass. I changed the rounding to ROUND_HALF_DOWN as recorded."}),
        ("turn_end", "assistant", {}),
    )
    assert s.pending is None
    _feed(
        s,
        ("tool_call", "assistant", {"tool": "Edit", "call_id": "e", "input": {"file_path": "a.py"}}),
        ("tool_result", "tool", {"call_id": "e", "text": "ok"}),
        ("turn_end", "assistant", {}),
    )
    assert s.pending is not None and s.pending.trigger == "edit"


def test_the_same_request_with_other_whitespace_is_judged_once_and_a_note_never_repeats_word_for_word():
    s = W.WatchState()
    a = W.Moment(trigger="task", text="Fix the rounding of half cents.", recent=[], diff="")
    b = W.Moment(trigger="task", text="Fix the rounding of half cents.\n", recent=[], diff="")
    assert W.skip_reason(s, a) is None and W.skip_reason(s, b) == "already judged"
    assert W.accept_note(s, a, "Before you change round_cents: half cents round down.", ["m1"])
    assert not W.accept_note(s, a, "Before you change round_cents: half cents round down.", ["m2"])


def test_tools_are_known_by_what_their_names_say_not_by_one_agents_names():
    for name in ("Edit", "MultiEdit", "apply_patch", "str_replace_editor", "write_file", "patch"):
        assert W.tool_family(name) == "edit", name
    for name in ("Bash", "shell", "exec_command", "terminal"):
        assert W.tool_family(name) == "command", name
    for name in ("Read", "read_file", "search_files", "TodoWrite", "dispatch_task"):
        assert W.tool_family(name) == "other", name


def test_an_edit_without_a_diff_is_judged_by_its_own_text():
    s = W.WatchState()
    _feed(
        s,
        (
            "tool_call",
            "assistant",
            {"tool": "patch", "call_id": "p", "input": {"path": "pay.py", "new_string": "HALF_UP"}},
        ),
        ("tool_result", "tool", {"call_id": "p", "text": "ok"}),
    )
    m = W.take_moment(s)
    assert m is not None and m.trigger == "edit" and W.search_queries(m) == ["Change to pay.py:\nHALF_UP"]
    assert W.skip_reason(s, m) is None


def test_a_read_after_a_cd_on_its_own_line_is_not_a_moment():
    """Beta: `cd <repo>` then `git grep` / `gh run view` on the next line ran a check each (11 of 14)."""
    for cmd in (
        "cd /repo\ngit grep -nE '^FROM ' -- '*Dockerfile*'",
        "cd /repo\ngh run view 39605604118 --json jobs",
        "cd /repo\ngit fetch -q origin main && git show origin/main:uv.lock",
        "command -v docker >/dev/null && docker info",
    ):
        assert W.READ_ONLY_CMD.match(cmd), cmd
    for cmd in ("cd /repo\nrm -rf build", "cd /repo\ngit push origin main", "gh pr merge 63", "gh run rerun 1"):
        assert not W.READ_ONLY_CMD.match(cmd), cmd


def test_a_request_that_names_no_work_gets_no_heads_up():
    """'/ship-it' searched memory by its own words and got a note about whatever shared them."""

    def reason(text: str) -> str | None:
        return W.skip_reason(W.WatchState(), W.Moment(trigger="task", text=text, recent=[], diff=""))

    assert reason("/ship-it") == "request names no work"
    assert reason("obs cleanup? https://github.com/acme/app/issues/12") == "request names no work"
    assert reason("can you fix 5969 and /ship-it") is None
    assert W.skip_reason(W.WatchState(), W.Moment(trigger="command", text="ls", recent=[], diff="")) is None


def test_a_saved_reply_leaves_out_the_lines_about_notes():
    """Saved, the agent's 'per Hindsight note #129' lines came back as rules (beta)."""
    from datetime import UTC, datetime

    from hindsight_api.engine.sessions.service import _turn

    reply = (
        "Freed 12 GB from the build caches.\n"
        "> 🧠 **Thanks to Hindsight thought #129**, I left ~/.cache/huggingface alone.\n"
        "> 🧠 *Hindsight thought #130, set aside* — the 15-folder list; does not apply: no drop.\n"
        "> 🧠 **From Hindsight memory (live session #131)** — K is 1024."
    )
    turn = _turn("message", {"role": "assistant", "text": reply}, datetime.now(UTC))
    assert turn is not None and turn.content == "Freed 12 GB from the build caches."
    only = _turn(
        "message", {"role": "assistant", "text": "> 🧠 *Hindsight thought #9, already done* — x"}, datetime.now(UTC)
    )
    assert only is None


def test_a_rule_the_agent_set_aside_is_not_raised_again_and_the_check_is_told():
    """Beta: one rule set aside twice ("only my suggestion") came back a third time, reworded."""
    s = W.WatchState(
        set_aside=["The recorded rule is: the real fix is to sign the app with a fixed certificate of its own."]
    )
    edit = W.Moment(trigger="edit", text="", recent=[], diff="d")
    again = "This misses the recorded fix: sign the app with a fixed certificate of its own."
    other = "Retries stay at 7 for the upload client."
    assert not W.accept_note(s, edit, again, ["m1"])
    assert W.accept_note(s, edit, other, ["m2"])
    ctx = W.situation("fix it", edit, [], s.set_aside)
    assert "I SET THESE ASIDE OR ALREADY DID THEM (never raise them again):\nThe recorded rule is" in ctx


def test_the_file_a_request_points_to_becomes_part_of_the_request():
    """A scheduled job's request is 'Read JOB.md and do what it says': its rules live in the file."""
    s = W.WatchState()
    W.read_step(s, W.Step(kind="ask", text="You are running the job. Read /jobs/site/JOB.md and do what it says."))
    W.read_step(s, W.Step(kind="tool", id="1", tool="Bash", input={"command": "cat /jobs/site/JOB.md"}))
    W.read_step(s, W.Step(kind="result", id="1", text="This Bash command requires approval: cat"))
    W.read_step(s, W.Step(kind="tool", id="9", tool="Read", input={"file_path": "/jobs/site/JOB.md"}))
    W.read_step(s, W.Step(kind="result", id="9", text="Improve at most 2 pages. Links do not count."))
    W.read_step(s, W.Step(kind="tool", id="2", tool="Read", input={"file_path": "/other/README.md"}))
    W.read_step(s, W.Step(kind="result", id="2", text="unrelated"))
    assert s.request_file == "/jobs/site/JOB.md:\nImprove at most 2 pages. Links do not count."
    # A refused or short read never replaces the file; a longer read of it does.
    W.read_step(s, W.Step(kind="tool", id="3", tool="Bash", input={"command": "cat /jobs/site/JOB.md"}))
    W.read_step(s, W.Step(kind="result", id="3", text="denied", error=True))
    assert s.request_file.endswith("Links do not count.")
    m = W.Moment(trigger="command", text="git status", recent=[], diff="")
    assert "THE FILE MY REQUEST POINTS TO (its rules are part of my request and are current" in W.situation(
        s.goal, m, [], [], s.request_file
    )
    W.read_step(s, W.Step(kind="ask", text="now something else entirely, fix the footer links"))
    assert s.request_file == ""


def test_a_heads_up_about_code_the_request_never_mentions_is_dropped():
    """The answer model guesses which code a request will change; guessed wrong, the note was noise."""
    assert W.headsup_names_other_code(
        "Before you change indexer handling: keep NoManifest typed.", "handle the 404 pls"
    )
    assert W.headsup_names_other_code("Before you change the OpenClaw plugin: no 0.13.x.", "/ship-it")
    # A word the request shares, even misspelled, keeps it; a conflict is never judged this way.
    assert not W.headsup_names_other_code(
        "Before you change `bank-catalog-size`: hold off.", "so we skipped the bank cataglog size thing"
    )
    assert not W.headsup_names_other_code("You set 10; the rule is 7.", "anything")


def test_a_sessions_commands_are_checked_at_most_once_a_minute():
    from datetime import UTC, datetime, timedelta

    from hindsight_api.engine.sessions.service import _next_command_check

    now = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    s = W.WatchState()
    assert _next_command_check(s, now) is None  # never checked one: now
    s.last_command_check = (now - timedelta(seconds=20)).isoformat()
    assert _next_command_check(s, now) == now + timedelta(seconds=W.COMMAND_CHECK_EVERY_S - 20)
    s.last_command_check = (now - timedelta(seconds=W.COMMAND_CHECK_EVERY_S + 1)).isoformat()
    assert _next_command_check(s, now) is None


def test_a_heads_up_the_session_already_had_is_not_repeated_for_the_next_request():
    s = W.WatchState()
    task = W.Moment(trigger="task", text="Fix the dashboards page", recent=[], diff="")
    first = "Before you change the dashboards: agents save definitions over MCP; /admin/dashboards draws them with Vega-Lite."
    again = "Before you change the analytics dashboards: agents save dashboard definitions over MCP; /admin/dashboards draws them with Vega-Lite."
    assert W.accept_note(s, task, first, ["m1"])
    W.read_step(s, W.Step(kind="ask", text="now add a chart for retain latency to the dashboards"))
    assert not W.accept_note(s, task, again, ["m2"])
    assert W.accept_note(s, task, "Before you change retain metrics: latency is reported in ms, never s.", ["m3"])


def test_a_follow_up_keeps_the_earlier_requests_for_conflict_checks():
    """'ok continue' alone hid what the session was for: a conflict check sees the earlier requests too."""
    s = W.WatchState()
    _feed(
        s,
        ("message", "user", {"text": "Make the gateway rerank exactly like core's reranker"}),
        ("message", "user", {"text": "open the PR now"}),
        ("message", "user", {"text": "ok continue your work as expected"}),
    )
    assert s.earlier_goals == ["Make the gateway rerank exactly like core's reranker", "open the PR now"]
    cmd = W.Moment(trigger="command", text="git commit -m 'rerank like core'", recent=[], diff="")
    ctx = W.situation(s.goal, cmd, [], [], "", s.earlier_goals)
    assert ctx.startswith("EARLIER IN THIS SESSION I ASKED (still part of my request):\nMake the gateway")
    assert "EARLIER" not in W.situation(
        s.goal, W.Moment(trigger="task", text=s.goal, recent=[], diff=""), [], [], "", s.earlier_goals
    )


def test_a_conflict_quoting_a_rule_already_quoted_is_a_repeat():
    """The same quoted rule on another file, from a rewritten memory, does not go out again."""
    s = W.WatchState()
    cmd = W.Moment(trigger="command", text="python bench.py", recent=[], diff="")
    first = 'Change `bench.py`: do not use pairwise selection; keep selection "scoped to pointwise ranking only".'
    again = (
        "Change `docs/configuration.mdx`: keep selection \u201cscoped to pointwise ranking only\u201d; do not "
        'document "one small request per compared pair".'
    )
    other = 'Change `docs/configuration.mdx`: the default stays "prune candidates off by default".'
    assert W.accept_note(s, cmd, first, ["m1"])
    assert not W.accept_note(s, cmd, again, ["m2"])
    assert W.accept_note(s, cmd, other, ["m3"])


def test_a_heads_up_sharing_only_the_word_default_names_other_code():
    note = "Before you change the dev-alpha multi-loop default: ALPHA_EVENT_LOOPS>1 needs a newer core."
    assert W.headsup_names_other_code(note, "can we just enable it by default so we don't need any change?")
    assert not W.headsup_names_other_code(note, "make the dev-alpha loops default to 2")


def test_an_answer_saying_there_is_nothing_to_say_is_not_a_note():
    s = W.WatchState()
    cmd = W.Moment(trigger="command", text="echo summary >> cleanup.log", recent=[], diff="")
    assert not W.accept_note(s, cmd, "No conflict. The log entry is consistent with the recorded fact.", ["m1"])
    assert not W.accept_note(s, cmd, "Nothing.", ["m2"])
    assert W.accept_note(s, cmd, "None of the pruning may touch ~/dev: the job's rule keeps it.", ["m3"])
    assert W.accept_note(s, cmd, "Keep the floor at 60 GB, not 50.", ["m4"])
    agrees = "The user is not contradicting any decision; they listed the path as left, which aligns with the rule."
    assert not W.accept_note(s, cmd, agrees, ["m5"])
    following = 'You are following your request to update the pricing in INTERNAL.md to: "decision $0.02/M".'
    assert not W.accept_note(s, cmd, following, ["m7"])
    decided = 'You are following the decision that "Graphiti will not receive a separate article".'
    assert not W.accept_note(s, cmd, decided, ["m8"])
    latest = "You are following your own latest request: 'never let the decision model write the final text'."
    assert not W.accept_note(s, cmd, latest, ["m9"])
    follows = "You are waiting for the index check with a sleep 20 loop. This follows the rule in JOB.md lines 14-15."
    assert not W.accept_note(s, cmd, follows, ["m11"])
    narrated = (
        "The user is adding scenarios to test_consolidation.py. This aligns with the rule that each op has one file."
    )
    assert not W.accept_note(s, cmd, narrated, ["m10"])
    later = "The header is fine. Deleting ~/dev/archive contradicts the recorded keep-list for that folder."
    assert W.accept_note(s, cmd, later, ["m6"])


def test_a_command_started_in_the_background_says_so():
    """Claude Code backgrounds by a flag, so the command text alone looks like a foreground run."""
    s = W.WatchState()
    cmd = {"command": "uv run index_check.py > .tmp/index_check.txt 2>&1", "run_in_background": True}
    _feed(s, ("tool_call", "assistant", {"tool": "Bash", "input": cmd, "call_id": "c1"}))
    assert s.pending is not None and s.pending.text.endswith("# started in the background")


def test_a_command_the_harness_refused_is_not_a_failure():
    """An approval prompt or a quoting check is the harness, not the project failing."""
    for text in (
        "This Bash command contains multiple operations. The following parts require approval: cat JOB.md",
        "This command can't be checked before it runs",
        "Couldn't determine which page this action targets. Re-read tabs_context.",
    ):
        s = W.WatchState()
        _feed(
            s,
            ("tool_call", "assistant", {"tool": "Bash", "input": {"command": "cat JOB.md && ls"}, "call_id": "c"}),
            ("tool_result", "tool", {"text": text, "call_id": "c", "is_error": True}),
        )
        assert s.pending is None or s.pending.trigger != "failure", text


def test_a_busy_host_skips_checks(monkeypatch):
    """At a load the check would time out under, it waits instead of adding to the load."""
    s = W.WatchState()
    cmd = W.Moment(trigger="command", text="make build", recent=[], diff="")
    monkeypatch.setattr(W, "BUSY_LOAD_PER_CPU", 4.0)  # the autouse fixture turned it off
    monkeypatch.setattr(W.os, "getloadavg", lambda: (98.0, 80.0, 70.0))
    monkeypatch.setattr(W.os, "cpu_count", lambda: 14)
    assert W.skip_reason(s, cmd) == "host busy"
    monkeypatch.setattr(W.os, "getloadavg", lambda: (20.0, 20.0, 20.0))
    assert W.skip_reason(s, cmd) is None


def test_following_an_instruction_is_not_a_note():
    s = W.WatchState()
    cmd = W.Moment(trigger="command", text="gh pr checks 752", recent=[], diff="")
    assert not W.accept_note(s, cmd, "You are following your latest instruction to also wait for PR #752.", ["m1"])


def test_a_bare_literal_is_not_a_note():
    s = W.WatchState()
    cmd = W.Moment(trigger="command", text="make eval", recent=[], diff="")
    assert not W.accept_note(s, cmd, "AMB_RECALL_MAX_CHUNK_TOKENS=0", ["m1"])
    assert W.accept_note(s, cmd, "Keep `AMB_RECALL_MAX_CHUNK_TOKENS=0` so recall returns whole chunks.", ["m3"])
