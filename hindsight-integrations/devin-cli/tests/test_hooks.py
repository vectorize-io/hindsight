"""End-to-end tests for the recall and retain hooks.

These drive the hook `main()` / `run_retain()` entry points with fabricated
stdin and a fake Hindsight client, which is the only place the Devin-specific
wiring is observable: Devin CLI gives a hook `session_id` and nothing else, so
every path that Claude Code reaches via `transcript_path` or `cwd` here has to
come from the session database or from `DEVIN_PROJECT_DIR`.

The graceful-degradation cases matter more than usual. A hook that raises
returns a nonzero exit status, and Devin CLI reads exit code 2 as "block this
turn" — so a Hindsight outage must never propagate out of `main()`.
"""

import io
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from conftest import insert_message, make_memory


class HookEnv:
    """Handle for an isolated hook invocation environment.

    Only a few settings have `HINDSIGHT_*` env overrides, so tests that need to
    change anything else go through `configure()`, which rewrites the plugin's
    settings.json. `load_config()` reads it on every hook call, so this can be
    called right up until the call under test.
    """

    def __init__(self, root, sessions_db):
        self.root = root
        self.sessions_db = sessions_db
        self._settings = {}
        self._flush()

    def _flush(self):
        (self.root / "settings.json").write_text(json.dumps(self._settings))

    def configure(self, **settings):
        self._settings.update(settings)
        self._flush()
        return self


@pytest.fixture()
def hook_env(tmp_path, monkeypatch, sessions_db):
    """Isolate config, state, and the session DB for a hook invocation."""
    monkeypatch.setattr("lib.config.plugin_root", lambda: str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HINDSIGHT_DEVIN_SESSIONS_DB", str(sessions_db))
    for key in list(os.environ):
        if key.startswith("HINDSIGHT_") and key not in (
            "HINDSIGHT_DEVIN_CLI_DATA_DIR",
            "HINDSIGHT_DEVIN_SESSIONS_DB",
        ):
            monkeypatch.delenv(key, raising=False)
    env = HookEnv(tmp_path, sessions_db)
    # Retain every turn by default; the shipped default of 10 would gate out
    # every single-turn test below. The gate itself is covered explicitly.
    env.configure(retainEveryNTurns=1)
    return env


def _feed_stdin(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))


def _capture_stdout(monkeypatch) -> io.StringIO:
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    return buf


class TestRecallHook:
    def test_injects_memories_as_additional_context(self, hook_env, monkeypatch):
        import recall

        _feed_stdin(monkeypatch, {"prompt": "How do we deploy?", "session_id": "s1"})
        out = _capture_stdout(monkeypatch)

        client = MagicMock()
        client.recall.return_value = {"results": [make_memory("Deploys go through CI")]}
        with (
            patch("recall.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("recall.HindsightClient", return_value=client),
            patch("recall.ensure_bank_mission"),
        ):
            recall.main()

        payload = json.loads(out.getvalue())
        assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        context = payload["hookSpecificOutput"]["additionalContext"]
        assert "Deploys go through CI" in context
        assert context.startswith("<hindsight_memories>")
        assert context.endswith("</hindsight_memories>")

    def test_disabled_auto_recall_emits_nothing(self, hook_env, monkeypatch):
        import recall

        monkeypatch.setenv("HINDSIGHT_AUTO_RECALL", "false")
        _feed_stdin(monkeypatch, {"prompt": "How do we deploy?", "session_id": "s1"})
        out = _capture_stdout(monkeypatch)

        with patch("recall.HindsightClient") as client_cls:
            recall.main()

        assert out.getvalue() == ""
        client_cls.assert_not_called()

    def test_short_prompt_is_skipped(self, hook_env, monkeypatch):
        import recall

        _feed_stdin(monkeypatch, {"prompt": "hi", "session_id": "s1"})
        out = _capture_stdout(monkeypatch)

        with patch("recall.HindsightClient") as client_cls:
            recall.main()

        assert out.getvalue() == ""
        client_cls.assert_not_called()

    def test_no_results_emits_nothing(self, hook_env, monkeypatch):
        """An empty result set must not inject an empty memories block."""
        import recall

        _feed_stdin(monkeypatch, {"prompt": "How do we deploy?", "session_id": "s1"})
        out = _capture_stdout(monkeypatch)

        client = MagicMock()
        client.recall.return_value = {"results": []}
        with (
            patch("recall.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("recall.HindsightClient", return_value=client),
            patch("recall.ensure_bank_mission"),
        ):
            recall.main()

        assert out.getvalue() == ""

    def test_recall_api_failure_degrades_silently(self, hook_env, monkeypatch):
        """A Hindsight outage must not block the user's turn."""
        import recall

        _feed_stdin(monkeypatch, {"prompt": "How do we deploy?", "session_id": "s1"})

        client = MagicMock()
        client.recall.side_effect = OSError("connection refused")
        with (
            patch("recall.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("recall.HindsightClient", return_value=client),
            patch("recall.ensure_bank_mission"),
        ):
            recall.main()  # must not raise

    def test_unreachable_server_does_not_raise(self, hook_env, monkeypatch):
        import recall

        _feed_stdin(monkeypatch, {"prompt": "How do we deploy?", "session_id": "s1"})
        out = _capture_stdout(monkeypatch)

        with patch("recall.get_api_url", side_effect=RuntimeError("No Hindsight server")):
            recall.main()

        assert out.getvalue() == ""

    def test_malformed_stdin_does_not_raise(self, hook_env, monkeypatch):
        import recall

        monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))
        out = _capture_stdout(monkeypatch)

        recall.main()

        assert out.getvalue() == ""

    def test_multi_turn_query_reads_the_session_database(self, hook_env, monkeypatch, sessions_db):
        """recallContextTurns > 1 has no transcript file — it must use session_id.

        This is the Devin-specific substitution. If `session_id` stops being
        threaded from stdin into `read_session_messages`, the composed query
        silently degrades to the bare prompt and multi-turn recall quietly
        stops working.
        """
        import recall

        insert_message(sessions_db, "s1", 1, "m1", "user", "We use Kubernetes in prod")
        insert_message(sessions_db, "s1", 2, "m2", "assistant", "Noted, k8s in prod")
        monkeypatch.setenv("HINDSIGHT_RECALL_CONTEXT_TURNS", "3")
        _feed_stdin(monkeypatch, {"prompt": "How do we deploy?", "session_id": "s1"})
        _capture_stdout(monkeypatch)

        client = MagicMock()
        client.recall.return_value = {"results": []}
        with (
            patch("recall.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("recall.HindsightClient", return_value=client),
            patch("recall.ensure_bank_mission"),
        ):
            recall.main()

        query = client.recall.call_args.kwargs["query"]
        assert "Kubernetes" in query, (
            f"multi-turn recall must pull prior turns from the session DB via session_id; got query: {query!r}"
        )
        assert "How do we deploy?" in query

    def test_unknown_session_id_falls_back_to_the_bare_prompt(self, hook_env, monkeypatch):
        """A session DB miss degrades to single-turn recall, not a crash."""
        import recall

        monkeypatch.setenv("HINDSIGHT_RECALL_CONTEXT_TURNS", "3")
        _feed_stdin(monkeypatch, {"prompt": "How do we deploy?", "session_id": "nonexistent"})
        _capture_stdout(monkeypatch)

        client = MagicMock()
        client.recall.return_value = {"results": []}
        with (
            patch("recall.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("recall.HindsightClient", return_value=client),
            patch("recall.ensure_bank_mission"),
        ):
            recall.main()

        assert "How do we deploy?" in client.recall.call_args.kwargs["query"]

    def test_query_is_truncated_to_max_chars(self, hook_env, monkeypatch):
        import recall

        monkeypatch.setenv("HINDSIGHT_RECALL_MAX_QUERY_CHARS", "50")
        _feed_stdin(monkeypatch, {"prompt": "x" * 500, "session_id": "s1"})
        _capture_stdout(monkeypatch)

        client = MagicMock()
        client.recall.return_value = {"results": []}
        with (
            patch("recall.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("recall.HindsightClient", return_value=client),
            patch("recall.ensure_bank_mission"),
        ):
            recall.main()

        assert len(client.recall.call_args.kwargs["query"]) <= 50


class TestFilterByMinScores:
    """Score floors drop weak results before they reach the prompt."""

    def _config(self):
        return {"debug": False}

    def test_no_floors_passes_everything_through(self):
        import recall

        results = [{"text": "a", "scores": {"relevance": 0.1}}]
        assert recall.filter_by_min_scores(results, {}, self._config()) == results

    def test_result_below_floor_is_dropped(self):
        import recall

        results = [
            {"text": "weak", "scores": {"relevance": 0.1}},
            {"text": "strong", "scores": {"relevance": 0.9}},
        ]
        filtered = recall.filter_by_min_scores(results, {"relevance": 0.5}, self._config())
        assert [r["text"] for r in filtered] == ["strong"]

    def test_missing_score_field_is_not_filtered_out(self):
        """Absent is not the same as low — don't drop what we can't judge."""
        import recall

        results = [{"text": "unscored", "scores": {}}]
        filtered = recall.filter_by_min_scores(results, {"relevance": 0.5}, self._config())
        assert filtered == results

    def test_invalid_floor_is_ignored_rather_than_fatal(self):
        import recall

        results = [{"text": "a", "scores": {"relevance": 0.1}}]
        filtered = recall.filter_by_min_scores(results, {"relevance": "not-a-number"}, self._config())
        assert filtered == results

    @pytest.mark.parametrize("bad", ["nan", "NaN", float("nan")])
    def test_a_nan_floor_is_reported_rather_than_applied(self, bad, capsys):
        """float() accepts "nan", and every `value < nan` comparison is False.

        Note what this does *not* change: a NaN floor never dropped anything,
        and rejecting it does not drop anything either, so the filtering
        outcome is identical. What changes is that a broken floor now says so.
        Silently accepting it left the operator with a configured floor that
        did nothing and no way to find out — the whole cost of this bug is
        diagnostic, so that is what the test asserts on.
        """
        import recall

        results = [{"text": "weak", "scores": {"relevance": 0.1}}]
        filtered = recall.filter_by_min_scores(results, {"relevance": bad}, {"debug": True})

        assert filtered == results
        assert "non-finite" in capsys.readouterr().err, "a NaN floor was accepted as a working floor"

    def test_an_infinite_floor_is_rejected_rather_than_honoured(self):
        """Unlike NaN, `inf` does change the outcome: it rejects everything.

        A floor no result can clear is a config typo rather than an intent, so
        it is dropped along with NaN instead of being obeyed literally.
        """
        import recall

        results = [{"text": "a", "scores": {"relevance": 0.9}}]
        filtered = recall.filter_by_min_scores(results, {"relevance": "inf"}, self._config())
        assert filtered == results


class TestRetainHook:
    def test_retains_session_messages_read_from_the_database(self, hook_env, sessions_db):
        import retain

        insert_message(sessions_db, "s1", 1, "m1", "user", "Deploy via CI")
        insert_message(sessions_db, "s1", 2, "m2", "assistant", "Understood")

        client = MagicMock()
        client.retain.return_value = {"status": "ok"}
        with (
            patch("retain.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("retain.HindsightClient", return_value=client),
            patch("retain.ensure_bank_mission"),
        ):
            retain.run_retain({"session_id": "s1"})

        kwargs = client.retain.call_args.kwargs
        assert "Deploy via CI" in kwargs["content"]
        assert "Understood" in kwargs["content"]
        assert kwargs["document_id"] == "s1"
        assert kwargs["context"] == "devin-cli"

    def test_empty_session_skips_the_api_call(self, hook_env):
        """A DB miss (or a schema change) must be a no-op, not an empty retain."""
        import retain

        client = MagicMock()
        with (
            patch("retain.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("retain.HindsightClient", return_value=client),
            patch("retain.ensure_bank_mission"),
        ):
            retain.run_retain({"session_id": "nonexistent"})

        client.retain.assert_not_called()

    def test_disabled_auto_retain_skips_everything(self, hook_env, monkeypatch, sessions_db):
        import retain

        insert_message(sessions_db, "s1", 1, "m1", "user", "Deploy via CI")
        monkeypatch.setenv("HINDSIGHT_AUTO_RETAIN", "false")

        with patch("retain.HindsightClient") as client_cls:
            retain.run_retain({"session_id": "s1"})

        client_cls.assert_not_called()

    def test_retain_failure_degrades_silently(self, hook_env, sessions_db):
        import retain

        insert_message(sessions_db, "s1", 1, "m1", "user", "Deploy via CI")

        client = MagicMock()
        client.retain.side_effect = OSError("connection refused")
        with (
            patch("retain.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("retain.HindsightClient", return_value=client),
            patch("retain.ensure_bank_mission"),
        ):
            retain.run_retain({"session_id": "s1"})  # must not raise

    def test_session_id_tag_template_is_resolved(self, hook_env, sessions_db):
        """settings.json ships retainTags: ["{session_id}"] — it must expand."""
        import retain

        insert_message(sessions_db, "s1", 1, "m1", "user", "Deploy via CI")
        hook_env.configure(retainTags=["{session_id}", "static"])

        client = MagicMock()
        client.retain.return_value = {"status": "ok"}
        with (
            patch("retain.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("retain.HindsightClient", return_value=client),
            patch("retain.ensure_bank_mission"),
        ):
            retain.run_retain({"session_id": "s1"})

        assert client.retain.call_args.kwargs["tags"] == ["s1", "static"]

    def test_metadata_records_the_session_id(self, hook_env, sessions_db):
        import retain

        insert_message(sessions_db, "s1", 1, "m1", "user", "Deploy via CI")

        client = MagicMock()
        client.retain.return_value = {"status": "ok"}
        with (
            patch("retain.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("retain.HindsightClient", return_value=client),
            patch("retain.ensure_bank_mission"),
        ):
            retain.run_retain({"session_id": "s1"})

        assert client.retain.call_args.kwargs["metadata"]["session_id"] == "s1"

    def test_second_retain_only_sends_new_messages(self, hook_env, sessions_db):
        """Incremental retain: state tracks how far we got, so turns aren't resent."""
        import retain

        insert_message(sessions_db, "s1", 1, "m1", "user", "First question")
        insert_message(sessions_db, "s1", 2, "m2", "assistant", "First answer")

        client = MagicMock()
        client.retain.return_value = {"status": "ok"}
        patches = (
            patch("retain.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("retain.HindsightClient", return_value=client),
            patch("retain.ensure_bank_mission"),
        )
        with patches[0], patches[1], patches[2]:
            retain.run_retain({"session_id": "s1"})
            insert_message(sessions_db, "s1", 3, "m3", "user", "Second question")
            insert_message(sessions_db, "s1", 4, "m4", "assistant", "Second answer")
            retain.run_retain({"session_id": "s1"})

        second_content = client.retain.call_args.kwargs["content"]
        assert "Second question" in second_content
        assert "First question" not in second_content, "already-retained turns must not be resent on the next Stop hook"

    def test_no_new_messages_skips_the_api_call(self, hook_env, sessions_db):
        import retain

        insert_message(sessions_db, "s1", 1, "m1", "user", "Only question")

        client = MagicMock()
        client.retain.return_value = {"status": "ok"}
        with (
            patch("retain.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("retain.HindsightClient", return_value=client),
            patch("retain.ensure_bank_mission"),
        ):
            retain.run_retain({"session_id": "s1"})
            assert client.retain.call_count == 1
            retain.run_retain({"session_id": "s1"})
            assert client.retain.call_count == 1, "no new turns → no second retain"

    def test_retain_every_n_turns_gates_the_api_call(self, hook_env, sessions_db):
        import retain

        insert_message(sessions_db, "s1", 1, "m1", "user", "Question")
        hook_env.configure(retainEveryNTurns=3)

        client = MagicMock()
        client.retain.return_value = {"status": "ok"}
        with (
            patch("retain.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("retain.HindsightClient", return_value=client),
            patch("retain.ensure_bank_mission"),
        ):
            retain.run_retain({"session_id": "s1"})
            assert client.retain.call_count == 0, "turn 1 of 3 must not retain"
            retain.run_retain({"session_id": "s1"})
            assert client.retain.call_count == 0, "turn 2 of 3 must not retain"
            retain.run_retain({"session_id": "s1"})
            assert client.retain.call_count == 1, "turn 3 of 3 must retain"

    def test_force_bypasses_the_turn_gate(self, hook_env, sessions_db):
        """SessionEnd passes force=True so a partial window still gets retained."""
        import retain

        insert_message(sessions_db, "s1", 1, "m1", "user", "Question")
        hook_env.configure(retainEveryNTurns=10)

        client = MagicMock()
        client.retain.return_value = {"status": "ok"}
        with (
            patch("retain.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("retain.HindsightClient", return_value=client),
            patch("retain.ensure_bank_mission"),
        ):
            retain.run_retain({"session_id": "s1"}, force=True)

        assert client.retain.call_count == 1

    def test_a_negative_overlap_cannot_empty_the_retain_window(self, hook_env, sessions_db):
        """load_config guarantees an int, not a sensible one.

        `window_turns = retain_every_n + overlap_turns` feeds
        slice_last_turns_by_user_boundary(), which returns [] for turns <= 0 —
        and an empty transcript is skipped. So a negative overlap switches
        retention off silently, including on the forced session-end retain.
        Overlap can only ever widen the window.
        """
        import retain

        insert_message(sessions_db, "s1", 1, "m1", "user", "Question")
        insert_message(sessions_db, "s1", 2, "m2", "assistant", "Answer")
        hook_env.configure(retainMode="chunked", retainEveryNTurns=2, retainOverlapTurns=-10)

        client = MagicMock()
        client.retain.return_value = {"status": "ok"}
        with (
            patch("retain.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("retain.HindsightClient", return_value=client),
            patch("retain.ensure_bank_mission"),
        ):
            retain.run_retain({"session_id": "s1"}, force=True)

        assert client.retain.call_count == 1, "a negative retainOverlapTurns silently disabled retention"
        assert "Question" in client.retain.call_args.kwargs["content"]


class TestToolCallFolding:
    """Devin CLI stores tool calls beside content; content.py wants inline blocks."""

    def test_folding_is_skipped_when_disabled(self):
        import retain

        messages = [{"role": "assistant", "content": "hi", "tool_calls": [{"function": {"name": "x"}}]}]
        assert retain._apply_tool_call_folding(messages, False) is messages

    def test_folding_converts_tool_calls_to_content_blocks(self):
        import retain

        messages = [
            {
                "role": "assistant",
                "content": "Running it",
                "tool_calls": [{"function": {"name": "shell", "arguments": json.dumps({"cmd": "ls"})}}],
            }
        ]
        folded = retain._apply_tool_call_folding(messages, True)
        blocks = folded[0]["content"]
        assert {"type": "text", "text": "Running it"} in blocks
        assert {"type": "tool_use", "name": "shell", "input": {"cmd": "ls"}} in blocks

    def test_messages_without_tool_calls_pass_through_unchanged(self):
        import retain

        messages = [{"role": "user", "content": "plain"}]
        assert retain._apply_tool_call_folding(messages, True) == messages

    def test_retained_content_includes_tool_calls_when_enabled(self, hook_env, sessions_db):
        import retain

        insert_message(
            sessions_db,
            "s1",
            1,
            "m1",
            "assistant",
            "Running it",
            tool_calls=[{"function": {"name": "shell", "arguments": json.dumps({"cmd": "ls -la"})}}],
        )
        hook_env.configure(retainToolCalls=True)

        client = MagicMock()
        client.retain.return_value = {"status": "ok"}
        with (
            patch("retain.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("retain.HindsightClient", return_value=client),
            patch("retain.ensure_bank_mission"),
        ):
            retain.run_retain({"session_id": "s1"})

        assert "shell" in client.retain.call_args.kwargs["content"]


class TestRecallQueryLimitDisabled:
    def test_zero_max_query_chars_sends_the_full_prompt(self, hook_env, monkeypatch):
        """`recallMaxQueryChars: 0` is documented as "no limit".

        truncate_recall_query() honours that and returns early, but main() then
        re-applied the cap unguarded — turning the query into query[:0], so every
        recall searched for an empty string and returned nothing.
        """
        import recall

        hook_env.configure(recallMaxQueryChars=0)
        prompt = "How do we deploy? " + "x" * 2000
        _feed_stdin(monkeypatch, {"prompt": prompt, "session_id": "s1"})
        _capture_stdout(monkeypatch)

        client = MagicMock()
        client.recall.return_value = {"results": []}
        with (
            patch("recall.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("recall.HindsightClient", return_value=client),
            patch("recall.ensure_bank_mission"),
        ):
            recall.main()

        sent_query = client.recall.call_args.kwargs.get("query") or client.recall.call_args.args[0]
        assert sent_query == prompt


class TestDiagnosticStateFailureCannotSwallowRecall:
    """write_state() raises on an unwritable state dir — deliberately.

    But recall.py persists a purely diagnostic breadcrumb *before* printing its
    response, so letting that escape would drop the memories the user already
    paid for: recall succeeded, the API call was made, and the hook still
    emitted nothing.
    """

    def test_memories_are_still_emitted_when_the_state_write_fails(self, hook_env, monkeypatch):
        import recall

        _feed_stdin(monkeypatch, {"prompt": "How do we deploy?", "session_id": "s1"})
        out = _capture_stdout(monkeypatch)

        def _explode(*_args, **_kwargs):
            raise OSError("read-only file system")

        client = MagicMock()
        client.recall.return_value = {"results": [make_memory("Deploys go through CI")]}
        with (
            patch("recall.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("recall.HindsightClient", return_value=client),
            patch("recall.ensure_bank_mission"),
            patch("recall.write_state", _explode),
        ):
            recall.main()

        payload = json.loads(out.getvalue())
        assert "Deploys go through CI" in payload["hookSpecificOutput"]["additionalContext"]


class TestIncrementalRetainKeepsEveryNewMessage:
    """The checkpoint already narrows the slice — narrowing it twice loses work.

    `retain_full_window` was set from `start_index == 0`, so every retain after
    the first asked the formatter for "the last turn only" *within* a slice
    that was already exactly the unretained messages. Whatever fell outside
    that last turn was dropped, and the empty-transcript path then committed
    the full message count, so no later run retained it either.
    """

    def _retain(self, session_id):
        import retain

        client = MagicMock()
        client.retain.return_value = {"status": "ok"}
        with (
            patch("retain.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("retain.HindsightClient", return_value=client),
            patch("retain.ensure_bank_mission"),
        ):
            retain.run_retain({"session_id": session_id})
        return client

    def test_an_assistant_only_delta_is_still_retained(self, hook_env, sessions_db):
        """A slice with no user message at all used to format to nothing."""
        insert_message(sessions_db, "s1", 1, "m1", "user", "Deploy via CI")
        insert_message(sessions_db, "s1", 2, "m2", "assistant", "Understood")
        self._retain("s1")

        insert_message(sessions_db, "s1", 3, "m3", "assistant", "Follow-up detail")
        client = self._retain("s1")

        client.retain.assert_called_once()
        assert "Follow-up detail" in client.retain.call_args.kwargs["content"]

    def test_a_delta_spanning_two_turns_retains_both(self, hook_env, sessions_db):
        """After a failed retain the slice covers more than one turn."""
        insert_message(sessions_db, "s1", 1, "m1", "user", "First question")
        insert_message(sessions_db, "s1", 2, "m2", "assistant", "First answer")
        self._retain("s1")

        insert_message(sessions_db, "s1", 3, "m3", "user", "Second question")
        insert_message(sessions_db, "s1", 4, "m4", "assistant", "Second answer")
        insert_message(sessions_db, "s1", 5, "m5", "user", "Third question")
        insert_message(sessions_db, "s1", 6, "m6", "assistant", "Third answer")
        client = self._retain("s1")

        content = client.retain.call_args.kwargs["content"]
        assert "Second question" in content, "an earlier turn in the delta was silently dropped"
        assert "Third question" in content

    def test_an_assistant_only_delta_is_not_checkpointed_away(self, hook_env, sessions_db):
        """The regression's real cost: no later run could recover it either.

        Asserted across every retain in the session rather than on the last
        one. Checking only the final call passes either way — the message is
        absent there whether it was retained earlier (correct) or checkpointed
        away without ever being sent (the bug).
        """
        insert_message(sessions_db, "s1", 1, "m1", "user", "Deploy via CI")
        insert_message(sessions_db, "s1", 2, "m2", "assistant", "Understood")
        sent = list(self._retain("s1").retain.call_args_list)

        insert_message(sessions_db, "s1", 3, "m3", "assistant", "Follow-up detail")
        sent += self._retain("s1").retain.call_args_list

        insert_message(sessions_db, "s1", 4, "m4", "user", "Next question")
        sent += self._retain("s1").retain.call_args_list

        assert any("Follow-up detail" in call.kwargs["content"] for call in sent), (
            "the assistant delta was checkpointed past without ever being retained"
        )


class TestMalformedOptionalSettingsCannotDisableRecall:
    """A bad optional setting should cost that setting, not the whole hook.

    These values come straight from user-edited JSON, so they can be any
    shape. Each used to reach a `.get()` or `.items()` outside any try block,
    raising out of `main()` — so recall emitted nothing at all, and a setting
    the user added to *tune* recall silently switched it off.
    """

    @pytest.mark.parametrize("bad", [[], "high", 0.5, True])
    def test_a_non_object_min_scores_does_not_disable_recall(self, hook_env, monkeypatch, bad):
        import recall

        hook_env.configure(recallMinScores=bad)
        _feed_stdin(monkeypatch, {"prompt": "How do we deploy?", "session_id": "s1"})
        out = _capture_stdout(monkeypatch)

        client = MagicMock()
        client.recall.return_value = {"results": [make_memory("Deploys go through CI")]}
        with (
            patch("recall.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("recall.HindsightClient", return_value=client),
            patch("recall.ensure_bank_mission"),
        ):
            recall.main()

        assert "Deploys go through CI" in out.getvalue()

    @pytest.mark.parametrize("bad", [None, "tags", ["a"]])
    def test_a_malformed_bank_filter_costs_only_that_bank(self, hook_env, monkeypatch, bad):
        """The primary bank's memories were being discarded along with it."""
        import recall

        hook_env.configure(recallAdditionalBanks=["other"], recallAdditionalBankFilters={"other": bad})
        _feed_stdin(monkeypatch, {"prompt": "How do we deploy?", "session_id": "s1"})
        out = _capture_stdout(monkeypatch)

        client = MagicMock()
        client.recall.return_value = {"results": [make_memory("Deploys go through CI")]}
        with (
            patch("recall.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("recall.HindsightClient", return_value=client),
            patch("recall.ensure_bank_mission"),
        ):
            recall.main()

        assert "Deploys go through CI" in out.getvalue()

    def test_a_non_object_filter_container_is_also_tolerated(self, hook_env, monkeypatch):
        """The container is read before any per-entry guard can run."""
        import recall

        hook_env.configure(recallAdditionalBanks=["other"], recallAdditionalBankFilters=["not", "a", "dict"])
        _feed_stdin(monkeypatch, {"prompt": "How do we deploy?", "session_id": "s1"})
        out = _capture_stdout(monkeypatch)

        client = MagicMock()
        client.recall.return_value = {"results": [make_memory("Deploys go through CI")]}
        with (
            patch("recall.get_api_url", return_value="http://127.0.0.1:9078"),
            patch("recall.HindsightClient", return_value=client),
            patch("recall.ensure_bank_mission"),
        ):
            recall.main()

        assert "Deploys go through CI" in out.getvalue()


class TestHooksNeverExitNonZero:
    """A nonzero hook exit is a blocking error — it stops the user's turn.

    Both hooks document "0 — always (graceful degradation on any error)" and
    both broke it under `debug`, which let a diagnostic flag change control
    flow: turning debug on to investigate a recall failure escalated it from
    "no memories this turn" to a rejected prompt, and so changed the behaviour
    of the very failure being investigated.

    Asserted on source because the guarded path is the `__main__` block, which
    a subprocess would have to be induced to fail from the outside. The
    property is syntactic anyway — there is exactly one construct by which a
    hook could exit nonzero, so grepping for it is a complete check rather
    than a proxy for one.
    """

    @pytest.mark.parametrize("script", ["recall.py", "retain.py", "session_start.py", "session_end.py"])
    def test_no_hook_script_can_exit_nonzero(self, script):
        path = os.path.join(os.path.dirname(__file__), "..", "scripts", script)
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()

        offenders = [ln.strip() for ln in lines if "sys.exit(" in ln and "sys.exit(0)" not in ln]

        assert not offenders, f"{script} can exit nonzero, which blocks the turn: {offenders}"
