"""End-to-end tests for the Cursor CLI hook scripts.

Mocks the Cursor hook runtime:
  - stdin  → io.StringIO(json.dumps(hook_input))
  - stdout → io.StringIO() captured for assertions
  - urllib.request.urlopen → fake HTTP responses
  - HOME → tmp_path (isolates ~/.hindsight/cursor-cli.json and state)
"""

import importlib.util
import io
import json
import os
import sys
from unittest.mock import patch

import pytest
from conftest import SCRIPTS_DIR, FakeHTTPResponse, make_hook_input, make_memory, make_transcript_file, make_user_config


def _run_hook(
    module_name,
    hook_input,
    monkeypatch,
    tmp_path,
    urlopen_side_effect=None,
    user_config=None,
    env_overrides=None,
    set_default_api_url=True,
):
    """Import and run a hook script's main() with mocked stdin/stdout/HTTP."""
    monkeypatch.setenv("HOME", str(tmp_path))

    for k in list(os.environ):
        if k.startswith("HINDSIGHT_"):
            monkeypatch.delenv(k, raising=False)

    if set_default_api_url:
        monkeypatch.setenv("HINDSIGHT_API_URL", "http://fake:9077")

    if env_overrides:
        for k, v in env_overrides.items():
            monkeypatch.setenv(k, v)

    cfg = {"retainEveryNTurns": 1, "autoRecall": True, "autoRetain": True}
    if user_config:
        cfg.update(user_config)
    make_user_config(tmp_path, cfg)

    stdin_data = io.StringIO(json.dumps(hook_input))
    stdout_capture = io.StringIO()

    spec = importlib.util.spec_from_file_location(
        module_name + "_fresh", os.path.join(SCRIPTS_DIR, f"{module_name}.py")
    )
    mod = importlib.util.module_from_spec(spec)

    default_response = FakeHTTPResponse({"results": []})
    side_effect = urlopen_side_effect or (lambda *a, **kw: default_response)

    with (
        patch("sys.stdin", stdin_data),
        patch("sys.stdout", stdout_capture),
        patch("urllib.request.urlopen", side_effect=side_effect),
    ):
        spec.loader.exec_module(mod)
        mod.main()

    return stdout_capture.getvalue()


# ---------------------------------------------------------------------------
# recall hook (beforeSubmitPrompt)
# ---------------------------------------------------------------------------


class TestRecallHook:
    def test_outputs_additional_context_when_memories_found(self, monkeypatch, tmp_path):
        memory = make_memory("Paris is the capital of France", "world")
        response = FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(prompt="What is the capital of France?")
        output = _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *a, **kw: response,
        )

        data = json.loads(output)
        assert data["continue"] is True
        assert "additional_context" in data
        assert "Paris is the capital of France" in data["additional_context"]
        assert "<hindsight_memories>" in data["additional_context"]

    def test_no_output_when_no_memories(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(prompt="hello there world")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path)
        assert output.strip() == ""

    def test_no_output_for_short_prompt(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(prompt="hi")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path)
        assert output.strip() == ""

    def test_graceful_on_api_error(self, monkeypatch, tmp_path):
        def raise_error(*a, **kw):
            raise OSError("connection refused")

        hook_input = make_hook_input(prompt="What is my project about?")
        output = _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=raise_error,
        )
        assert output.strip() == ""

    def test_output_format_matches_cursor_spec(self, monkeypatch, tmp_path):
        memory = make_memory("User prefers Python")
        response = FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(prompt="What language should I use?")
        output = _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *a, **kw: response,
        )

        data = json.loads(output)
        # Cursor's beforeSubmitPrompt output is a flat object, not a
        # hookSpecificOutput envelope like Codex's.
        assert "continue" in data
        assert "additional_context" in data
        assert data["continue"] is True

    def test_multi_turn_context_from_transcript(self, monkeypatch, tmp_path):
        messages = [
            {"role": "user", "content": "I use Python for all my scripts"},
            {"role": "assistant", "content": "Noted!"},
        ]
        transcript = make_transcript_file(tmp_path, messages)

        captured_body = {}

        def capture_and_respond(req, timeout=None):
            if "/recall" in req.full_url:
                captured_body["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({"results": []})

        hook_input = make_hook_input(prompt="What language should I use?", transcript_path=transcript)
        _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture_and_respond,
            user_config={"recallContextTurns": 2},
        )

        if "body" in captured_body:
            assert "Python" in captured_body["body"].get("query", "")

    def test_recall_timeout_is_configurable(self, monkeypatch, tmp_path):
        memory = make_memory("User prefers Python")
        captured = {}

        def capture_timeout(req, timeout=None):
            captured["timeout"] = timeout
            return FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(prompt="What language should I use?")
        _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture_timeout,
            user_config={"recallTimeout": 42},
        )

        assert captured["timeout"] == 42

    def test_disabled_auto_recall_produces_no_output(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(prompt="What is the capital of France?")
        output = _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            user_config={"autoRecall": False},
        )
        assert output.strip() == ""

    def test_recalls_even_when_continue_is_always_true(self, monkeypatch, tmp_path):
        """Memory errors must never block the user's prompt."""
        response = FakeHTTPResponse({"results": [make_memory("anything")]})
        hook_input = make_hook_input(prompt="anything goes here")
        output = _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *a, **kw: response,
        )
        data = json.loads(output)
        assert data["continue"] is True

    def test_uses_workspace_roots_for_project(self, monkeypatch, tmp_path):
        """When CURSOR_PROJECT_DIR is unset, falls back to workspace_roots[0]."""
        memory = make_memory("hi")
        captured = {}

        def capture(req, timeout=None):
            captured["ua"] = req.get_header("User-agent")
            return FakeHTTPResponse({"results": [memory]})

        # Strip CURSOR_PROJECT_DIR to force fallback path.
        monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
        hook_input = make_hook_input(prompt="anything goes here", workspace_roots=["/work/myapp"])
        _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
        )
        assert captured.get("ua", "").startswith("hindsight-cursor-cli/")


# ---------------------------------------------------------------------------
# retain hook (stop)
# ---------------------------------------------------------------------------


class TestRetainHook:
    def test_posts_transcript_to_hindsight(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)

        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({"status": "accepted"})

        hook_input = make_hook_input(transcript_path=transcript)
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert "body" in captured, "retain API was not called"
        assert "hello" in captured["body"]["items"][0]["content"]

    def test_no_retain_on_empty_transcript(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(transcript_path="/nonexistent/transcript.jsonl")
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url:
                captured["called"] = True
            return FakeHTTPResponse({})

        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)
        assert "called" not in captured

    def test_strips_memory_tags_before_retaining(self, monkeypatch, tmp_path):
        messages = [
            {"role": "user", "content": "<hindsight_memories>old memories</hindsight_memories> actual question"},
            {"role": "assistant", "content": "sure!"},
        ]
        transcript = make_transcript_file(tmp_path, messages)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path=transcript)
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        if "body" in captured:
            content = captured["body"]["items"][0]["content"]
            assert "old memories" not in content
            assert "actual question" in content

    def test_retain_posts_async_true(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path=transcript)
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        if "body" in captured:
            assert captured["body"].get("async") is True

    def test_retain_includes_cursor_cli_context_label(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path=transcript)
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        if "body" in captured:
            assert captured["body"]["items"][0]["context"] == "cursor-cli"

    def test_retain_skips_below_every_n_turns_threshold(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["called"] = True
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path=transcript)
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            user_config={"retainEveryNTurns": 3},
        )
        assert "called" not in captured

    def test_retain_uses_conversation_id_as_document_id(self, monkeypatch, tmp_path):
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, conversation_id="conv-doc-test")
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert "body" in captured
        assert captured["body"]["items"][0]["document_id"] == "conv-doc-test"

    def test_retain_falls_back_to_session_id(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "q"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)
        hook_input.pop("conversation_id", None)
        hook_input["session_id"] = "sess-fallback"
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert captured.get("body", {}).get("items", [{}])[0].get("document_id") == "sess-fallback"

    def test_graceful_on_retain_api_error(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "test"}, {"role": "assistant", "content": "response"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)

        def raise_error(req, timeout=None):
            if "/memories" in req.full_url:
                raise OSError("connection refused")
            return FakeHTTPResponse({})

        # Should not raise
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=raise_error)

    def test_disabled_auto_retain_does_not_call_api(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)
        captured = {}

        def capture(req, timeout=None):
            captured["called"] = True
            return FakeHTTPResponse({})

        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            user_config={"autoRetain": False},
        )
        assert "called" not in captured

    def test_reads_cursor_transcript_format(self, monkeypatch, tmp_path):
        """Retain should correctly parse the on-disk Cursor SDK envelope."""
        messages = [
            {"role": "user", "content": "I like TypeScript"},
            {"role": "assistant", "content": "Great choice!"},
        ]
        transcript = make_transcript_file(tmp_path, messages, cursor_format=True)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path=transcript)
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert "body" in captured, "retain API was not called"
        content = captured["body"]["items"][0]["content"]
        assert "TypeScript" in content

    def test_stop_hook_emits_no_stdout(self, monkeypatch, tmp_path):
        """Cursor's `stop` is fire-and-forget — we don't emit a followup_message."""
        messages = [{"role": "user", "content": "x"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)
        output = _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *a, **kw: FakeHTTPResponse({}),
        )
        assert output.strip() == ""


# ---------------------------------------------------------------------------
# sessionEnd hook
# ---------------------------------------------------------------------------


class TestSessionEndHook:
    def test_forces_final_retain_below_every_n_turns_threshold(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "short session"}, {"role": "assistant", "content": "saved"}]
        transcript = make_transcript_file(tmp_path, messages)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({"status": "accepted"})

        hook_input = make_hook_input(
            transcript_path=transcript,
            conversation_id="conv-session-end",
            reason="completed",
        )
        output = _run_hook(
            "session_end",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            user_config={"retainEveryNTurns": 10},
        )

        assert output.strip() == ""
        assert "body" in captured, "sessionEnd final retain was not called"
        item = captured["body"]["items"][0]
        assert "short session" in item["content"]
        assert item["document_id"] == "conv-session-end"

    def test_no_final_retain_without_transcript_path(self, monkeypatch, tmp_path):
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url:
                captured["called"] = True
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path="", reason="user_close")
        _run_hook(
            "session_end",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            user_config={"retainEveryNTurns": 10},
        )

        assert "called" not in captured


# ---------------------------------------------------------------------------
# sessionStart hook
# ---------------------------------------------------------------------------


class TestSessionStartHook:
    def test_no_output_when_server_reachable(self, monkeypatch, tmp_path):
        """sessionStart is fire-and-forget: no banner, no additional_context.

        Mirrors codex and claude-code: the hook just health-checks and
        pre-warms the daemon. Any user-facing output here is invented
        surface and a divergence risk — the recall output is the only
        agent-visible channel.
        """
        health_response = FakeHTTPResponse({}, status=200)

        def health_then_empty(req, timeout=None):
            if "/health" in req.full_url:
                return health_response
            return FakeHTTPResponse({})

        hook_input = make_hook_input()
        output = _run_hook(
            "session_start",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=health_then_empty,
            set_default_api_url=False,
        )
        assert output.strip() == ""

    def test_no_output_when_server_unreachable(self, monkeypatch, tmp_path):
        def raise_error(req, timeout=None):
            raise OSError("connection refused")

        hook_input = make_hook_input()
        output = _run_hook(
            "session_start",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=raise_error,
            set_default_api_url=False,
        )
        # Fire-and-forget — never raise. Output is empty in both paths.
        assert output.strip() == ""

    def test_both_disabled_produces_no_output(self, monkeypatch, tmp_path):
        hook_input = make_hook_input()
        output = _run_hook(
            "session_start",
            hook_input,
            monkeypatch,
            tmp_path,
            user_config={"autoRecall": False, "autoRetain": False},
        )
        assert output.strip() == ""
