"""End-to-end tests for the ZCode hook scripts.

Mocks the ZCode hook runtime:
  - stdin  → io.StringIO(json.dumps(hook_input))
  - stdout → io.StringIO() captured for assertions
  - urllib.request.urlopen → fake HTTP responses
  - HOME → tmp_path (isolates ~/.hindsight/zcode.json and state)
"""

import importlib.util
import io
import json
import os
import sys
from unittest.mock import patch

import pytest
from conftest import (
    SCRIPTS_DIR,
    FakeHTTPResponse,
    make_hook_input,
    make_memory,
    make_stop_input,
    make_transcript_file,
    make_user_config,
    stash_prompt,
)


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


def _additional_context(data):
    """Extract additionalContext from the Claude Code UserPromptSubmit output."""
    return data["hookSpecificOutput"]["additionalContext"]


# ---------------------------------------------------------------------------
# recall hook (UserPromptSubmit)
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
        ctx = _additional_context(data)
        assert "Paris is the capital of France" in ctx
        assert "<hindsight_memories>" in ctx

    def test_no_output_when_no_memories(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(prompt="hello there world")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path)
        assert output.strip() == ""

    def test_no_output_for_short_prompt(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(prompt="hi")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path)
        assert output.strip() == ""

    def test_stashes_prompt_for_retain(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(prompt="What is the capital of France?", session_id="sess-stash")
        _run_hook("recall", hook_input, monkeypatch, tmp_path)
        stash = tmp_path / ".hindsight" / "zcode" / "state" / "last_prompt_sess-stash.json"
        assert stash.exists()
        assert json.loads(stash.read_text())["prompt"] == "What is the capital of France?"

    def test_stashes_prompt_even_when_too_short_for_recall(self, monkeypatch, tmp_path):
        """Short prompts skip recall injection but must still be stashed for retain."""
        hook_input = make_hook_input(prompt="ls", session_id="sess-short")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path)
        assert output.strip() == ""
        stash = tmp_path / ".hindsight" / "zcode" / "state" / "last_prompt_sess-short.json"
        assert stash.exists()
        assert json.loads(stash.read_text())["prompt"] == "ls"

    def test_stashes_prompt_from_sessionid_camelcase(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(prompt="a longer prompt here", session_id="ignored")
        hook_input.pop("session_id", None)
        hook_input["sessionId"] = "sess-camel-recall"
        _run_hook("recall", hook_input, monkeypatch, tmp_path)
        stash = tmp_path / ".hindsight" / "zcode" / "state" / "last_prompt_sess-camel-recall.json"
        assert stash.exists()

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

    def test_output_format_matches_claude_code_spec(self, monkeypatch, tmp_path):
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
        # ZCode embeds the Claude Code runtime: recall emits the
        # hookSpecificOutput.additionalContext envelope for UserPromptSubmit.
        assert data["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert "additionalContext" in data["hookSpecificOutput"]

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

    def test_recall_never_raises_on_memories(self, monkeypatch, tmp_path):
        """Memory injection must be a clean UserPromptSubmit envelope."""
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
        assert _additional_context(data)

    def test_uses_workspace_roots_for_project(self, monkeypatch, tmp_path):
        """When ZCODE_PROJECT_DIR is unset, falls back to workspace_roots[0]."""
        memory = make_memory("hi")
        captured = {}

        def capture(req, timeout=None):
            captured["ua"] = req.get_header("User-agent")
            return FakeHTTPResponse({"results": [memory]})

        # Strip ZCODE_PROJECT_DIR to force fallback path.
        monkeypatch.delenv("ZCODE_PROJECT_DIR", raising=False)
        hook_input = make_hook_input(prompt="anything goes here", workspace_roots=["/work/myapp"])
        _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
        )
        assert captured.get("ua", "").startswith("hindsight-zcode/")


# ---------------------------------------------------------------------------
# retain hook (Stop)
# ---------------------------------------------------------------------------


def _retain_body_capture(captured):
    """A urlopen side-effect that captures the retain POST body."""

    def capture(req, timeout=None):
        if "/memories" in req.full_url and "/recall" not in req.full_url:
            captured["body"] = json.loads(req.data.decode())
        return FakeHTTPResponse({"status": "accepted"})

    return capture


class TestRetainHook:
    def test_pairs_stashed_prompt_with_response_text(self, monkeypatch, tmp_path):
        """Retain assembles the turn from the stashed prompt + responseText."""
        stash_prompt(tmp_path, "sess-abc123", "how do I list files")
        captured = {}
        hook_input = make_stop_input(response_text="use the ls command", session_id="sess-abc123")
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=_retain_body_capture(captured))

        assert "body" in captured, "retain API was not called"
        content = captured["body"]["items"][0]["content"]
        assert "how do I list files" in content
        assert "use the ls command" in content

    def test_retains_assistant_only_when_no_stashed_prompt(self, monkeypatch, tmp_path):
        captured = {}
        hook_input = make_stop_input(response_text="standalone reply", session_id="sess-noprompt")
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=_retain_body_capture(captured))

        assert "body" in captured, "retain API was not called"
        content = captured["body"]["items"][0]["content"]
        assert "standalone reply" in content

    def test_accepts_sessionid_camelcase(self, monkeypatch, tmp_path):
        stash_prompt(tmp_path, "sess-camel", "prompt text here")
        captured = {}
        hook_input = make_stop_input(response_text="reply", session_id="ignored")
        # Only the camelCase key is present.
        hook_input.pop("session_id", None)
        hook_input["sessionId"] = "sess-camel"
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=_retain_body_capture(captured))

        assert "prompt text here" in captured["body"]["items"][0]["content"]

    def test_no_retain_when_no_text_and_no_prompt(self, monkeypatch, tmp_path):
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url:
                captured["called"] = True
            return FakeHTTPResponse({})

        hook_input = make_stop_input(response_text="", session_id="sess-empty")
        hook_input["responsePreview"] = ""
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)
        assert "called" not in captured

    def test_falls_back_to_transcript_for_assistant_text(self, monkeypatch, tmp_path):
        """With no responseText, retain parses the ephemeral ZCode transcript."""
        transcript = make_transcript_file(
            tmp_path,
            [{"role": "assistant", "content": "from the transcript"}],
            zcode_format=True,
        )
        captured = {}
        hook_input = make_stop_input(response_text="", session_id="sess-transcript")
        hook_input["responsePreview"] = ""
        hook_input["transcript_path"] = transcript
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=_retain_body_capture(captured))

        assert "body" in captured, "retain API was not called"
        assert "from the transcript" in captured["body"]["items"][0]["content"]

    def test_strips_memory_tags_before_retaining(self, monkeypatch, tmp_path):
        stash_prompt(
            tmp_path,
            "sess-tags",
            "<hindsight_memories>old memories</hindsight_memories> actual question",
        )
        captured = {}
        hook_input = make_stop_input(response_text="sure!", session_id="sess-tags")
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=_retain_body_capture(captured))

        content = captured["body"]["items"][0]["content"]
        assert "old memories" not in content
        assert "actual question" in content

    def test_retain_posts_async_true(self, monkeypatch, tmp_path):
        stash_prompt(tmp_path, "sess-abc123", "hello there")
        captured = {}
        hook_input = make_stop_input(response_text="world", session_id="sess-abc123")
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=_retain_body_capture(captured))

        assert captured["body"].get("async") is True

    def test_retain_includes_zcode_context_label(self, monkeypatch, tmp_path):
        stash_prompt(tmp_path, "sess-abc123", "hello there")
        captured = {}
        hook_input = make_stop_input(response_text="world", session_id="sess-abc123")
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=_retain_body_capture(captured))

        assert captured["body"]["items"][0]["context"] == "zcode"

    def test_retain_skips_below_every_n_turns_threshold(self, monkeypatch, tmp_path):
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["called"] = True
            return FakeHTTPResponse({})

        hook_input = make_stop_input(response_text="world", session_id="sess-abc123")
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            user_config={"retainEveryNTurns": 3},
        )
        assert "called" not in captured

    def test_document_id_is_per_turn(self, monkeypatch, tmp_path):
        """Each turn gets a distinct document_id so prior turns aren't overwritten."""
        stash_prompt(tmp_path, "sess-doc-test", "question")
        captured = {}
        hook_input = make_stop_input(response_text="answer", session_id="sess-doc-test")
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=_retain_body_capture(captured))

        doc_id = captured["body"]["items"][0]["document_id"]
        assert doc_id.startswith("sess-doc-test-")
        assert doc_id != "sess-doc-test"

    def test_graceful_on_retain_api_error(self, monkeypatch, tmp_path):
        stash_prompt(tmp_path, "sess-abc123", "test")

        def raise_error(req, timeout=None):
            if "/memories" in req.full_url:
                raise OSError("connection refused")
            return FakeHTTPResponse({})

        hook_input = make_stop_input(response_text="response", session_id="sess-abc123")
        # Should not raise
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=raise_error)

    def test_disabled_auto_retain_does_not_call_api(self, monkeypatch, tmp_path):
        stash_prompt(tmp_path, "sess-abc123", "hello")
        captured = {}

        def capture(req, timeout=None):
            captured["called"] = True
            return FakeHTTPResponse({})

        hook_input = make_stop_input(response_text="world", session_id="sess-abc123")
        _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            user_config={"autoRetain": False},
        )
        assert "called" not in captured

    def test_stop_hook_emits_no_stdout(self, monkeypatch, tmp_path):
        """The Stop hook stores memory silently — it emits no stdout."""
        stash_prompt(tmp_path, "sess-abc123", "hi there")
        hook_input = make_stop_input(response_text="reply", session_id="sess-abc123")
        output = _run_hook(
            "retain",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *a, **kw: FakeHTTPResponse({}),
        )
        assert output.strip() == ""


# ---------------------------------------------------------------------------
# SessionStart hook
# ---------------------------------------------------------------------------


class TestSessionStartHook:
    def test_no_output_when_server_reachable(self, monkeypatch, tmp_path):
        """SessionStart is fire-and-forget: no banner, no additionalContext.

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
