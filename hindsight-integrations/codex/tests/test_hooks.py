"""End-to-end tests for recall.py and retain.py hook scripts.

Mocks the Codex hook runtime:
  - stdin  → io.StringIO(json.dumps(hook_input))
  - stdout → io.StringIO() captured for assertions
  - urllib.request.urlopen → fake HTTP responses
  - HOME → tmp_path (isolates ~/.hindsight/codex.json and state)
"""

import importlib
import io
import json
import os
import sys
from unittest.mock import patch

import pytest

from conftest import FakeHTTPResponse, make_hook_input, make_memory, make_transcript_file, make_user_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_hook(module_name, hook_input, monkeypatch, tmp_path, urlopen_side_effect=None, user_config=None):
    """Import and run a hook script's main() with mocked stdin/stdout/HTTP."""
    # Isolate HOME so ~/.hindsight/codex.json and state land in tmp_path
    monkeypatch.setenv("HOME", str(tmp_path))

    # Strip real HINDSIGHT_* env vars
    for k in list(os.environ):
        if k.startswith("HINDSIGHT_"):
            monkeypatch.delenv(k, raising=False)

    # Set required API URL via env var
    monkeypatch.setenv("HINDSIGHT_API_URL", "http://fake:9077")

    # Write user config (enables retain on every turn + any overrides)
    cfg = {"retainEveryNTurns": 1, "autoRecall": True, "autoRetain": True}
    if user_config:
        cfg.update(user_config)
    make_user_config(tmp_path, cfg)

    stdin_data = io.StringIO(json.dumps(hook_input))
    stdout_capture = io.StringIO()

    # Force reimport so the module picks up patched env
    scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
    spec = importlib.util.spec_from_file_location(
        module_name + "_fresh", os.path.join(scripts_dir, f"{module_name}.py")
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
# recall hook
# ---------------------------------------------------------------------------


class TestRecallHook:
    def test_outputs_additional_context_when_memories_found(self, monkeypatch, tmp_path):
        memory = make_memory("Paris is the capital of France", "world")
        response = FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(prompt="What is the capital of France?")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path,
                           urlopen_side_effect=lambda *a, **kw: response)

        data = json.loads(output)
        context = data["hookSpecificOutput"]["additionalContext"]
        assert "Paris is the capital of France" in context
        assert "<hindsight_memories>" in context

    def test_recall_min_scores_filters_low_scoring_memories(self, monkeypatch, tmp_path):
        low_semantic = make_memory("Marginal match")
        low_semantic["scores"] = {"semantic": 0.42, "reranker": 0.8}
        low_reranker = make_memory("Junk reranker match")
        low_reranker["scores"] = {"semantic": 0.9, "reranker": 0.03}
        no_scores = make_memory("BM25-only match")
        good = make_memory("Relevant match")
        good["scores"] = {"semantic": 0.91, "reranker": 0.45}
        response = FakeHTTPResponse({"results": [low_semantic, low_reranker, no_scores, good]})

        hook_input = make_hook_input(prompt="What deployment rule applies?")
        output = _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *a, **kw: response,
            user_config={"recallMinScores": {"semantic": 0.65, "reranker": 0.2}},
        )

        context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
        assert "Marginal match" not in context
        assert "Junk reranker match" not in context
        assert "BM25-only match" in context
        assert "Relevant match" in context

    def test_recall_min_scores_ignores_invalid_floor(self, monkeypatch, tmp_path):
        memory = make_memory("Relevant match")
        memory["scores"] = {"semantic": 0.91}
        response = FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(prompt="What deployment rule applies?")
        output = _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *a, **kw: response,
            user_config={"recallMinScores": {"semantic": None}},
        )

        context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
        assert "Relevant match" in context

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
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path, urlopen_side_effect=raise_error)
        assert output.strip() == ""

    def test_output_format_matches_codex_spec(self, monkeypatch, tmp_path):
        memory = make_memory("User prefers Python")
        response = FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(prompt="What language should I use?")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path,
                           urlopen_side_effect=lambda *a, **kw: response)

        data = json.loads(output)
        assert data["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert "additionalContext" in data["hookSpecificOutput"]

    def test_multi_turn_context_from_transcript(self, monkeypatch, tmp_path):
        """When recallContextTurns > 1, prior transcript is included in query."""
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
        _run_hook("recall", hook_input, monkeypatch, tmp_path,
                  urlopen_side_effect=capture_and_respond,
                  user_config={"recallContextTurns": 2})

        if "body" in captured_body:
            assert "Python" in captured_body["body"].get("query", "")

    def test_recall_timeout_is_configurable(self, monkeypatch, tmp_path):
        memory = make_memory("User prefers Python")
        captured = {}

        def capture_timeout(req, timeout=None):
            captured["timeout"] = timeout
            return FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(prompt="What language should I use?")
        output = _run_hook(
            "recall",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture_timeout,
            user_config={"recallTimeout": 42},
        )

        data = json.loads(output)
        assert data["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert captured["timeout"] == 42

    def test_disabled_auto_recall_produces_no_output(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(prompt="What is the capital of France?")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path,
                           user_config={"autoRecall": False})
        assert output.strip() == ""


# ---------------------------------------------------------------------------
# retain hook
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

    def test_retain_includes_codex_context_label(self, monkeypatch, tmp_path):
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
            assert captured["body"]["items"][0]["context"] == "codex"

    def test_retain_skips_below_every_n_turns_threshold(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["called"] = True
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path=transcript)
        # retainEveryNTurns=3 — first call should be skipped
        _run_hook("retain", hook_input, monkeypatch, tmp_path,
                  urlopen_side_effect=capture,
                  user_config={"retainEveryNTurns": 3})
        assert "called" not in captured

    def test_retain_uses_session_id_as_document_id(self, monkeypatch, tmp_path):
        messages = [
            {"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"},
        ]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id="sess-doc-test")
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert "body" in captured
        assert captured["body"]["items"][0]["document_id"] == "sess-doc-test"

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

        _run_hook("retain", hook_input, monkeypatch, tmp_path,
                  urlopen_side_effect=capture,
                  user_config={"autoRetain": False})
        assert "called" not in captured

    def test_reads_codex_response_item_format(self, monkeypatch, tmp_path):
        """Retain should correctly parse the actual Codex on-disk transcript format."""
        messages = [
            {"role": "user", "content": "I like TypeScript"},
            {"role": "assistant", "content": "Great choice!"},
        ]
        transcript = make_transcript_file(tmp_path, messages, codex_format=True)
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
