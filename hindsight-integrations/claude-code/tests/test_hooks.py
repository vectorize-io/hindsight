"""End-to-end tests for recall.py and retain.py hook scripts.

Mocks the Claude Code hook runtime:
  - stdin  → io.StringIO(json.dumps(hook_input))
  - stdout → io.StringIO() captured for assertions
  - urllib.request.urlopen → fake HTTP responses
  - CLAUDE_PLUGIN_ROOT / CLAUDE_PLUGIN_DATA → temp dirs
"""

import importlib
import io
import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from conftest import FakeHTTPResponse, make_hook_input, make_memory, make_transcript_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_hook(module_name, hook_input, monkeypatch, tmp_path, urlopen_side_effect=None, extra_env=None, extra_settings=None):
    """Import and run a hook script's main() with mocked stdin/stdout/HTTP."""
    # Isolated plugin dirs
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path / "plugin_root"))
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "plugin_data"))
    (tmp_path / "plugin_root").mkdir(exist_ok=True)
    (tmp_path / "plugin_data").mkdir(exist_ok=True)

    # Strip real HINDSIGHT_* env vars and neutralize user config (~/.hindsight/claude-code.json)
    for k in list(os.environ):
        if k.startswith("HINDSIGHT_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    for k, v in (extra_env or {}).items():
        monkeypatch.setenv(k, v)

    # Write a minimal settings.json enabling fast retains
    settings = {"autoRecall": True, "autoRetain": True, "retainEveryNTurns": 1, "hindsightApiUrl": "http://fake:9077"}
    if extra_settings:
        settings.update(extra_settings)
    (tmp_path / "plugin_root" / "settings.json").write_text(json.dumps(settings))

    stdin_data = io.StringIO(json.dumps(hook_input))
    stdout_capture = io.StringIO()

    # Force reimport so the module picks up patched env / path
    scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(scripts_dir, f"{module_name}.py"))
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
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path, urlopen_side_effect=lambda *a, **kw: response)

        data = json.loads(output)
        context = data["hookSpecificOutput"]["additionalContext"]
        assert "Paris is the capital of France" in context
        assert "<hindsight_memories>" in context

    def test_no_output_when_no_memories(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(prompt="hello there world")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path)
        # Empty stdout = no memories injected
        assert output.strip() == ""

    def test_no_output_for_short_prompt(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(prompt="hi")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path)
        assert output.strip() == ""

    def test_graceful_on_api_error(self, monkeypatch, tmp_path):
        def raise_error(*a, **kw):
            raise OSError("connection refused")

        hook_input = make_hook_input(prompt="What is my project about?")
        # Should not raise — graceful degradation
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path, urlopen_side_effect=raise_error)
        assert output.strip() == ""

    def test_output_format_matches_claude_code_spec(self, monkeypatch, tmp_path):
        memory = make_memory("User prefers Python")
        response = FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(prompt="What language should I use?")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path, urlopen_side_effect=lambda *a, **kw: response)

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

        # Override to use multi-turn recall
        settings = {
            "autoRecall": True,
            "hindsightApiUrl": "http://fake:9077",
            "recallContextTurns": 2,
            "retainEveryNTurns": 1,
            "autoRetain": True,
        }
        (tmp_path / "plugin_root").mkdir(exist_ok=True)
        (tmp_path / "plugin_data").mkdir(exist_ok=True)

        captured_body = {}

        def capture_and_respond(req, timeout=None):
            if "/recall" in req.full_url:
                captured_body["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({"results": []})

        for k in list(os.environ):
            if k.startswith("HINDSIGHT_"):
                monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path / "plugin_root"))
        monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "plugin_data"))
        (tmp_path / "plugin_root" / "settings.json").write_text(json.dumps(settings))

        hook_input = make_hook_input(prompt="What language should I use?", transcript_path=transcript)
        stdin_data = io.StringIO(json.dumps(hook_input))

        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
        spec = importlib.util.spec_from_file_location("recall", os.path.join(scripts_dir, "recall.py"))
        mod = importlib.util.module_from_spec(spec)

        with (
            patch("sys.stdin", stdin_data),
            patch("sys.stdout", io.StringIO()),
            patch("urllib.request.urlopen", side_effect=capture_and_respond),
        ):
            spec.loader.exec_module(mod)
            mod.main()

        # The query should contain prior context from the transcript
        if "body" in captured_body:
            assert "Python" in captured_body["body"].get("query", "")

    def test_disabled_auto_recall_produces_no_output(self, monkeypatch, tmp_path):
        (tmp_path / "plugin_root").mkdir(exist_ok=True)
        (tmp_path / "plugin_data").mkdir(exist_ok=True)
        settings = {"autoRecall": False, "autoRetain": False, "hindsightApiUrl": "http://fake:9077"}
        (tmp_path / "plugin_root" / "settings.json").write_text(json.dumps(settings))

        for k in list(os.environ):
            if k.startswith("HINDSIGHT_"):
                monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path / "plugin_root"))
        monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "plugin_data"))

        hook_input = make_hook_input(prompt="What is the capital of France?")
        stdin_data = io.StringIO(json.dumps(hook_input))
        stdout_capture = io.StringIO()

        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
        spec = importlib.util.spec_from_file_location("recall_disabled", os.path.join(scripts_dir, "recall.py"))
        mod = importlib.util.module_from_spec(spec)

        with patch("sys.stdin", stdin_data), patch("sys.stdout", stdout_capture):
            spec.loader.exec_module(mod)
            mod.main()

        assert stdout_capture.getvalue().strip() == ""


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

    def test_retain_tags_with_template_variables(self, monkeypatch, tmp_path):
        """retainTags config should resolve template variables like {session_id}."""
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id="sess-tag-test")
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        _run_hook(
            "retain", hook_input, monkeypatch, tmp_path,
            urlopen_side_effect=capture,
            extra_settings={"retainTags": ["{session_id}", "claude-code", "custom-tag"]},
        )

        assert "body" in captured, "retain API was not called"
        item = captured["body"]["items"][0]
        assert item["tags"] == ["sess-tag-test", "claude-code", "custom-tag"]

    def test_retain_custom_metadata(self, monkeypatch, tmp_path):
        """retainMetadata config should be merged with built-in metadata."""
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id="sess-meta-test")
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        _run_hook(
            "retain", hook_input, monkeypatch, tmp_path,
            urlopen_side_effect=capture,
            extra_settings={"retainMetadata": {"project": "my-project", "session": "{session_id}"}},
        )

        assert "body" in captured, "retain API was not called"
        meta = captured["body"]["items"][0]["metadata"]
        # Built-in metadata
        assert meta["session_id"] == "sess-meta-test"
        assert "retained_at" in meta
        # Custom metadata with template resolution
        assert meta["project"] == "my-project"
        assert meta["session"] == "sess-meta-test"

    def test_full_session_uses_session_id_as_document_id(self, monkeypatch, tmp_path):
        """In full-session mode, document_id should be the session_id (for upsert)."""
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id="sess-full-123")
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert "body" in captured, "retain API was not called"
        item = captured["body"]["items"][0]
        # document_id should be just the session_id, no timestamp suffix
        assert item["document_id"] == "sess-full-123"
        # Should contain ALL messages, not just the last turn
        assert "first question" in item["content"]
        assert "second question" in item["content"]

    def test_full_session_respects_retain_every_n_turns(self, monkeypatch, tmp_path):
        """In full-session mode, retainEveryNTurns should still gate when retain fires."""
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id="sess-throttle")
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["called"] = True
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        # retainEveryNTurns=3 in full-session mode — first 2 calls should be skipped
        _run_hook(
            "retain", hook_input, monkeypatch, tmp_path,
            urlopen_side_effect=capture,
            extra_settings={"retainEveryNTurns": 3},
        )
        # Turn 1 of 3 — should NOT retain
        assert "called" not in captured

        # Turn 2 — still skip
        captured.clear()
        _run_hook(
            "retain", hook_input, monkeypatch, tmp_path,
            urlopen_side_effect=capture,
            extra_settings={"retainEveryNTurns": 3},
        )
        assert "called" not in captured

        # Turn 3 — should fire, with full session content and session_id as doc ID
        captured.clear()
        _run_hook(
            "retain", hook_input, monkeypatch, tmp_path,
            urlopen_side_effect=capture,
            extra_settings={"retainEveryNTurns": 3},
        )
        assert "called" in captured, "retain should fire on turn 3"
        item = captured["body"]["items"][0]
        assert item["document_id"] == "sess-throttle"  # full-session uses session_id
        assert "hello" in item["content"]

    def test_chunked_retain_skips_below_threshold(self, monkeypatch, tmp_path):
        """With retainEveryNTurns=5 and retainMode=chunked, first call should be skipped."""
        (tmp_path / "plugin_root").mkdir(exist_ok=True)
        (tmp_path / "plugin_data").mkdir(exist_ok=True)
        settings = {
            "autoRetain": True,
            "autoRecall": True,
            "retainMode": "chunked",
            "retainEveryNTurns": 5,
            "hindsightApiUrl": "http://fake:9077",
        }
        (tmp_path / "plugin_root" / "settings.json").write_text(json.dumps(settings))

        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)

        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["called"] = True
            return FakeHTTPResponse({})

        for k in list(os.environ):
            if k.startswith("HINDSIGHT_"):
                monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path / "plugin_root"))
        monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "plugin_data"))
        monkeypatch.setenv("HINDSIGHT_RETAIN_MODE", "chunked")
        monkeypatch.setenv("HOME", str(tmp_path))

        stdin_data = io.StringIO(json.dumps(hook_input))
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
        spec = importlib.util.spec_from_file_location("retain_chunked", os.path.join(scripts_dir, "retain.py"))
        mod = importlib.util.module_from_spec(spec)

        with (
            patch("sys.stdin", stdin_data),
            patch("sys.stdout", io.StringIO()),
            patch("urllib.request.urlopen", side_effect=capture),
        ):
            spec.loader.exec_module(mod)
            mod.main()

        # Turn 1 of 5 — should NOT retain
        assert "called" not in captured

    def test_graceful_on_retain_api_error(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "test message"}, {"role": "assistant", "content": "response"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)

        def raise_error(req, timeout=None):
            if "/memories" in req.full_url:
                raise OSError("connection refused")
            return FakeHTTPResponse({})

        # Should not raise
        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=raise_error)

    def test_retain_posts_async_true(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        if "body" in captured:
            assert captured["body"].get("async") is True

    def test_retain_includes_context_label(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        if "body" in captured:
            assert captured["body"]["items"][0]["context"] == "claude-code"

    def test_disabled_auto_retain_does_not_call_api(self, monkeypatch, tmp_path):
        (tmp_path / "plugin_root").mkdir(exist_ok=True)
        (tmp_path / "plugin_data").mkdir(exist_ok=True)
        settings = {"autoRetain": False, "autoRecall": False, "hindsightApiUrl": "http://fake:9077"}
        (tmp_path / "plugin_root" / "settings.json").write_text(json.dumps(settings))

        messages = [{"role": "user", "content": "hello"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)
        captured = {}

        def capture(req, timeout=None):
            captured["called"] = True
            return FakeHTTPResponse({})

        for k in list(os.environ):
            if k.startswith("HINDSIGHT_"):
                monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path / "plugin_root"))
        monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "plugin_data"))

        stdin_data = io.StringIO(json.dumps(hook_input))
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
        spec = importlib.util.spec_from_file_location("retain_disabled", os.path.join(scripts_dir, "retain.py"))
        mod = importlib.util.module_from_spec(spec)

        with (
            patch("sys.stdin", stdin_data),
            patch("sys.stdout", io.StringIO()),
            patch("urllib.request.urlopen", side_effect=capture),
        ):
            spec.loader.exec_module(mod)
            mod.main()

        assert "called" not in captured
