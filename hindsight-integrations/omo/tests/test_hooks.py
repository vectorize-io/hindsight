"""End-to-end tests for recall.py and retain.py hook scripts.

Mocks the OMO hook runtime:
  - stdin  → io.StringIO(json.dumps(hook_input))
  - stdout → io.StringIO() captured for assertions
  - urllib.request.urlopen → fake HTTP responses
  - PLUGIN_ROOT / PLUGIN_DATA → temp dirs
"""

import importlib
import io
import json
import os
from unittest.mock import patch

import pytest

import sys
import os

# Ensure conftest helpers are importable when run from parent dir
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from conftest import FakeHTTPResponse, make_hook_input, make_memory, make_transcript_file


def _run_hook(module_name, hook_input, monkeypatch, tmp_path, urlopen_side_effect=None, extra_env=None, extra_settings=None):
    """Import and run a hook script's main() with mocked stdin/stdout/HTTP."""
    monkeypatch.setenv("PLUGIN_ROOT", str(tmp_path / "plugin_root"))
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path / "plugin_data"))
    (tmp_path / "plugin_root").mkdir(exist_ok=True)
    (tmp_path / "plugin_data").mkdir(exist_ok=True)

    for k in list(os.environ):
        if k.startswith("HINDSIGHT_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    for k, v in (extra_env or {}).items():
        monkeypatch.setenv(k, v)

    settings = {
        "autoRecall": True,
        "autoRetain": True,
        "retainEveryNTurns": 1,
        "hindsightApiUrl": "http://fake:9077",
        "hindsightApiToken": "hsk_test",
    }
    if extra_settings:
        settings.update(extra_settings)
    (tmp_path / "plugin_root" / "settings.json").write_text(json.dumps(settings))

    stdin_data = io.StringIO(json.dumps(hook_input))
    stdout_capture = io.StringIO()

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

    def test_output_format_matches_hook_spec(self, monkeypatch, tmp_path):
        memory = make_memory("User prefers Python")
        response = FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(prompt="What language should I use?")
        output = _run_hook("recall", hook_input, monkeypatch, tmp_path, urlopen_side_effect=lambda *a, **kw: response)

        data = json.loads(output)
        assert data["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert "additionalContext" in data["hookSpecificOutput"]

    def test_sends_bearer_token(self, monkeypatch, tmp_path):
        """Cloud mode should send Authorization: Bearer header."""
        captured_headers = {}

        def capture_request(req, timeout=None):
            captured_headers["auth"] = req.get_header("Authorization")
            return FakeHTTPResponse({"results": []})

        hook_input = make_hook_input(prompt="What is my project about?")
        _run_hook(
            "recall", hook_input, monkeypatch, tmp_path,
            urlopen_side_effect=capture_request,
            extra_settings={"hindsightApiToken": "hsk_mykey"},
        )

        assert captured_headers.get("auth") == "Bearer hsk_mykey"

    def test_skips_cloud_without_token(self, monkeypatch, tmp_path):
        """Cloud URL without API token should silently skip recall."""
        called = {}

        def should_not_be_called(*a, **kw):
            called["yes"] = True
            return FakeHTTPResponse({"results": []})

        hook_input = make_hook_input(prompt="What is my project about?")
        _run_hook(
            "recall", hook_input, monkeypatch, tmp_path,
            urlopen_side_effect=should_not_be_called,
            extra_settings={
                "hindsightApiUrl": "https://api.hindsight.vectorize.io",
                "hindsightApiToken": None,
            },
        )

        assert "yes" not in called


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

    def test_retain_context_label_is_omo(self, monkeypatch, tmp_path):
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
            assert captured["body"]["items"][0]["context"] == "omo"

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
        hook_input = make_hook_input(transcript_path=transcript)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        if "body" in captured:
            assert captured["body"].get("async") is True

    def test_graceful_on_retain_api_error(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "test message"}, {"role": "assistant", "content": "response"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)

        def raise_error(req, timeout=None):
            if "/memories" in req.full_url:
                raise OSError("connection refused")
            return FakeHTTPResponse({})

        _run_hook("retain", hook_input, monkeypatch, tmp_path, urlopen_side_effect=raise_error)

    def test_full_session_uses_session_id_as_document_id(self, monkeypatch, tmp_path):
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

        assert "body" in captured
        item = captured["body"]["items"][0]
        assert item["document_id"] == "sess-full-123"
        assert "first question" in item["content"]
        assert "second question" in item["content"]
