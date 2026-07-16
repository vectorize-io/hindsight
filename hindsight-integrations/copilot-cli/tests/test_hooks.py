"""End-to-end tests for the GitHub Copilot CLI hook scripts.

Mocks the Copilot CLI hook runtime:
  - stdin  → io.StringIO(json.dumps(hook_input))
  - stdout → io.StringIO() captured for assertions
  - urllib.request.urlopen → fake HTTP responses
  - HOME → tmp_path (isolates ~/.hindsight/copilot-cli.json and state)
"""

import importlib.util
import io
import json
import os
import sys
from unittest.mock import patch

from conftest import (
    SCRIPTS_DIR,
    FakeHTTPResponse,
    make_hook_input,
    make_memory,
    make_subagent_hook_input,
    make_transcript_file,
    make_user_config,
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


# ---------------------------------------------------------------------------
# sessionStart hook (recall)
# ---------------------------------------------------------------------------


class TestSessionStartHook:
    def test_outputs_additional_context_when_memories_found(self, monkeypatch, tmp_path):
        memory = make_memory("Paris is the capital of France", "world")
        response = FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(initial_prompt="What is the capital of France?")
        output = _run_hook(
            "session_start",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *a, **kw: response,
        )

        data = json.loads(output)
        assert "additionalContext" in data
        assert "Paris is the capital of France" in data["additionalContext"]
        assert "<hindsight_memories>" in data["additionalContext"]

    def test_no_output_when_no_memories(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(initial_prompt="hello there world")
        output = _run_hook("session_start", hook_input, monkeypatch, tmp_path)
        assert output.strip() == ""

    def test_falls_back_to_project_query_without_initial_prompt(self, monkeypatch, tmp_path):
        """Interactive sessions rarely have an initialPrompt — falls back to cwd."""
        memory = make_memory("User prefers Python")
        captured = {}

        def capture(req, timeout=None):
            if "/recall" in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({"results": [memory]})

        hook_input = make_hook_input(cwd="/home/user/myproject", initial_prompt=None)
        _run_hook("session_start", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert "body" in captured
        assert "myproject" in captured["body"]["query"]

    def test_graceful_on_api_error(self, monkeypatch, tmp_path):
        def raise_error(*a, **kw):
            raise OSError("connection refused")

        hook_input = make_hook_input(initial_prompt="What is my project about?")
        output = _run_hook(
            "session_start",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=raise_error,
            set_default_api_url=False,
        )
        assert output.strip() == ""

    def test_disabled_auto_recall_produces_no_output(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(initial_prompt="What is the capital of France?")
        output = _run_hook(
            "session_start",
            hook_input,
            monkeypatch,
            tmp_path,
            user_config={"autoRecall": False},
        )
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

    def test_no_output_when_server_unreachable_and_retain_enabled(self, monkeypatch, tmp_path):
        """Server unreachable → background pre-start kicked off, no crash, no output."""

        def raise_error(*a, **kw):
            raise OSError("connection refused")

        hook_input = make_hook_input(initial_prompt="anything")
        output = _run_hook(
            "session_start",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=raise_error,
            set_default_api_url=False,
            user_config={"embedPackagePath": "/nonexistent/embed"},
        )
        assert output.strip() == ""

    def test_user_agent_uses_copilot_cli_branding(self, monkeypatch, tmp_path):
        captured = {}

        def capture(req, timeout=None):
            captured["ua"] = req.get_header("User-agent")
            return FakeHTTPResponse({"results": [make_memory("hi")]})

        hook_input = make_hook_input(initial_prompt="anything goes here")
        _run_hook("session_start", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)
        assert captured.get("ua", "").startswith("hindsight-copilot-cli/")


# ---------------------------------------------------------------------------
# subagentStart hook (recall)
# ---------------------------------------------------------------------------


class TestSubagentStartHook:
    def test_outputs_additional_context_when_memories_found(self, monkeypatch, tmp_path):
        memory = make_memory("Project uses pytest for testing")
        response = FakeHTTPResponse({"results": [memory]})

        hook_input = make_subagent_hook_input(agent_name="explore")
        output = _run_hook(
            "subagent_start",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *a, **kw: response,
        )

        data = json.loads(output)
        assert "additionalContext" in data
        assert "Project uses pytest for testing" in data["additionalContext"]

    def test_always_uses_fallback_query_never_specific_task(self, monkeypatch, tmp_path):
        """subagentStart payload never carries per-invocation task text."""
        captured = {}

        def capture(req, timeout=None):
            if "/recall" in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({"results": []})

        hook_input = make_subagent_hook_input(cwd="/home/user/coolproject", agent_name="research")
        _run_hook("subagent_start", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert "body" in captured
        assert "coolproject" in captured["body"]["query"]

    def test_no_output_when_no_memories(self, monkeypatch, tmp_path):
        hook_input = make_subagent_hook_input(agent_name="code-review")
        output = _run_hook("subagent_start", hook_input, monkeypatch, tmp_path)
        assert output.strip() == ""

    def test_never_blocks_on_api_error(self, monkeypatch, tmp_path):
        def raise_error(*a, **kw):
            raise OSError("connection refused")

        hook_input = make_subagent_hook_input(agent_name="rubber-duck")
        output = _run_hook(
            "subagent_start",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=raise_error,
            set_default_api_url=False,
        )
        assert output.strip() == ""

    def test_disabled_auto_recall_produces_no_output(self, monkeypatch, tmp_path):
        hook_input = make_subagent_hook_input(agent_name="security-review")
        output = _run_hook(
            "subagent_start",
            hook_input,
            monkeypatch,
            tmp_path,
            user_config={"autoRecall": False},
        )
        assert output.strip() == ""

    def test_works_for_multiple_agent_names(self, monkeypatch, tmp_path):
        memory = make_memory("hi")
        for agent_name in [
            "explore",
            "task",
            "research",
            "code-review",
            "rubber-duck",
            "security-review",
            "custom-agent",
        ]:
            hook_input = make_subagent_hook_input(agent_name=agent_name)
            output = _run_hook(
                "subagent_start",
                hook_input,
                monkeypatch,
                tmp_path,
                urlopen_side_effect=lambda *a, **kw: FakeHTTPResponse({"results": [memory]}),
            )
            data = json.loads(output)
            assert "additionalContext" in data


# ---------------------------------------------------------------------------
# agentStop hook (retain)
# ---------------------------------------------------------------------------


class TestAgentStopHook:
    def test_posts_transcript_to_hindsight(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)

        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({"status": "accepted"})

        hook_input = make_hook_input(transcript_path=transcript, stop_reason="end_turn")
        _run_hook("agent_stop", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert "body" in captured, "retain API was not called"
        assert "hello" in captured["body"]["items"][0]["content"]

    def test_no_retain_on_empty_transcript(self, monkeypatch, tmp_path):
        hook_input = make_hook_input(transcript_path="/nonexistent/transcript.jsonl")
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url:
                captured["called"] = True
            return FakeHTTPResponse({})

        _run_hook("agent_stop", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)
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
        _run_hook("agent_stop", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert "body" in captured
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
        _run_hook("agent_stop", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert captured["body"].get("async") is True

    def test_retain_includes_copilot_cli_context_label(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        transcript = make_transcript_file(tmp_path, messages)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path=transcript)
        _run_hook("agent_stop", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert captured["body"]["items"][0]["context"] == "copilot-cli"

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
            "agent_stop",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            user_config={"retainEveryNTurns": 3},
        )
        assert "called" not in captured

    def test_retain_uses_session_id_as_document_id(self, monkeypatch, tmp_path):
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id="sess-doc-test")
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        _run_hook("agent_stop", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

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
        _run_hook("agent_stop", hook_input, monkeypatch, tmp_path, urlopen_side_effect=raise_error)

    def test_disabled_auto_retain_does_not_call_api(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)
        captured = {}

        def capture(req, timeout=None):
            captured["called"] = True
            return FakeHTTPResponse({})

        _run_hook(
            "agent_stop",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            user_config={"autoRetain": False},
        )
        assert "called" not in captured

    def test_reads_sdk_transcript_format(self, monkeypatch, tmp_path):
        """Retain should correctly parse the type-nested SDK envelope."""
        messages = [
            {"role": "user", "content": "I like TypeScript"},
            {"role": "assistant", "content": "Great choice!"},
        ]
        transcript = make_transcript_file(tmp_path, messages, sdk_format=True)
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        hook_input = make_hook_input(transcript_path=transcript)
        _run_hook("agent_stop", hook_input, monkeypatch, tmp_path, urlopen_side_effect=capture)

        assert "body" in captured, "retain API was not called"
        content = captured["body"]["items"][0]["content"]
        assert "TypeScript" in content

    def test_emits_no_stdout(self, monkeypatch, tmp_path):
        """agentStop output is never used to force another turn by this integration."""
        messages = [{"role": "user", "content": "x"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript)
        output = _run_hook(
            "agent_stop",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *a, **kw: FakeHTTPResponse({}),
        )
        assert output.strip() == ""

    def test_caches_transcript_path_for_session_end(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hello"}]
        transcript = make_transcript_file(tmp_path, messages)
        hook_input = make_hook_input(transcript_path=transcript, session_id="sess-cache-test")

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_API_URL", "http://fake:9077")
        make_user_config(tmp_path, {"retainEveryNTurns": 1})

        stdin_data = io.StringIO(json.dumps(hook_input))
        with (
            patch("sys.stdin", stdin_data),
            patch("sys.stdout", io.StringIO()),
            patch("urllib.request.urlopen", side_effect=lambda *a, **kw: FakeHTTPResponse({})),
        ):
            spec = importlib.util.spec_from_file_location(
                "agent_stop_cache_test", os.path.join(SCRIPTS_DIR, "agent_stop.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()

        sys.path.insert(0, SCRIPTS_DIR)
        from lib.state import get_cached_session_transcript

        cached = get_cached_session_transcript("sess-cache-test")
        assert cached is not None
        assert cached["transcript_path"] == transcript


# ---------------------------------------------------------------------------
# sessionEnd hook
# ---------------------------------------------------------------------------


class TestSessionEndHook:
    def test_forces_final_retain_using_cached_transcript_path(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "short session"}, {"role": "assistant", "content": "saved"}]
        transcript = make_transcript_file(tmp_path, messages)

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_API_URL", "http://fake:9077")
        make_user_config(tmp_path, {"retainEveryNTurns": 10})

        sys.path.insert(0, SCRIPTS_DIR)
        from lib.state import cache_session_transcript

        cache_session_transcript("conv-session-end", transcript, "/home/user/myproject")

        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url and "/recall" not in req.full_url:
                captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({"status": "accepted"})

        hook_input = make_hook_input(session_id="conv-session-end", reason="user_exit")
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

    def test_no_final_retain_without_cached_transcript(self, monkeypatch, tmp_path):
        captured = {}

        def capture(req, timeout=None):
            if "/memories" in req.full_url:
                captured["called"] = True
            return FakeHTTPResponse({})

        hook_input = make_hook_input(session_id="conv-no-cache", reason="user_exit")
        _run_hook(
            "session_end",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            user_config={"retainEveryNTurns": 10},
        )

        assert "called" not in captured

    def test_clears_cache_after_final_retain(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hi"}]
        transcript = make_transcript_file(tmp_path, messages)

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_API_URL", "http://fake:9077")
        make_user_config(tmp_path, {"retainEveryNTurns": 10})

        sys.path.insert(0, SCRIPTS_DIR)
        from lib.state import cache_session_transcript, get_cached_session_transcript

        cache_session_transcript("conv-clear-test", transcript, "/home/user/myproject")

        hook_input = make_hook_input(session_id="conv-clear-test", reason="complete")
        _run_hook(
            "session_end",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=lambda *a, **kw: FakeHTTPResponse({}),
            user_config={"retainEveryNTurns": 10},
        )

        assert get_cached_session_transcript("conv-clear-test") is None

    def test_disabled_auto_retain_skips_final_retain(self, monkeypatch, tmp_path):
        messages = [{"role": "user", "content": "hi"}]
        transcript = make_transcript_file(tmp_path, messages)

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_API_URL", "http://fake:9077")
        make_user_config(tmp_path, {"retainEveryNTurns": 10})

        sys.path.insert(0, SCRIPTS_DIR)
        from lib.state import cache_session_transcript

        cache_session_transcript("conv-disabled-test", transcript, "/home/user/myproject")

        captured = {}

        def capture(req, timeout=None):
            captured["called"] = True
            return FakeHTTPResponse({})

        hook_input = make_hook_input(session_id="conv-disabled-test", reason="complete")
        _run_hook(
            "session_end",
            hook_input,
            monkeypatch,
            tmp_path,
            urlopen_side_effect=capture,
            user_config={"autoRetain": False},
        )

        assert "called" not in captured

        from lib.state import get_cached_session_transcript

        assert get_cached_session_transcript("conv-disabled-test") is None
