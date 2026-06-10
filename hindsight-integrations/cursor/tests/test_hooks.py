"""Tests for Cursor plugin hook scripts (session_start and retain)."""

import importlib
import io
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Import the hook scripts as modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class TestSessionStartHook:
    def test_skips_when_auto_recall_disabled(self, monkeypatch, capsys):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_AUTO_RECALL", "false")
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"workspace_roots": ["/tmp/test"]})))

        import session_start

        importlib.reload(session_start)
        session_start.main()

        output = capsys.readouterr()
        assert output.out == ""  # No JSON output means no context injected

    def test_outputs_context_on_results(self, monkeypatch, capsys):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")

        mock_client = MagicMock()
        mock_client.recall.return_value = {
            "results": [{"text": "User prefers TypeScript", "type": "world", "mentioned_at": "2026-01-01"}]
        }

        hook_input = {"workspace_roots": ["/tmp/test-project"], "cwd": "/tmp/test-project"}
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook_input)))

        import session_start

        importlib.reload(session_start)

        with (
            patch.object(session_start, "get_api_url", return_value="http://localhost:8888"),
            patch.object(session_start, "HindsightClient", return_value=mock_client),
            patch.object(session_start, "ensure_bank_mission"),
            patch.object(session_start, "write_state"),
        ):
            session_start.main()

        output = capsys.readouterr()
        result = json.loads(output.out)
        assert "additionalContext" in result
        assert "User prefers TypeScript" in result["additionalContext"]
        assert "hindsight_memories" in result["additionalContext"]

    def test_no_output_on_empty_results(self, monkeypatch, capsys):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")

        mock_client = MagicMock()
        mock_client.recall.return_value = {"results": []}

        hook_input = {"workspace_roots": ["/tmp/test-project"]}
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook_input)))

        import session_start

        importlib.reload(session_start)

        with (
            patch.object(session_start, "get_api_url", return_value="http://localhost:8888"),
            patch.object(session_start, "HindsightClient", return_value=mock_client),
            patch.object(session_start, "ensure_bank_mission"),
        ):
            session_start.main()

        output = capsys.readouterr()
        assert output.out == ""

    def test_builds_query_from_workspace_roots(self, monkeypatch, capsys):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")

        mock_client = MagicMock()
        mock_client.recall.return_value = {"results": [{"text": "Uses FastAPI", "type": "world"}]}

        hook_input = {"workspace_roots": ["/home/user/projects/my-app"]}
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook_input)))

        import session_start

        importlib.reload(session_start)

        with (
            patch.object(session_start, "get_api_url", return_value="http://localhost:8888"),
            patch.object(session_start, "HindsightClient", return_value=mock_client),
            patch.object(session_start, "ensure_bank_mission"),
            patch.object(session_start, "write_state"),
        ):
            session_start.main()

        # Verify the query included the project name
        call_kwargs = mock_client.recall.call_args[1]
        assert "my-app" in call_kwargs["query"]

    def test_allows_daemon_start(self, monkeypatch, capsys):
        """sessionStart should allow daemon start since it runs at session beginning."""
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")

        mock_client = MagicMock()
        mock_client.recall.return_value = {"results": []}

        hook_input = {"workspace_roots": ["/tmp/test"]}
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook_input)))

        import session_start

        importlib.reload(session_start)

        mock_get_url = MagicMock(return_value="http://localhost:9077")
        with (
            patch.object(session_start, "get_api_url", mock_get_url),
            patch.object(session_start, "HindsightClient", return_value=mock_client),
            patch.object(session_start, "ensure_bank_mission"),
        ):
            session_start.main()

        # Verify allow_daemon_start=True was passed
        mock_get_url.assert_called_once()
        call_kwargs = mock_get_url.call_args[1]
        assert call_kwargs["allow_daemon_start"] is True


class TestRetainHook:
    def test_skips_when_auto_retain_disabled(self, monkeypatch, capsys):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_AUTO_RETAIN", "false")
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"conversation_id": "c1"})))

        import retain

        importlib.reload(retain)
        retain.main()

        output = capsys.readouterr()
        assert output.out == ""

    def test_skips_empty_transcript(self, monkeypatch, capsys):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_API_URL", "http://localhost:8888")

        hook_input = {"conversation_id": "c1", "transcript_path": "/nonexistent/transcript.jsonl"}
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook_input)))

        import retain

        importlib.reload(retain)
        retain.main()

    def test_read_transcript_parses_flat_format(self, tmp_path):
        """Sanity: flat shape {role, content} still works after the parser
        rewrite."""
        import retain

        transcript = tmp_path / "flat.jsonl"
        transcript.write_text('{"role": "user", "content": "Hello"}\n{"role": "assistant", "content": "Hi back"}\n')
        msgs = retain.read_transcript(str(transcript))
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "Hello"}
        assert msgs[1] == {"role": "assistant", "content": "Hi back"}

    def test_read_transcript_parses_type_nested_format(self, tmp_path):
        """Sanity: type-nested shape {type: "user", message: {...}}."""
        import retain

        transcript = tmp_path / "typenested.jsonl"
        transcript.write_text(
            '{"type": "user", "message": {"role": "user", "content": "Hello"}}\n'
            '{"type": "assistant", "message": {"role": "assistant", "content": "Hi"}}\n'
        )
        msgs = retain.read_transcript(str(transcript))
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user" and msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "assistant" and msgs[1]["content"] == "Hi"

    def test_read_transcript_parses_cursor3_role_nested_with_block_content(self, tmp_path):
        """Regression: Cursor 3.6.31's stop hook writes transcripts as
        {role: "user", message: {content: [{type:"text", text:"..."}, ...]}}.

        The pre-fix parser checked entry["type"] (missing) or top-level
        entry["content"] (also missing — content is under message) and
        silently dropped every line. retain.py then bailed with
        empty_transcript on every Cursor 3 stop hook, even though the
        transcript file existed and had real content.
        """
        import retain

        transcript = tmp_path / "cursor3.jsonl"
        transcript.write_text(
            '{"role":"user","message":{"content":[{"type":"text","text":"Remember Vim over Emacs"}]}}\n'
            '{"role":"assistant","message":{"content":['
            '{"type":"text","text":"Got it. Saving."},'
            '{"type":"tool_use","name":"Shell","input":{"command":"curl ..."}}]}}\n'
        )
        msgs = retain.read_transcript(str(transcript))
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert "Remember Vim over Emacs" in msgs[0]["content"]
        assert msgs[1]["role"] == "assistant"
        assert "Got it. Saving." in msgs[1]["content"]
        # Tool-use blocks are surfaced as compact markers (not dropped) so
        # downstream Answer:/Thought: detection still sees coherent structure.
        assert "[tool_use:Shell]" in msgs[1]["content"]

    def test_retains_transcript(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_RETAIN_EVERY_N_TURNS", "1")

        mock_client = MagicMock()
        mock_client.retain.return_value = {"status": "ok"}

        # Write a test transcript
        transcript_path = tmp_path / "transcript.jsonl"
        messages = [
            {"role": "user", "content": "Build a React app"},
            {"role": "assistant", "content": "I'll create a React app for you."},
        ]
        transcript_path.write_text("\n".join(json.dumps(m) for m in messages))

        hook_input = {
            "conversation_id": "conv-123",
            "transcript_path": str(transcript_path),
            "cwd": "/tmp/test",
        }
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook_input)))
        monkeypatch.setenv("CURSOR_PLUGIN_DATA", str(tmp_path / "data"))

        import retain

        importlib.reload(retain)

        with (
            patch.object(retain, "get_api_url", return_value="http://localhost:8888"),
            patch.object(retain, "HindsightClient", return_value=mock_client),
            patch.object(retain, "ensure_bank_mission"),
        ):
            retain.main()

        mock_client.retain.assert_called_once()
        call_kwargs = mock_client.retain.call_args
        assert "bank_id" in call_kwargs[1]
        assert call_kwargs[1]["context"] == "cursor"


class TestManifest:
    def test_plugin_json_valid(self):
        plugin_path = os.path.join(os.path.dirname(__file__), "..", ".cursor-plugin", "plugin.json")
        with open(plugin_path) as f:
            manifest = json.load(f)

        assert manifest["name"] == "hindsight-memory"
        assert "description" in manifest
        assert manifest["version"]
        assert manifest["license"] == "MIT"

    def test_hooks_json_valid(self):
        hooks_path = os.path.join(os.path.dirname(__file__), "..", "hooks", "hooks.json")
        with open(hooks_path) as f:
            hooks = json.load(f)

        assert hooks["version"] == 1
        assert "sessionStart" in hooks["hooks"]
        assert "stop" in hooks["hooks"]
        # beforeSubmitPrompt should NOT be present (it doesn't support additionalContext)
        assert "beforeSubmitPrompt" not in hooks["hooks"]

    def test_settings_json_valid(self):
        settings_path = os.path.join(os.path.dirname(__file__), "..", "settings.json")
        with open(settings_path) as f:
            settings = json.load(f)

        assert settings["bankId"] == "cursor"
        assert settings["retainContext"] == "cursor"
        assert settings["agentName"] == "cursor"
        assert "autoRecall" in settings
        assert "autoRetain" in settings
