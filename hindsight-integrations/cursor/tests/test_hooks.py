"""Tests for Cursor plugin hook scripts (recall, session_start, and retain)."""

import importlib
import io
import json
import os
import sys
from unittest.mock import MagicMock, patch

# Import the hook scripts as modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class TestRecallHook:
    def test_skips_when_auto_recall_disabled(self, monkeypatch, capsys):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_AUTO_RECALL", "false")
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": "remember this"})))

        import recall

        importlib.reload(recall)
        recall.main()

        assert capsys.readouterr().out == ""

    def test_hook_input_reader_strips_windows_utf8_bom(self, monkeypatch):
        from lib.hook_io import read_hook_input

        monkeypatch.setattr("sys.stdin", io.BytesIO(b'\xef\xbb\xbf{"prompt":"remember this"}'))

        assert read_hook_input() == {"prompt": "remember this"}

    def test_outputs_context_for_each_prompt(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("CURSOR_PROJECT_DIR", str(tmp_path))
        mock_client = MagicMock()
        mock_client.recall.return_value = {
            "results": [{"text": "User prefers TypeScript", "type": "world", "mentioned_at": "2026-01-01"}]
        }
        hook_input = {
            "prompt": "What language should I use?",
            "conversation_id": "conv-1",
        }
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook_input)))

        import recall

        importlib.reload(recall)
        with (
            patch.object(recall, "get_api_url", return_value="http://localhost:8888"),
            patch.object(recall, "HindsightClient", return_value=mock_client),
            patch.object(recall, "ensure_bank_mission"),
            patch.object(recall, "write_state"),
            patch.object(recall, "write_session_rules", return_value=True) as write_rules,
            patch.object(recall, "ensure_gitignored") as ensure_gitignored,
        ):
            recall.main()

        result = json.loads(capsys.readouterr().out)
        assert result["continue"] is True
        assert "additional_context" not in result
        rule_content = write_rules.call_args.args[1]
        assert "User prefers TypeScript" in rule_content
        assert "hindsight_memories" in rule_content
        assert mock_client.recall.call_args.kwargs["query"] == "What language should I use?"
        write_rules.assert_called_once()
        ensure_gitignored.assert_called_once()

    def test_no_output_on_empty_results(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        mock_client = MagicMock()
        mock_client.recall.return_value = {"results": []}
        hook_input = {"prompt": "What should I remember?", "workspace_roots": [str(tmp_path)]}
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook_input)))

        import recall

        importlib.reload(recall)
        with (
            patch.object(recall, "get_api_url", return_value="http://localhost:8888"),
            patch.object(recall, "HindsightClient", return_value=mock_client),
            patch.object(recall, "ensure_bank_mission"),
            patch.object(recall, "write_state"),
        ):
            recall.main()

        assert capsys.readouterr().out == ""

    def test_recall_timeout_is_configurable(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_RECALL_TIMEOUT", "42")
        mock_client = MagicMock()
        mock_client.recall.return_value = {"results": [{"text": "memory"}]}
        monkeypatch.setattr(
            "sys.stdin",
            io.StringIO(json.dumps({"prompt": "What should I remember?", "workspace_roots": [str(tmp_path)]})),
        )

        import recall

        importlib.reload(recall)
        with (
            patch.object(recall, "get_api_url", return_value="http://localhost:8888"),
            patch.object(recall, "HindsightClient", return_value=mock_client),
            patch.object(recall, "ensure_bank_mission"),
            patch.object(recall, "write_state"),
            patch.object(recall, "write_session_rules", return_value=False),
        ):
            recall.main()

        assert mock_client.recall.call_args.kwargs["timeout"] == 42

    def test_multi_turn_query_uses_cursor_transcript(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_RECALL_CONTEXT_TURNS", "2")
        mock_client = MagicMock()
        mock_client.recall.return_value = {"results": []}
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            '{"role":"user","content":"I prefer Python"}\n'
            '{"role":"assistant","content":"Noted"}\n'
        )
        monkeypatch.setenv("CURSOR_TRANSCRIPT_PATH", str(transcript_path))
        hook_input = {
            "prompt": "Which language should I use?",
            "workspace_roots": [str(tmp_path)],
        }
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook_input)))

        import recall

        importlib.reload(recall)
        with (
            patch.object(recall, "get_api_url", return_value="http://localhost:8888"),
            patch.object(recall, "HindsightClient", return_value=mock_client),
            patch.object(recall, "ensure_bank_mission"),
            patch.object(recall, "write_state"),
        ):
            recall.main()

        assert "I prefer Python" in mock_client.recall.call_args.kwargs["query"]

    def test_filters_recall_results_by_score_floor(self):
        import recall

        results = [
            {"text": "keep", "scores": {"final": 0.8}},
            {"text": "drop", "scores": {"final": 0.2}},
            {"text": "missing score", "scores": {}},
        ]
        filtered = recall.filter_by_min_scores(results, {"final": 0.5}, {})

        assert [result["text"] for result in filtered] == ["keep", "missing score"]


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
        assert "additional_context" in result
        assert "User prefers TypeScript" in result["additional_context"]
        assert "hindsight_memories" in result["additional_context"]

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

    def test_recall_timeout_is_configurable_for_session_start(self, monkeypatch, capsys):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_RECALL_TIMEOUT", "42")

        mock_client = MagicMock()
        mock_client.recall.return_value = {"results": []}
        monkeypatch.setattr(
            "sys.stdin",
            io.StringIO(json.dumps({"workspace_roots": ["/tmp/test-project"]})),
        )

        import session_start

        importlib.reload(session_start)

        with (
            patch.object(session_start, "get_api_url", return_value="http://localhost:8888"),
            patch.object(session_start, "HindsightClient", return_value=mock_client),
            patch.object(session_start, "ensure_bank_mission"),
        ):
            session_start.main()

        assert mock_client.recall.call_args.kwargs["timeout"] == 42

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
        transcript.write_text(
            '{"role": "user", "content": "Hello"}\n{"role": "assistant", "content": "Hi back"}\n',
            encoding="utf-8",
        )
        msgs = retain.read_transcript(str(transcript))
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "Hello"}
        assert msgs[1] == {"role": "assistant", "content": "Hi back"}

    def test_read_transcript_accepts_utf8_content_on_windows(self, tmp_path):
        import retain

        transcript = tmp_path / "unicode.jsonl"
        transcript.write_text(
            json.dumps({"role": "user", "content": "Use café and 日本語"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        assert retain.read_transcript(str(transcript))[0]["content"] == "Use café and 日本語"

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
        assert "[tool_use:Shell]" not in msgs[1]["content"]

        rich_msgs = retain.read_transcript(str(transcript), include_tools=True)
        assert isinstance(rich_msgs[1]["content"], list)
        assert rich_msgs[1]["content"][1]["type"] == "tool_use"

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

    def test_retain_excludes_external_previews_and_tool_markers(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_RETAIN_EVERY_N_TURNS", "1")
        monkeypatch.setattr("os.path.expanduser", lambda _: str(tmp_path / "home"))

        mock_client = MagicMock()
        mock_client.retain.return_value = {"status": "ok"}
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            '{"role":"user","message":{"content":['
            '{"type":"text","text":"Use pytest"},'
            '{"type":"text","text":"<external_links>scraped preview</external_links>"}]}}\n'
            '{"role":"assistant","message":{"content":['
            '{"type":"text","text":"Will do."},'
            '{"type":"tool_use","name":"WebSearch","input":{"query":"secret"}}]}}\n'
        )
        monkeypatch.setattr(
            "sys.stdin",
            io.StringIO(
                json.dumps(
                    {
                        "conversation_id": "conv-clean",
                        "transcript_path": str(transcript_path),
                    }
                )
            ),
        )
        monkeypatch.setenv("CURSOR_PLUGIN_DATA", str(tmp_path / "data"))

        import retain

        importlib.reload(retain)
        with (
            patch.object(retain, "get_api_url", return_value="http://localhost:8888"),
            patch.object(retain, "HindsightClient", return_value=mock_client),
            patch.object(retain, "ensure_bank_mission"),
        ):
            retain.main()

        saved_content = mock_client.retain.call_args.kwargs["content"]
        assert "Use pytest" in saved_content
        assert "Will do." in saved_content
        assert "external_links" not in saved_content
        assert "scraped preview" not in saved_content
        assert "WebSearch" not in saved_content
        assert "secret" not in saved_content

    def test_retains_transcript_from_cursor_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_RETAIN_EVERY_N_TURNS", "1")

        mock_client = MagicMock()
        mock_client.retain.return_value = {"status": "ok"}

        transcript_path = tmp_path / "cursor-session.jsonl"
        transcript_path.write_text(
            '{"role":"user","message":{"content":[{"type":"text","text":"Use pytest"}]}}\n'
            '{"role":"assistant","message":{"content":[{"type":"text","text":"Will do."}]}}\n'
        )
        monkeypatch.setenv("CURSOR_TRANSCRIPT_PATH", str(transcript_path))
        monkeypatch.setenv("CURSOR_PLUGIN_DATA", str(tmp_path / "data"))
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"status": "completed", "loop_count": 0})))

        import retain

        importlib.reload(retain)
        with (
            patch.object(retain, "get_api_url", return_value="http://localhost:8888"),
            patch.object(retain, "HindsightClient", return_value=mock_client),
            patch.object(retain, "ensure_bank_mission"),
        ):
            retain.main()

        mock_client.retain.assert_called_once()
        assert mock_client.retain.call_args.kwargs["document_id"].startswith("cursor-session-")

    def test_full_session_document_id_is_scoped_to_conversation(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_RETAIN_EVERY_N_TURNS", "1")
        mock_client = MagicMock()
        mock_client.retain.return_value = {"status": "ok"}

        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            '{"role": "user", "content": "Remember this"}\n'
            '{"role": "assistant", "content": "I will"}\n'
        )
        hook_input = {
            "conversation_id": "conv-upsert",
            "transcript_path": str(transcript_path),
            "workspace_roots": [str(tmp_path)],
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

        assert mock_client.retain.call_args.kwargs["document_id"].startswith("conv-upsert-")


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
        assert "beforeSubmitPrompt" in hooks["hooks"]
        assert "stop" in hooks["hooks"]
        assert hooks["hooks"]["sessionStart"][0]["command"].startswith("python3 ")
        assert hooks["hooks"]["sessionStart"][0]["timeout"] == 15
        assert hooks["hooks"]["beforeSubmitPrompt"][0]["command"].startswith("python3 ")
        assert hooks["hooks"]["beforeSubmitPrompt"][0]["timeout"] == 45
        assert hooks["hooks"]["stop"][0]["command"].startswith("python3 ")
        assert hooks["hooks"]["stop"][0]["timeout"] == 15

    def test_settings_json_valid(self):
        settings_path = os.path.join(os.path.dirname(__file__), "..", "settings.json")
        with open(settings_path) as f:
            settings = json.load(f)

        assert settings["bankId"] == "cursor"
        assert settings["retainContext"] == "cursor"
        assert settings["agentName"] == "cursor"
        assert "autoRecall" in settings
        assert "autoRetain" in settings
