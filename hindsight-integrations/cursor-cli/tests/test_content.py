"""Tests for lib/content.py — content processing for the Cursor CLI integration."""

import json
import os

from lib.content import (
    compose_recall_query,
    format_current_time,
    format_memories,
    prepare_retention_transcript,
    read_transcript,
    slice_last_turns_by_user_boundary,
    strip_memory_tags,
    truncate_recall_query,
)


class TestStripMemoryTags:
    def test_strips_hindsight_memories_block(self):
        text = "real content <hindsight_memories>noise</hindsight_memories> more"
        out = strip_memory_tags(text)
        assert "noise" not in out
        assert "real content" in out
        assert "more" in out

    def test_strips_relevant_memories_block(self):
        text = "<relevant_memories>old</relevant_memories> fresh"
        assert "old" not in strip_memory_tags(text)
        assert "fresh" in strip_memory_tags(text)

    def test_handles_no_tags(self):
        assert strip_memory_tags("nothing to strip") == "nothing to strip"


class TestReadTranscript:
    def test_returns_empty_for_missing_file(self, tmp_path):
        assert read_transcript(str(tmp_path / "nope.jsonl")) == []

    def test_reads_flat_format(self, tmp_path):
        f = tmp_path / "transcript.jsonl"
        f.write_text(
            "\n".join(
                [
                    json.dumps({"role": "user", "content": "hi"}),
                    json.dumps({"role": "assistant", "content": "hello"}),
                ]
            )
        )
        msgs = read_transcript(str(f))
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hi"}
        assert msgs[1] == {"role": "assistant", "content": "hello"}

    def test_reads_cursor_sdk_envelope(self, tmp_path):
        f = tmp_path / "transcript.jsonl"
        f.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
                        }
                    ),
                ]
            )
        )
        msgs = read_transcript(str(f))
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hi"}
        assert msgs[1] == {"role": "assistant", "content": "hello"}

    def test_rich_reader_preserves_tool_calls(self, tmp_path):
        f = tmp_path / "transcript.jsonl"
        f.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "message": {"role": "user", "content": [{"type": "text", "text": "list files"}]},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {"role": "assistant", "content": [{"type": "text", "text": "running ls"}]},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "tool_call",
                            "name": "shell",
                            "args": {"command": "ls"},
                            "status": "completed",
                            "result": "a.txt\nb.txt",
                        }
                    ),
                ]
            )
        )
        msgs = read_transcript(str(f), include_tool_calls=True)
        # 1 user message + 1 assistant message with structured content
        assert len(msgs) == 2
        assistant = msgs[1]
        assert assistant["role"] == "assistant"
        assert isinstance(assistant["content"], list)
        tool_uses = [b for b in assistant["content"] if b.get("type") == "tool_use"]
        tool_results = [b for b in assistant["content"] if b.get("type") == "tool_result"]
        assert tool_uses and tool_uses[0]["name"] == "shell"
        assert tool_results and "a.txt" in tool_results[0]["content"]


class TestComposeRecallQuery:
    def test_single_turn_returns_latest(self):
        q = compose_recall_query("hi", [], 1)
        assert q == "hi"

    def test_multi_turn_includes_context(self):
        msgs = [
            {"role": "user", "content": "I use Python"},
            {"role": "assistant", "content": "Noted"},
        ]
        q = compose_recall_query("What language?", msgs, 2)
        assert "Python" in q
        assert "Prior context:" in q
        assert "What language?" in q

    def test_skips_memory_tags(self):
        msgs = [
            {"role": "user", "content": "<hindsight_memories>noise</hindsight_memories> keep me"},
        ]
        q = compose_recall_query("anything", msgs, 1)
        assert q == "anything"


class TestTruncateRecallQuery:
    def test_under_limit_unchanged(self):
        q = "short query"
        assert truncate_recall_query(q, q, 100) == q

    def test_over_limit_drops_context(self):
        query = "Prior context:\n\nuser: a\nassistant: b\n\nlatest"
        truncated = truncate_recall_query(query, "latest", 20)
        assert "Prior context:" not in truncated
        assert "latest" in truncated


class TestSliceLastTurns:
    def test_zero_turns(self):
        assert slice_last_turns_by_user_boundary([{"role": "user", "content": "a"}], 0) == []

    def test_slices_to_last_n_user_boundaries(self):
        msgs = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "1a"},
            {"role": "user", "content": "2"},
            {"role": "assistant", "content": "2a"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "3a"},
        ]
        sliced = slice_last_turns_by_user_boundary(msgs, 2)
        # Should include user turn 2 onwards.
        assert [m["content"] for m in sliced if m["role"] == "user"] == ["2", "3"]


class TestFormatMemories:
    def test_empty(self):
        assert format_memories([]) == ""

    def test_with_results(self):
        results = [
            {"text": "Paris is in France", "type": "world", "mentioned_at": "2024-01-01"},
            {"text": "User likes espresso", "type": "experience"},
        ]
        out = format_memories(results)
        assert "Paris is in France [world] (2024-01-01)" in out
        assert "User likes espresso [experience]" in out


class TestFormatCurrentTime:
    def test_format(self):
        out = format_current_time()
        assert len(out) == 16  # YYYY-MM-DD HH:MM
        assert out[4] == "-"
        assert out[10] == " "


class TestPrepareRetentionTranscript:
    def test_full_window_text(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        text, count = prepare_retention_transcript(msgs, retain_full_window=True)
        assert text is not None
        assert "[role: user]" in text
        assert "hi" in text
        assert "[assistant:end]" in text

    def test_last_turn_only(self):
        msgs = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": "new reply"},
        ]
        text, count = prepare_retention_transcript(msgs, retain_full_window=False)
        assert "new question" in text
        assert "old" not in text
        assert count == 2

    def test_empty_returns_none(self):
        text, count = prepare_retention_transcript([])
        assert text is None
        assert count == 0

    def test_strips_hindsight_memories_before_retaining(self):
        msgs = [
            {"role": "user", "content": "<hindsight_memories>x</hindsight_memories> actual question"},
        ]
        text, _ = prepare_retention_transcript(msgs, retain_full_window=True)
        assert "x" not in text
        assert "actual question" in text

    def test_rich_includes_tool_use(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "list files"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "running"},
                    {"type": "tool_use", "name": "shell", "input": {"command": "ls"}},
                    {"type": "tool_result", "content": "a.txt"},
                ],
            },
        ]
        text, count = prepare_retention_transcript(
            msgs,
            retain_full_window=True,
            include_tool_calls=True,
        )
        assert text is not None
        assert "tool_use" in text
        assert "shell" in text
