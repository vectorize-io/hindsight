"""Tests for lib/content.py — pure content-processing functions."""

import json
import os
import sys

import pytest

from lib.content import (
    compose_recall_query,
    format_memories,
    is_synthetic_codex_user_message,
    prepare_retention_transcript,
    read_transcript,
    slice_last_turns_by_user_boundary,
    strip_memory_tags,
    truncate_recall_query,
)


def _write_jsonl(tmp_path, entries):
    f = tmp_path / "transcript.jsonl"
    f.write_text("\n".join(json.dumps(e) for e in entries))
    return str(f)


# ---------------------------------------------------------------------------
# strip_memory_tags
# ---------------------------------------------------------------------------


class TestStripMemoryTags:
    def test_strips_hindsight_memories_block(self):
        raw = "before\n<hindsight_memories>secret</hindsight_memories>\nafter"
        result = strip_memory_tags(raw)
        assert "hindsight_memories" not in result
        assert "before" in result
        assert "after" in result

    def test_strips_relevant_memories_block(self):
        raw = "text <relevant_memories>old stuff</relevant_memories> text"
        result = strip_memory_tags(raw)
        assert "relevant_memories" not in result
        assert "old stuff" not in result

    def test_passthrough_clean_text(self):
        raw = "no memory tags here"
        assert strip_memory_tags(raw) == raw

    def test_strips_multiline_block(self):
        raw = "<hindsight_memories>\n- mem1\n- mem2\n</hindsight_memories>"
        assert strip_memory_tags(raw).strip() == ""


class TestSyntheticCodexUserMessage:
    def test_detects_agents_startup_instructions(self):
        raw = "# AGENTS.md instructions for /home/coder/project\n\n<INSTRUCTIONS>\nremember things\n</INSTRUCTIONS>"
        assert is_synthetic_codex_user_message(raw)

    def test_does_not_match_normal_agents_discussion(self):
        raw = "We should update AGENTS.md instructions for this repo."
        assert not is_synthetic_codex_user_message(raw)


# ---------------------------------------------------------------------------
# read_transcript — flat format
# ---------------------------------------------------------------------------


class TestReadTranscriptFlat:
    def test_reads_flat_format(self, tmp_path):
        path = _write_jsonl(tmp_path, [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ])
        msgs = read_transcript(path)
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hello"}

    def test_returns_empty_for_missing_file(self):
        assert read_transcript("/nonexistent/path.jsonl") == []

    def test_returns_empty_for_empty_string(self):
        assert read_transcript("") == []


class TestReadTranscriptCodexFormat:
    def test_reads_codex_response_item_format(self, tmp_path):
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is Python?"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "A programming language."}],
                    "phase": "final_answer",
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "What is Python?"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "A programming language."

    def test_skips_non_final_answer_assistant_messages(self, tmp_path):
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "thinking..."}],
                    "phase": "reasoning",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "The answer is 42."}],
                    "phase": "final_answer",
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path)
        assert len(msgs) == 1
        assert msgs[0]["content"] == "The answer is 42."

    def test_skips_non_message_response_items(self, tmp_path):
        entries = [
            {"type": "response_item", "payload": {"type": "tool_call", "name": "Bash"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_skips_invalid_roles(self, tmp_path):
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "system message"}],
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path)
        assert len(msgs) == 0

    def test_skips_codex_agents_startup_instructions(self, tmp_path):
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "# AGENTS.md instructions for /home/coder/project\n\n<INSTRUCTIONS>\nsetup rules\n</INSTRUCTIONS>",
                        }
                    ],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "actual user request"}],
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path)
        assert len(msgs) == 1
        assert msgs[0]["content"] == "actual user request"

    def test_skips_blank_lines_gracefully(self, tmp_path):
        f = tmp_path / "transcript.jsonl"
        f.write_text('\n{"role": "user", "content": "hi"}\n\n{"role": "assistant", "content": "hey"}\n')
        msgs = read_transcript(str(f))
        assert len(msgs) == 2


# ---------------------------------------------------------------------------
# slice_last_turns_by_user_boundary
# ---------------------------------------------------------------------------


def _msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


class TestSliceLastTurnsByUserBoundary:
    def test_returns_all_when_fewer_turns_than_requested(self):
        msgs = _msgs(("user", "hi"), ("assistant", "hello"))
        assert slice_last_turns_by_user_boundary(msgs, 5) == msgs

    def test_slices_to_last_one_turn(self):
        msgs = _msgs(
            ("user", "first"), ("assistant", "a1"),
            ("user", "second"), ("assistant", "a2"),
        )
        result = slice_last_turns_by_user_boundary(msgs, 1)
        assert result[0]["content"] == "second"
        assert len(result) == 2

    def test_slices_to_last_two_turns(self):
        msgs = _msgs(
            ("user", "u1"), ("assistant", "a1"),
            ("user", "u2"), ("assistant", "a2"),
            ("user", "u3"), ("assistant", "a3"),
        )
        result = slice_last_turns_by_user_boundary(msgs, 2)
        assert result[0]["content"] == "u2"
        assert len(result) == 4

    def test_empty_list_returns_empty(self):
        assert slice_last_turns_by_user_boundary([], 3) == []

    def test_zero_turns_returns_empty(self):
        assert slice_last_turns_by_user_boundary(_msgs(("user", "hi")), 0) == []

    def test_non_list_returns_empty(self):
        assert slice_last_turns_by_user_boundary(None, 1) == []


# ---------------------------------------------------------------------------
# compose_recall_query
# ---------------------------------------------------------------------------


class TestComposeRecallQuery:
    def test_single_turn_returns_latest_only(self):
        msgs = _msgs(("user", "previous"), ("assistant", "reply"))
        result = compose_recall_query("new query", msgs, recall_context_turns=1)
        assert result == "new query"

    def test_multi_turn_includes_prior_context(self):
        msgs = _msgs(("user", "prior question"), ("assistant", "prior answer"))
        result = compose_recall_query("current question", msgs, recall_context_turns=2)
        assert "Prior context:" in result
        assert "prior question" in result
        assert "current question" in result

    def test_skips_duplicate_of_latest_query(self):
        msgs = _msgs(("user", "same question"), ("assistant", "answer"))
        result = compose_recall_query("same question", msgs, recall_context_turns=2)
        assert result.count("same question") == 1

    def test_empty_messages_returns_latest(self):
        result = compose_recall_query("query", [], recall_context_turns=3)
        assert result == "query"

    def test_strips_memory_tags_from_context(self):
        msgs = _msgs(("user", "<hindsight_memories>secret</hindsight_memories> actual question"))
        result = compose_recall_query("now", msgs, recall_context_turns=2)
        assert "hindsight_memories" not in result
        assert "secret" not in result

    def test_filters_by_recall_roles(self):
        msgs = _msgs(("user", "user msg"), ("assistant", "assistant msg"))
        result = compose_recall_query("query", msgs, recall_context_turns=2, recall_roles=["user"])
        assert "user msg" in result
        assert "assistant msg" not in result


# ---------------------------------------------------------------------------
# truncate_recall_query
# ---------------------------------------------------------------------------


class TestTruncateRecallQuery:
    def test_short_query_unchanged(self):
        q = "short"
        assert truncate_recall_query(q, q, max_chars=100) == q

    def test_plain_query_truncated_to_max(self):
        q = "x" * 50
        result = truncate_recall_query(q, q, max_chars=20)
        assert len(result) <= 20

    def test_preserves_latest_when_context_dropped(self):
        latest = "final question"
        query = f"Prior context:\n\nuser: old stuff\nassistant: old reply\n\n{latest}"
        result = truncate_recall_query(query, latest, max_chars=30)
        assert latest in result

    def test_drops_oldest_context_lines_first(self):
        latest = "latest"
        query = f"Prior context:\n\nuser: oldest\nassistant: old\nuser: newer\n\n{latest}"
        max_chars = len(f"Prior context:\n\nnewer\n\n{latest}") + 5
        result = truncate_recall_query(query, latest, max_chars=max_chars)
        if "Prior context:" in result:
            assert "oldest" not in result

    def test_zero_max_returns_query_unchanged(self):
        q = "anything"
        assert truncate_recall_query(q, q, max_chars=0) == q


# ---------------------------------------------------------------------------
# format_memories
# ---------------------------------------------------------------------------


class TestFormatMemories:
    def test_formats_single_memory(self):
        mems = [{"text": "Paris is the capital", "type": "world", "mentioned_at": "2024-01-01"}]
        result = format_memories(mems)
        assert "Paris is the capital" in result
        assert "[world]" in result
        assert "(2024-01-01)" in result

    def test_formats_multiple_memories(self):
        mems = [
            {"text": "mem1", "type": "experience", "mentioned_at": "2024-01-01"},
            {"text": "mem2", "type": "world", "mentioned_at": "2024-02-01"},
        ]
        result = format_memories(mems)
        assert "mem1" in result
        assert "mem2" in result

    def test_empty_list_returns_empty_string(self):
        assert format_memories([]) == ""

    def test_missing_optional_fields_graceful(self):
        mems = [{"text": "bare memory"}]
        result = format_memories(mems)
        assert "bare memory" in result


# ---------------------------------------------------------------------------
# prepare_retention_transcript
# ---------------------------------------------------------------------------


class TestPrepareRetentionTranscript:
    def test_formats_last_turn_by_default(self):
        msgs = _msgs(("user", "old"), ("assistant", "old reply"), ("user", "new"), ("assistant", "new reply"))
        transcript, count = prepare_retention_transcript(msgs, retain_full_window=False)
        assert "new" in transcript
        assert "new reply" in transcript
        assert count == 2

    def test_full_window_retains_all(self):
        msgs = _msgs(("user", "msg1"), ("assistant", "reply1"), ("user", "msg2"), ("assistant", "reply2"))
        transcript, count = prepare_retention_transcript(msgs, retain_full_window=True)
        assert "msg1" in transcript
        assert "msg2" in transcript
        assert count == 4

    def test_strips_memory_tags(self):
        msgs = _msgs(("user", "<hindsight_memories>leaked</hindsight_memories> actual question"))
        transcript, _ = prepare_retention_transcript(msgs, retain_full_window=True)
        assert "leaked" not in transcript
        assert "actual question" in transcript

    def test_filters_by_retain_roles(self):
        msgs = _msgs(("user", "user msg"), ("assistant", "assistant msg"))
        transcript, _ = prepare_retention_transcript(msgs, retain_roles=["user"], retain_full_window=True)
        assert "user msg" in transcript
        assert "assistant msg" not in transcript

    def test_empty_messages_returns_none(self):
        result, count = prepare_retention_transcript([])
        assert result is None
        assert count == 0

    def test_role_markers_present(self):
        msgs = _msgs(("user", "hello"))
        transcript, _ = prepare_retention_transcript(msgs, retain_full_window=True)
        assert "[role: user]" in transcript
        assert "[user:end]" in transcript

    def test_no_user_message_returns_none(self):
        msgs = [{"role": "assistant", "content": "only assistant"}]
        result, _ = prepare_retention_transcript(msgs, retain_full_window=False)
        assert result is None


# ---------------------------------------------------------------------------
# read_transcript — rich format (include_tool_calls=True)
# ---------------------------------------------------------------------------


class TestReadTranscriptRich:
    def test_reads_messages_as_structured_blocks(self, tmp_path):
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi there"}],
                    "phase": "final_answer",
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path, include_tool_calls=True)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == [{"type": "text", "text": "hello"}]
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == [{"type": "text", "text": "hi there"}]

    def test_skips_codex_agents_startup_instructions(self, tmp_path):
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "# AGENTS.md instructions for /home/coder/project\n\n<INSTRUCTIONS>\nsetup rules\n</INSTRUCTIONS>",
                        }
                    ],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "actual user request"}],
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path, include_tool_calls=True)
        assert len(msgs) == 1
        assert msgs[0]["content"] == [{"type": "text", "text": "actual user request"}]

    def test_reads_function_call(self, tmp_path):
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "list files"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "Read",
                    "arguments": '{"file_path": "/tmp/foo.txt"}',
                    "call_id": "call_1",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "file contents here",
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path, include_tool_calls=True)
        assert len(msgs) == 2  # user + assistant
        assistant = msgs[1]
        assert assistant["role"] == "assistant"
        assert len(assistant["content"]) == 2
        assert assistant["content"][0]["type"] == "tool_use"
        assert assistant["content"][0]["name"] == "Read"
        assert assistant["content"][0]["input"] == {"file_path": "/tmp/foo.txt"}
        assert assistant["content"][1]["type"] == "tool_result"
        assert "file contents here" in assistant["content"][1]["content"]

    def test_reads_local_shell_call(self, tmp_path):
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "run ls"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "local_shell_call",
                    "call_id": "call_1",
                    "status": "completed",
                    "action": {"type": "exec", "command": ["ls", "-la"]},
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path, include_tool_calls=True)
        assistant = msgs[1]
        assert assistant["content"][0]["type"] == "tool_use"
        assert assistant["content"][0]["name"] == "shell"
        assert assistant["content"][0]["input"]["command"] == ["ls", "-la"]

    def test_reads_exec_command_end_event(self, tmp_path):
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "check git status"}],
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "exec_command_end",
                    "call_id": "call_1",
                    "command": ["git", "status"],
                    "aggregated_output": "On branch main\nnothing to commit",
                    "exit_code": 0,
                    "status": "completed",
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path, include_tool_calls=True)
        assistant = msgs[1]
        assert assistant["content"][0]["type"] == "tool_use"
        assert assistant["content"][0]["name"] == "shell"
        assert assistant["content"][0]["input"]["command"] == ["git", "status"]
        assert assistant["content"][1]["type"] == "tool_result"
        assert "On branch main" in assistant["content"][1]["content"]

    def test_reads_exec_command_end_with_nonzero_exit(self, tmp_path):
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "run bad cmd"}],
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "exec_command_end",
                    "call_id": "call_1",
                    "command": ["false"],
                    "aggregated_output": "",
                    "exit_code": 1,
                    "status": "failed",
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path, include_tool_calls=True)
        assistant = msgs[1]
        result = assistant["content"][1]
        assert "exit_code: 1" in result["content"]
        assert "status: failed" in result["content"]

    def test_reads_patch_apply_end_event(self, tmp_path):
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "fix the bug"}],
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "patch_apply_end",
                    "call_id": "call_1",
                    "changes": [{"file": "main.py", "diff": "+fixed line"}],
                    "status": "applied",
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path, include_tool_calls=True)
        assistant = msgs[1]
        assert assistant["content"][0]["type"] == "tool_use"
        assert assistant["content"][0]["name"] == "patch"
        assert assistant["content"][1]["content"] == "status: applied"

    def test_reads_web_search_call(self, tmp_path):
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "search for python docs"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "python documentation"},
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path, include_tool_calls=True)
        assistant = msgs[1]
        assert assistant["content"][0]["type"] == "tool_use"
        assert assistant["content"][0]["name"] == "web_search"
        assert assistant["content"][0]["input"]["query"] == "python documentation"

    def test_skips_non_final_answer_in_rich_mode(self, tmp_path):
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "thinking..."}],
                    "phase": "reasoning",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "The answer"}],
                    "phase": "final_answer",
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path, include_tool_calls=True)
        assert len(msgs) == 1
        assert msgs[0]["content"][0]["text"] == "The answer"

    def test_flat_format_works_in_rich_mode(self, tmp_path):
        path = _write_jsonl(tmp_path, [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        msgs = read_transcript(path, include_tool_calls=True)
        assert len(msgs) == 2
        assert msgs[0]["content"] == [{"type": "text", "text": "hello"}]
        assert msgs[1]["content"] == [{"type": "text", "text": "hi"}]

    def test_truncates_long_tool_output(self, tmp_path):
        long_output = "x" * 3000
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "do something"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "Read",
                    "arguments": "{}",
                    "call_id": "call_1",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": long_output,
                },
            },
        ]
        path = _write_jsonl(tmp_path, entries)
        msgs = read_transcript(path, include_tool_calls=True)
        result = msgs[1]["content"][1]
        assert len(result["content"]) < 3000
        assert "truncated" in result["content"]


# ---------------------------------------------------------------------------
# prepare_retention_transcript — JSON format (include_tool_calls=True)
# ---------------------------------------------------------------------------


class TestPrepareRetentionTranscriptJson:
    def test_json_format_basic(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        transcript, count = prepare_retention_transcript(
            msgs, retain_full_window=True, include_tool_calls=True
        )
        assert count == 2
        parsed = json.loads(transcript)
        assert len(parsed) == 2
        assert parsed[0]["role"] == "user"
        assert parsed[1]["role"] == "assistant"

    def test_json_format_with_tool_calls(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "list files"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "shell", "input": {"command": ["ls"]}},
                    {"type": "tool_result", "content": "file1.txt"},
                    {"type": "text", "text": "Here are the files."},
                ],
            },
        ]
        transcript, count = prepare_retention_transcript(
            msgs, retain_full_window=True, include_tool_calls=True
        )
        parsed = json.loads(transcript)
        assistant = parsed[1]
        assert any(b["type"] == "tool_use" for b in assistant["content"])
        assert any(b["type"] == "tool_result" for b in assistant["content"])

    def test_json_format_strips_memory_tags(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<hindsight_memories>secret</hindsight_memories> real question"},
                ],
            },
        ]
        transcript, _ = prepare_retention_transcript(
            msgs, retain_full_window=True, include_tool_calls=True
        )
        assert "hindsight_memories" not in transcript
        assert "secret" not in transcript
        assert "real question" in transcript

    def test_json_format_filters_roles(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "user msg"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "assistant msg"}]},
        ]
        transcript, count = prepare_retention_transcript(
            msgs, retain_roles=["user"], retain_full_window=True, include_tool_calls=True
        )
        parsed = json.loads(transcript)
        assert count == 1
        assert all(m["role"] == "user" for m in parsed)
