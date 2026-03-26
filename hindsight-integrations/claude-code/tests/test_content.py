"""Tests for lib/content.py — pure content-processing functions."""

import pytest

from lib.content import (
    _extract_text_content,
    _is_channel_message_tool,
    compose_recall_query,
    format_memories,
    prepare_retention_transcript,
    slice_last_turns_by_user_boundary,
    strip_channel_envelope,
    strip_memory_tags,
    truncate_recall_query,
)


# ---------------------------------------------------------------------------
# strip_channel_envelope
# ---------------------------------------------------------------------------


class TestStripChannelEnvelope:
    def test_strips_channel_xml(self):
        raw = '<channel source="plugin:telegram:telegram" chat_id="123">Hello world</channel>'
        assert strip_channel_envelope(raw) == "Hello world"

    def test_passthrough_plain_text(self):
        assert strip_channel_envelope("just plain text") == "just plain text"

    def test_strips_multiline_channel(self):
        raw = "<channel source='s'>\nline1\nline2\n</channel>"
        assert strip_channel_envelope(raw) == "line1\nline2"

    def test_passthrough_when_no_channel_tag(self):
        raw = "<other>stuff</other>"
        assert strip_channel_envelope(raw) == raw


# ---------------------------------------------------------------------------
# strip_memory_tags
# ---------------------------------------------------------------------------


class TestStripMemoryTags:
    def test_strips_hindsight_memories_block(self):
        raw = "before\n<hindsight_memories>secret</hindsight_memories>\nafter"
        assert "hindsight_memories" not in strip_memory_tags(raw)
        assert "before" in strip_memory_tags(raw)
        assert "after" in strip_memory_tags(raw)

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


# ---------------------------------------------------------------------------
# slice_last_turns_by_user_boundary
# ---------------------------------------------------------------------------


def _msgs(*pairs):
    """Build a message list from (role, content) pairs."""
    return [{"role": r, "content": c} for r, c in pairs]


class TestSliceLastTurnsByUserBoundary:
    def test_returns_all_when_fewer_turns_than_requested(self):
        msgs = _msgs(("user", "hi"), ("assistant", "hello"))
        assert slice_last_turns_by_user_boundary(msgs, 5) == msgs

    def test_slices_to_last_one_turn(self):
        msgs = _msgs(
            ("user", "first"),
            ("assistant", "a1"),
            ("user", "second"),
            ("assistant", "a2"),
        )
        result = slice_last_turns_by_user_boundary(msgs, 1)
        assert result[0]["content"] == "second"
        assert len(result) == 2

    def test_slices_to_last_two_turns(self):
        msgs = _msgs(
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
            ("assistant", "a2"),
            ("user", "u3"),
            ("assistant", "a3"),
        )
        result = slice_last_turns_by_user_boundary(msgs, 2)
        assert result[0]["content"] == "u2"
        assert len(result) == 4

    def test_empty_list_returns_empty(self):
        assert slice_last_turns_by_user_boundary([], 3) == []

    def test_zero_turns_returns_empty(self):
        msgs = _msgs(("user", "hi"))
        assert slice_last_turns_by_user_boundary(msgs, 0) == []

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
        # duplicate user msg should be dropped from context
        assert result.count("same question") == 1

    def test_empty_messages_returns_latest(self):
        result = compose_recall_query("query", [], recall_context_turns=3)
        assert result == "query"

    def test_strips_memory_tags_from_context(self):
        msgs = _msgs(
            ("user", "<hindsight_memories>secret</hindsight_memories> actual question"),
        )
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
        # Allow only the newest context line + latest
        result = truncate_recall_query(query, latest, max_chars=len(f"Prior context:\n\nnewer\n\n{latest}") + 5)
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

    def test_formats_multiple_memories_with_separator(self):
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
# _is_channel_message_tool
# ---------------------------------------------------------------------------


class TestIsChannelMessageTool:
    def test_telegram_send_message(self):
        block = {"type": "tool_use", "name": "mcp__telegram__sendMessage", "input": {"text": "hello"}}
        assert _is_channel_message_tool(block) is True

    def test_slack_reply_tool(self):
        block = {"type": "tool_use", "name": "mcp__slack__reply", "input": {"body": "hi there"}}
        assert _is_channel_message_tool(block) is True

    def test_operational_recall_tool_excluded(self):
        block = {"type": "tool_use", "name": "mcp__hindsight__recall", "input": {"query": "test"}}
        assert _is_channel_message_tool(block) is False

    def test_builtin_bash_tool_excluded(self):
        block = {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}
        assert _is_channel_message_tool(block) is False

    def test_mcp_tool_without_text_field_excluded(self):
        block = {"type": "tool_use", "name": "mcp__something__action", "input": {"id": 123}}
        assert _is_channel_message_tool(block) is False

    def test_mcp_tool_with_empty_text_excluded(self):
        block = {"type": "tool_use", "name": "mcp__telegram__send", "input": {"text": "   "}}
        assert _is_channel_message_tool(block) is False

    def test_mcp_create_action_excluded(self):
        block = {"type": "tool_use", "name": "mcp__notion__create_page", "input": {"content": "hello"}}
        assert _is_channel_message_tool(block) is False


# ---------------------------------------------------------------------------
# _extract_text_content
# ---------------------------------------------------------------------------


class TestExtractTextContent:
    def test_plain_string_returned_as_is(self):
        assert _extract_text_content("hello", role="user") == "hello"

    def test_text_block_extracted(self):
        content = [{"type": "text", "text": "response text"}]
        assert _extract_text_content(content, role="assistant") == "response text"

    def test_thinking_block_excluded(self):
        content = [{"type": "thinking", "thinking": "private"}, {"type": "text", "text": "public"}]
        result = _extract_text_content(content, role="assistant")
        assert "private" not in result
        assert "public" in result

    def test_channel_tool_use_extracted_for_assistant(self):
        content = [{"type": "tool_use", "name": "mcp__telegram__send", "input": {"text": "hello user"}}]
        result = _extract_text_content(content, role="assistant")
        assert "hello user" in result

    def test_tool_use_not_extracted_for_user(self):
        content = [{"type": "tool_use", "name": "mcp__telegram__send", "input": {"text": "hello user"}}]
        result = _extract_text_content(content, role="user")
        assert "hello user" not in result

    def test_empty_list_returns_empty_string(self):
        assert _extract_text_content([], role="assistant") == ""

    def test_non_string_non_list_returns_empty(self):
        assert _extract_text_content(None, role="user") == ""
        assert _extract_text_content(42, role="user") == ""


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

    def test_json_format_with_tool_calls(self):
        """When include_tool_calls=True, output should be JSON with tool_use blocks."""
        import json

        msgs = [
            {"role": "user", "content": "edit the file"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll edit that file."},
                    {
                        "type": "tool_use",
                        "name": "Edit",
                        "input": {"file_path": "/tmp/foo.py", "old_string": "old", "new_string": "new"},
                    },
                ],
            },
        ]
        transcript, count = prepare_retention_transcript(
            msgs, retain_full_window=True, include_tool_calls=True
        )
        assert transcript is not None
        data = json.loads(transcript)
        assert len(data) == 2
        assert data[0]["role"] == "user"
        assert data[1]["role"] == "assistant"
        # Should have both text and tool_use blocks
        block_types = [b["type"] for b in data[1]["content"]]
        assert "text" in block_types
        assert "tool_use" in block_types
        # Tool input should be preserved
        tool_block = next(b for b in data[1]["content"] if b["type"] == "tool_use")
        assert tool_block["name"] == "Edit"
        assert tool_block["input"]["file_path"] == "/tmp/foo.py"

    def test_json_format_excludes_hindsight_mcp_tools(self):
        """Hindsight MCP tools should be excluded even in JSON mode."""
        import json

        msgs = [
            {"role": "user", "content": "recall something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "name": "mcp__hindsight__recall", "input": {"query": "test"}},
                ],
            },
        ]
        transcript, _ = prepare_retention_transcript(
            msgs, retain_full_window=True, include_tool_calls=True
        )
        data = json.loads(transcript)
        assistant_blocks = data[1]["content"]
        assert len(assistant_blocks) == 1
        assert assistant_blocks[0]["type"] == "text"

    def test_json_format_includes_tool_results(self):
        """Tool results should be included in JSON mode."""
        import json

        msgs = [
            {"role": "user", "content": "run ls"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Running ls."},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_result", "tool_use_id": "123", "content": "file1.py\nfile2.py"},
                    {"type": "text", "text": "Here are the files."},
                ],
            },
        ]
        transcript, _ = prepare_retention_transcript(
            msgs, retain_full_window=True, include_tool_calls=True
        )
        data = json.loads(transcript)
        result_msg = next(m for m in data if any(b.get("type") == "tool_result" for b in m["content"]))
        result_block = next(b for b in result_msg["content"] if b["type"] == "tool_result")
        assert "file1.py" in result_block["content"]

    def test_without_tool_calls_uses_text_format(self):
        """Default (include_tool_calls=False) should use legacy text format."""
        msgs = _msgs(("user", "hello"), ("assistant", "world"))
        transcript, _ = prepare_retention_transcript(msgs, retain_full_window=True, include_tool_calls=False)
        assert "[role: user]" in transcript
        assert "[user:end]" in transcript
