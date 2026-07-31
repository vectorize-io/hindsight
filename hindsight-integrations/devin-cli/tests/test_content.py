"""Tests for lib/content.py — pure content-processing functions."""

import json
import re

import pytest

from lib.content import (
    _extract_text_content,
    _is_channel_message_tool,
    compose_recall_query,
    format_current_time,
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

    def test_a_tool_result_message_is_not_a_turn_boundary(self):
        """Tool results arrive as role:"user" messages carrying a tool_result block.

        Counting them as turns puts the boundary on the tool result instead of
        the prompt that caused it, so the window starts mid-turn and omits the
        user input it was meant to include. The sibling openclaw integration
        filters these the same way.
        """
        msgs = [
            {"role": "user", "content": "the real earlier question"},
            {"role": "assistant", "content": [{"type": "tool_use", "name": "read", "input": {}, "id": "t1"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file body"}]},
            {"role": "assistant", "content": "here is the answer"},
            {"role": "user", "content": "the latest question"},
            {"role": "assistant", "content": "a2"},
        ]

        result = slice_last_turns_by_user_boundary(msgs, 2)

        assert result[0]["content"] == "the real earlier question", (
            "the window started at a synthetic tool-result message, dropping a real user turn"
        )
        assert len(result) == 6

    def test_a_user_message_with_both_text_and_a_tool_result_still_counts(self):
        """Only messages with no real text are synthetic."""
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "x"},
                    {"type": "text", "text": "and also, what about this?"},
                ],
            },
            {"role": "assistant", "content": "a2"},
        ]

        result = slice_last_turns_by_user_boundary(msgs, 1)

        assert len(result) == 2


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
        transcript, count = prepare_retention_transcript(msgs, retain_full_window=True, include_tool_calls=True)
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
        transcript, _ = prepare_retention_transcript(msgs, retain_full_window=True, include_tool_calls=True)
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
        transcript, _ = prepare_retention_transcript(msgs, retain_full_window=True, include_tool_calls=True)
        data = json.loads(transcript)
        result_msg = next(m for m in data if any(b.get("type") == "tool_result" for b in m["content"]))
        result_block = next(b for b in result_msg["content"] if b["type"] == "tool_result")
        assert "file1.py" in result_block["content"]

    def test_json_format_handles_list_content_tool_results(self):
        """Tool results with list content (e.g. Agent subagent responses) should be extracted."""
        import json

        msgs = [
            {"role": "user", "content": "analyze the code"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Agent", "input": {"prompt": "check code"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": [
                            {"type": "text", "text": "Found 3 issues in the codebase."},
                            {"type": "text", "text": "1. Missing error handling in auth module"},
                        ],
                    },
                ],
            },
        ]
        transcript, _ = prepare_retention_transcript(msgs, retain_full_window=True, include_tool_calls=True)
        data = json.loads(transcript)
        result_msg = next(m for m in data if any(b.get("type") == "tool_result" for b in m["content"]))
        result_block = next(b for b in result_msg["content"] if b["type"] == "tool_result")
        assert "Found 3 issues" in result_block["content"]
        assert "Missing error handling" in result_block["content"]

    def test_without_tool_calls_uses_text_format(self):
        """Default (include_tool_calls=False) should use legacy text format."""
        msgs = _msgs(("user", "hello"), ("assistant", "world"))
        transcript, _ = prepare_retention_transcript(msgs, retain_full_window=True, include_tool_calls=False)
        assert "[role: user]" in transcript
        assert "[user:end]" in transcript


# ---------------------------------------------------------------------------
# format_current_time
# ---------------------------------------------------------------------------


class TestFormatCurrentTime:
    def test_includes_utc_suffix(self):
        # The "UTC" suffix prevents client LLMs from misreading the
        # timestamp as local time.
        assert format_current_time().endswith(" UTC")

    def test_format_shape(self):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", format_current_time())


class TestOperationalToolResultsAreDropped:
    """Filtering the Hindsight tool_use alone does not close the feedback loop.

    The tool_result is what carries the recalled memories verbatim, and it
    arrives in the *next* message — so the skipped ids must outlive one call.
    """

    def test_recall_result_does_not_re_enter_the_transcript(self):
        from lib.content import _prepare_json_transcript

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "what did we decide about auth?"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu_1", "name": "mcp__hindsight__recall", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "MEMORY: we chose JWT in March"}],
            },
        ]

        transcript, _count = _prepare_json_transcript(messages, {"user", "assistant"})

        assert "we chose JWT in March" not in transcript
        assert "what did we decide about auth?" in transcript

    def test_ordinary_tool_results_are_still_retained(self):
        from lib.content import _prepare_json_transcript

        messages = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu_2", "name": "Read", "input": {"file_path": "a.py"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_2", "content": "file body"}]},
        ]

        transcript, _count = _prepare_json_transcript(messages, {"user", "assistant"})

        assert "file body" in transcript


class TestRecallQueryLimitDisabled:
    def test_zero_max_chars_leaves_the_query_intact(self):
        """`recallMaxQueryChars: 0` is the documented way to disable the limit."""
        from lib.content import truncate_recall_query

        query = "a" * 5000
        assert truncate_recall_query(query, query, 0) == query


class TestMalformedTextBlocksAreSkipped:
    """Content blocks are decoded from Devin CLI's own SQLite payload.

    Their shape is not this plugin's to guarantee, and a `text` block whose
    value is null — what a truncated block looks like — used to raise
    AttributeError out of .strip(), aborting the whole transcript rather than
    dropping the one bad block.
    """

    @pytest.mark.parametrize("bad_text", [None, 42, {"nested": "object"}, ["a", "b"]])
    def test_extract_text_content_skips_a_non_string_text_block(self, bad_text):
        content = [
            {"type": "text", "text": bad_text},
            {"type": "text", "text": "the surviving turn"},
        ]

        result = _extract_text_content(content, role="assistant")

        assert "the surviving turn" in result

    @pytest.mark.parametrize("bad_text", [None, 42, {"nested": "object"}])
    def test_retention_transcript_survives_a_non_string_text_block(self, bad_text):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": bad_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": "still here"}]},
        ]

        transcript, count = prepare_retention_transcript(messages, retain_full_window=True)

        assert transcript is not None
        assert "still here" in transcript
        # The malformed message contributed no usable block, so it drops out of
        # the transcript entirely rather than appearing empty.
        assert count == 1

    def test_a_null_text_block_is_dropped_not_stringified(self):
        """str() coercion would retain the literal "None" as if it were said."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": None}]},
            {"role": "assistant", "content": [{"type": "text", "text": "real content"}]},
        ]

        transcript, _ = prepare_retention_transcript(messages, retain_full_window=True)

        assert "real content" in transcript
        assert "None" not in transcript

    def test_a_tool_use_block_with_a_null_name_does_not_abort_the_transcript(self):
        """`.get("name", "unknown")` only defaults a *missing* key.

        An explicit "name": null reached .startswith() and raised, killing the
        retain hook for the whole session.
        """
        messages = [
            {"role": "user", "content": "do the thing"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": None, "input": {"a": 1}},
                    {"type": "text", "text": "still here"},
                ],
            },
        ]

        transcript, _ = prepare_retention_transcript(messages, retain_full_window=True, include_tool_calls=True)

        assert transcript is not None
        assert "still here" in transcript
        assert "unknown" in transcript, "the unnamed tool call should fall back, not vanish"

    def test_a_tool_result_text_block_with_a_null_text_does_not_abort_the_transcript(self):
        """Same defect as the null `name` above, in the tool_result content path.

        `.get("text", "")` defaults only a missing key, so an explicit null
        reached .strip() and took the whole transcript with it.
        """
        messages = [
            {"role": "user", "content": "run it"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": None},
                            {"type": "text", "text": 42},
                            {"type": "text", "text": "the real output"},
                        ],
                    },
                ],
            },
        ]

        transcript, _ = prepare_retention_transcript(messages, retain_full_window=True, include_tool_calls=True)

        assert transcript is not None
        assert "the real output" in transcript


class TestChannelEnvelopeIsStrippedOnlyWhenItIsTheWholeMessage:
    """An unanchored search corrupted ordinary messages.

    The envelope wraps a channel message end to end. Matching it anywhere in
    the text meant any message that merely *mentioned* one was replaced by its
    inner text — the user's actual words were discarded from both recall and
    retention, silently and irreversibly.
    """

    def test_a_message_containing_an_envelope_is_left_intact(self):
        raw = 'please explain <channel source="x">hello</channel> thanks'
        assert strip_channel_envelope(raw) == raw

    def test_text_before_the_envelope_is_not_discarded(self):
        raw = 'context first <channel source="x">inner</channel>'
        assert strip_channel_envelope(raw) == raw

    def test_text_after_the_envelope_is_not_discarded(self):
        raw = '<channel source="x">inner</channel> and a follow-up question'
        assert strip_channel_envelope(raw) == raw

    def test_a_real_envelope_is_still_stripped(self):
        """Control: the anchoring must not break the case this exists for."""
        raw = '<channel source="plugin:telegram:telegram" chat_id="1">\nHello world\n</channel>'
        assert strip_channel_envelope(raw) == "Hello world"

    def test_surrounding_whitespace_does_not_defeat_the_match(self):
        """The real wrapper puts its tags on their own lines."""
        raw = '\n  <channel source="x">Hello world</channel>\n'
        assert strip_channel_envelope(raw) == "Hello world"


class TestOperationalToolResultsAreSuppressedWithoutAnId:
    """Dropping an operational tool_use is only half of breaking the feedback loop.

    Its *result* is what carries recalled memories back into the transcript, and
    suppression correlates the two by id. When the tool_use block carries no id,
    nothing is recorded — and the matching tool_result then passed straight
    through, retaining the memories that were just recalled. Every subsequent
    retain re-ingests them, so the bank compounds its own output.
    """

    def _retain(self, msgs):
        transcript, _ = prepare_retention_transcript(msgs, retain_full_window=True, include_tool_calls=True)
        return transcript or ""

    def _msgs(self, tool_use_extra, result_extra):
        return [
            {"role": "user", "content": "what do you know about me?"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "mcp__hindsight__recall", "input": {}, **tool_use_extra}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": "MEMORY-LEAKED-BACK", **result_extra}],
            },
        ]

    def test_an_id_less_operational_result_is_dropped(self):
        transcript = self._retain(self._msgs({}, {}))
        assert "MEMORY-LEAKED-BACK" not in transcript

    def test_a_null_tool_use_id_is_treated_as_absent(self):
        transcript = self._retain(self._msgs({}, {"tool_use_id": None}))
        assert "MEMORY-LEAKED-BACK" not in transcript

    def test_matching_ids_still_suppress(self):
        transcript = self._retain(self._msgs({"id": "call_1"}, {"tool_use_id": "call_1"}))
        assert "MEMORY-LEAKED-BACK" not in transcript

    def test_an_unrelated_result_is_kept_when_ids_are_present(self):
        """Suppression must stay targeted — this is ordinary transcript content."""
        transcript = self._retain(self._msgs({"id": "call_1"}, {"tool_use_id": "call_2"}))
        assert "MEMORY-LEAKED-BACK" in transcript

    def test_an_id_less_result_survives_when_no_id_less_call_was_dropped(self):
        msgs = [
            {"role": "user", "content": "run ls"},
            {"role": "assistant", "content": [{"type": "tool_use", "name": "Bash", "input": {}, "id": "call_1"}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "MEMORY-LEAKED-BACK"}]},
        ]
        assert "MEMORY-LEAKED-BACK" in self._retain(msgs)

    def test_an_unhashable_tool_use_id_does_not_raise(self):
        """A set lookup on a list raises TypeError and takes the retain hook down."""
        transcript = self._retain(self._msgs({"id": "call_1"}, {"tool_use_id": ["call_1"]}))
        assert "MEMORY-LEAKED-BACK" in transcript
