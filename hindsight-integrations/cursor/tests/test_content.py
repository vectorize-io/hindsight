"""Tests for content processing utilities."""

import pytest
from lib.content import (
    format_memories,
    prepare_retention_transcript,
    slice_last_turns_by_user_boundary,
    strip_memory_tags,
)


class TestStripMemoryTags:
    def test_strips_hindsight_memories(self):
        content = "Hello <hindsight_memories>secret</hindsight_memories> world"
        assert strip_memory_tags(content) == "Hello  world"

    def test_strips_relevant_memories(self):
        content = "Before <relevant_memories>data</relevant_memories> after"
        assert strip_memory_tags(content) == "Before  after"

    def test_no_tags_unchanged(self):
        content = "Just plain text"
        assert strip_memory_tags(content) == "Just plain text"


class TestSliceLastTurns:
    def test_returns_last_n_turns(self):
        messages = [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Reply 2"},
        ]
        result = slice_last_turns_by_user_boundary(messages, 1)
        assert len(result) == 2
        assert result[0]["content"] == "Turn 2"

    def test_returns_all_if_fewer_turns(self):
        messages = [
            {"role": "user", "content": "Only turn"},
            {"role": "assistant", "content": "Reply"},
        ]
        result = slice_last_turns_by_user_boundary(messages, 5)
        assert len(result) == 2

    def test_empty_messages(self):
        assert slice_last_turns_by_user_boundary([], 1) == []


class TestFormatMemories:
    def test_formats_with_type_and_date(self):
        results = [
            {"text": "User likes Python", "type": "world", "mentioned_at": "2026-01-01"},
        ]
        formatted = format_memories(results)
        assert "User likes Python" in formatted
        assert "[world]" in formatted
        assert "(2026-01-01)" in formatted

    def test_empty_results(self):
        assert format_memories([]) == ""


class TestPrepareRetentionTranscript:
    def test_formats_user_assistant(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        transcript, count = prepare_retention_transcript(messages, ["user", "assistant"], True)
        assert transcript is not None
        assert count == 2
        assert "Hello" in transcript
        assert "Hi there" in transcript

    def test_empty_messages(self):
        transcript, count = prepare_retention_transcript([], ["user", "assistant"], True)
        assert transcript is None
        assert count == 0

    def test_strips_memory_tags(self):
        messages = [
            {"role": "user", "content": "Hello <hindsight_memories>injected</hindsight_memories>"},
        ]
        transcript, count = prepare_retention_transcript(messages, ["user"], True)
        assert transcript is not None
        assert "hindsight_memories" not in transcript
        assert "injected" not in transcript
