"""Tests for content formatting."""

from hindsight_zed.content import compose_recall_query, format_memory_block, format_transcript
from hindsight_zed.threads_db import ThreadMessage, ZedThread


def test_compose_recall_query_uses_last_user_message():
    msgs = [
        ThreadMessage("user", "first question"),
        ThreadMessage("assistant", "an answer"),
        ThreadMessage("user", "the latest question"),
    ]
    assert compose_recall_query(msgs) == "the latest question"


def test_compose_recall_query_truncates():
    long = "x" * 2000
    assert len(compose_recall_query([ThreadMessage("user", long)], max_chars=100)) == 100


def test_compose_recall_query_empty():
    assert compose_recall_query([]) == ""


def test_format_memory_block():
    results = [
        {"text": "User prefers pytest", "type": "world", "mentioned_at": "2026-06-01"},
        {"text": "Building a parser", "type": "experience"},
        {"text": "  "},  # blank — skipped
        "not-a-dict",  # ignored
    ]
    block = format_memory_block(results)
    assert "- User prefers pytest [world] (2026-06-01)" in block
    assert "- Building a parser [experience]" in block
    assert block.count("\n- ") == 1  # two items -> one joiner; blank/invalid excluded


def test_format_memory_block_empty():
    assert format_memory_block([]) == ""


def test_format_transcript():
    thread = ZedThread(
        id="t1",
        title="x",
        updated_at="2026-06-10T00:00:00Z",
        messages=[ThreadMessage("user", "hello"), ThreadMessage("assistant", "hi there")],
    )
    out = format_transcript(thread)
    assert "[user]\nhello" in out
    assert "[assistant]\nhi there" in out
