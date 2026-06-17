"""
Unit tests for metadata inclusion in fact extraction LLM prompt.
"""

from datetime import datetime

from hindsight_api.engine.retain.fact_extraction import _build_user_message


def test_build_user_message_includes_metadata():
    """Metadata key-value pairs should appear in the user message."""
    event_date = datetime(2024, 6, 15, 12, 0, 0)
    metadata = {"title": "Q2 Planning Doc", "source": "confluence", "author": "Alice"}

    msg = _build_user_message(
        chunk="Some content.",
        chunk_index=0,
        total_chunks=1,
        event_date=event_date,
        context="planning meeting",
        metadata=metadata,
    )

    assert "title" in msg
    assert "Q2 Planning Doc" in msg
    assert "source" in msg
    assert "confluence" in msg
    assert "author" in msg
    assert "Alice" in msg


def test_build_user_message_no_metadata():
    """When metadata is empty, the message should still be valid and not include a metadata section."""
    event_date = datetime(2024, 6, 15, 12, 0, 0)

    msg = _build_user_message(
        chunk="Some content.",
        chunk_index=0,
        total_chunks=1,
        event_date=event_date,
        context="planning meeting",
        metadata={},
    )

    assert "Some content." in msg
    assert "Metadata:" not in msg


def test_build_user_message_without_metadata_arg():
    """Calling without metadata (default) should behave the same as empty metadata."""
    event_date = datetime(2024, 6, 15, 12, 0, 0)

    msg = _build_user_message(
        chunk="Some content.",
        chunk_index=0,
        total_chunks=1,
        event_date=event_date,
        context="none",
    )

    assert "Some content." in msg
    assert "Metadata:" not in msg
