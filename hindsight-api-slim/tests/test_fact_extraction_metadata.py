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


def test_build_user_message_restates_language_rule_after_content():
    """The keep-the-input's-language reminder follows the content, and is dropped when an output language is set."""
    kwargs = dict(chunk="Mañana tengo la entrevista.", chunk_index=0, total_chunks=1, event_date=None, context="")

    msg = _build_user_message(**kwargs)
    assert msg.endswith(
        "Mañana tengo la entrevista.\n\nWrite each fact in the language the Content above is written in."
    )

    # An explicit output language contradicts it (#3776), so it goes, and the content ends the message.
    assert _build_user_message(**kwargs, output_language="English").endswith("Mañana tengo la entrevista.")


def test_build_user_message_restates_custom_instructions_after_content():
    """In custom mode the message ends by pointing back at the instructions, after the language reminder."""
    kwargs = dict(chunk="Morale is high.", chunk_index=0, total_chunks=1, event_date=None, context="")

    assert "extraction instructions above" not in _build_user_message(**kwargs)
    msg = _build_user_message(**kwargs, custom_instructions=True)
    assert msg.endswith(
        "Write each fact in the language the Content above is written in.\n\n"
        "Extract only what the extraction instructions above ask for. Anything they leave out must not become a fact."
    )
