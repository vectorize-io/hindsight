"""
Unit tests for metadata inclusion in fact extraction LLM prompt.
"""

from datetime import datetime

from hindsight_api.engine.retain.fact_extraction import _build_user_message


def test_build_user_message_includes_metadata():
    """Semantic metadata key-value pairs should appear in the user message."""
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
    assert "author" in msg
    assert "Alice" in msg
    assert "  source:" in msg
    assert "confluence" in msg


def test_build_user_message_strips_routing_metadata_from_semantic_prompt():
    """Routing/storage metadata should not be available as semantic actors."""
    event_date = datetime(2024, 6, 15, 12, 0, 0)
    metadata = {
        "source_id": "nimbus-agent",
        "bank_id": "team-memory-prod",
        "source_system": "agent-runtime",
        "title": "Q2 Planning Doc",
        "author": "Alice",
    }

    msg = _build_user_message(
        chunk="Some content.",
        chunk_index=0,
        total_chunks=1,
        event_date=event_date,
        context="planning meeting",
        metadata=metadata,
    )

    assert "Q2 Planning Doc" in msg
    assert "Alice" in msg
    assert "nimbus-agent" not in msg
    assert "team-memory-prod" not in msg
    assert "agent-runtime" not in msg
    assert "  source_id:" not in msg
    assert "  bank_id:" not in msg
    assert "  source_system:" not in msg


def test_build_user_message_strips_source_system_style_metadata_values():
    event_date = datetime(2024, 6, 15, 12, 0, 0)
    metadata = {
        "sourceSystem": "agent-runtime",
        "sourceSystemId": "runtime-prod",
        "integration_trace": "trace-123",
        "title": "Q2 Planning Doc",
        "author": "Alice",
    }

    msg = _build_user_message(
        chunk="Some content.",
        chunk_index=0,
        total_chunks=1,
        event_date=event_date,
        context="planning meeting",
        metadata=metadata,
    )

    assert "Q2 Planning Doc" in msg
    assert "Alice" in msg
    assert "sourceSystem" not in msg
    assert "agent-runtime" not in msg
    assert "sourceSystemId" not in msg
    assert "runtime-prod" not in msg
    assert "integration_trace" not in msg
    assert "trace-123" not in msg


def test_build_user_message_strips_routing_metadata_aliases_and_nested_equivalents():
    event_date = datetime(2024, 6, 15, 12, 0, 0)
    metadata = {
        "bank": "team-memory-prod",
        "bankName": "Team Memory Production",
        "profile": "prod-profile",
        "profileName": "Production",
        "sourceName": "AgentRuntime",
        "sourceSystem": "agent-runtime",
        "title": "Q2 Planning Doc",
        "author": "Alice",
        "document": {
            "title": "Nested semantic title",
            "bankName": "Nested Bank",
            "source_id": "nested-source",
            "authors": [{"name": "Bob", "profileName": "Nested Profile"}],
        },
    }

    msg = _build_user_message(
        chunk="Some content.",
        chunk_index=0,
        total_chunks=1,
        event_date=event_date,
        context="planning meeting",
        metadata=metadata,
    )

    assert "Q2 Planning Doc" in msg
    assert "Alice" in msg
    assert "Nested semantic title" in msg
    assert "Bob" in msg
    assert "team-memory-prod" not in msg
    assert "Team Memory Production" not in msg
    assert "prod-profile" not in msg
    assert "Production" not in msg
    assert "AgentRuntime" not in msg
    assert "agent-runtime" not in msg
    assert "Nested Bank" not in msg
    assert "nested-source" not in msg
    assert "Nested Profile" not in msg


def test_build_user_message_preserves_semantic_values_that_look_like_prefixed_text():
    event_date = datetime(2024, 6, 15, 12, 0, 0)
    metadata = {
        "summary": "source: customer interview confirms Alice approved the plan",
        "owner": "team: platform",
        "title": "Q2 Planning Doc",
    }

    msg = _build_user_message(
        chunk="Some content.",
        chunk_index=0,
        total_chunks=1,
        event_date=event_date,
        context="planning meeting",
        metadata=metadata,
    )

    assert "source: customer interview confirms Alice approved the plan" in msg
    assert "team: platform" in msg
    assert "Q2 Planning Doc" in msg


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
