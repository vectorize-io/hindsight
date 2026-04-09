"""
Tests for retain update_mode='append' — appends new content to existing documents.
"""

import logging
from datetime import datetime, timezone

import pytest

from hindsight_api.engine.memory_engine import Budget

logger = logging.getLogger(__name__)


def _ts():
    return datetime.now(timezone.utc).timestamp()


@pytest.mark.asyncio
async def test_append_mode_concatenates_content(memory, request_context):
    """
    When update_mode='append', new content should be appended to the existing
    document and the full document should be reprocessed. Facts from both
    old and new content should be recallable.
    """
    bank_id = f"test_append_{_ts()}"
    document_id = "conversation-append"

    try:
        # First retain — initial content
        v1_units = await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Google as a software engineer.",
            context="team info",
            document_id=document_id,
            request_context=request_context,
        )
        assert len(v1_units) > 0, "v1 should create facts"

        doc_v1 = await memory.get_document(document_id, bank_id, request_context=request_context)
        v1_text = doc_v1["original_text"]
        assert "Alice works at Google" in v1_text

        # Second retain with append — add new content
        v2_units = await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": "Bob works at Microsoft as a data scientist.",
                    "context": "team info",
                    "document_id": document_id,
                    "update_mode": "append",
                }
            ],
            request_context=request_context,
        )

        # Verify document now contains both old and new content
        doc_v2 = await memory.get_document(document_id, bank_id, request_context=request_context)
        v2_text = doc_v2["original_text"]
        assert "Alice works at Google" in v2_text, "Original content should be preserved"
        assert "Bob works at Microsoft" in v2_text, "New content should be appended"

        # Verify facts from both old and new content are recallable
        result_alice = await memory.recall_async(
            bank_id=bank_id,
            query="Where does Alice work?",
            budget=Budget.MID,
            max_tokens=1000,
            request_context=request_context,
        )
        assert len(result_alice.results) > 0, "Should recall facts about Alice"

        result_bob = await memory.recall_async(
            bank_id=bank_id,
            query="Where does Bob work?",
            budget=Budget.MID,
            max_tokens=1000,
            request_context=request_context,
        )
        assert len(result_bob.results) > 0, "Should recall facts about Bob"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_append_mode_no_existing_document(memory, request_context):
    """
    When update_mode='append' but no existing document exists,
    it should behave like a normal retain (no content to prepend).
    """
    bank_id = f"test_append_new_{_ts()}"
    document_id = "new-doc-append"

    try:
        units = await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": "Charlie is a product manager at Stripe.",
                    "context": "team info",
                    "document_id": document_id,
                    "update_mode": "append",
                }
            ],
            request_context=request_context,
        )

        assert len(units) > 0, "Should create facts even with no existing document"
        # Flatten if nested
        flat_units = units[0] if units and isinstance(units[0], list) else units
        assert len(flat_units) > 0

        doc = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert "Charlie is a product manager" in doc["original_text"]

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_append_mode_requires_document_id(memory, request_context):
    """update_mode='append' without document_id should raise ValueError."""
    bank_id = f"test_append_no_docid_{_ts()}"

    with pytest.raises(ValueError, match="update_mode='append' requires a document_id"):
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": "Some content",
                    "update_mode": "append",
                }
            ],
            request_context=request_context,
        )


@pytest.mark.asyncio
async def test_append_mode_multiple_appends(memory, request_context):
    """Multiple appends should accumulate content over successive retains."""
    bank_id = f"test_multi_append_{_ts()}"
    document_id = "multi-append-doc"

    try:
        # Initial retain
        await memory.retain_async(
            bank_id=bank_id,
            content="Day 1: Alice joined the team.",
            context="journal",
            document_id=document_id,
            request_context=request_context,
        )

        # First append
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": "Day 2: Alice completed her onboarding.",
                    "context": "journal",
                    "document_id": document_id,
                    "update_mode": "append",
                }
            ],
            request_context=request_context,
        )

        # Second append
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": "Day 3: Alice shipped her first feature.",
                    "context": "journal",
                    "document_id": document_id,
                    "update_mode": "append",
                }
            ],
            request_context=request_context,
        )

        # Verify all content is present
        doc = await memory.get_document(document_id, bank_id, request_context=request_context)
        text = doc["original_text"]
        assert "Day 1" in text, "Original content should be present"
        assert "Day 2" in text, "First append should be present"
        assert "Day 3" in text, "Second append should be present"

        # All days should be recallable
        result = await memory.recall_async(
            bank_id=bank_id,
            query="What happened on Alice's first days?",
            budget=Budget.MID,
            max_tokens=1000,
            request_context=request_context,
        )
        assert len(result.results) > 0, "Should recall facts from all appends"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_replace_mode_is_default(memory, request_context):
    """Without update_mode (or update_mode='replace'), retain should replace content."""
    bank_id = f"test_replace_default_{_ts()}"
    document_id = "replace-doc"

    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Google.",
            context="team info",
            document_id=document_id,
            request_context=request_context,
        )

        # Retain again without update_mode — should replace
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": "Bob works at Microsoft.",
                    "context": "team info",
                    "document_id": document_id,
                }
            ],
            request_context=request_context,
        )

        doc = await memory.get_document(document_id, bank_id, request_context=request_context)
        text = doc["original_text"]
        # With replace, only new content should remain
        assert "Bob works at Microsoft" in text, "New content should be present"
        assert "Alice works at Google" not in text, "Old content should be replaced"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
