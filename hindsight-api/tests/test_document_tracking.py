"""
Tests for document tracking and upsert functionality.
"""
import logging
import pytest
from datetime import datetime, timezone
from hindsight_api import RequestContext


@pytest.mark.asyncio
async def test_document_creation_and_retrieval(memory, request_context):
    """Test that documents are created and can be retrieved."""
    bank_id = f"test_doc_{datetime.now(timezone.utc).timestamp()}"

    try:
        document_id = "meeting-001"

        # Store memory with document tracking
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Google. Bob works at Microsoft.",
            context="Team meeting",
            document_id=document_id,
            request_context=request_context,
        )

        # Retrieve document
        doc = await memory.get_document(document_id, bank_id, request_context=request_context)

        assert doc is not None
        assert doc["id"] == document_id
        assert doc["bank_id"] == bank_id
        assert "Alice works at Google" in doc["original_text"]
        assert doc["memory_unit_count"] > 0

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_document_upsert(memory, request_context):
    """Test that providing the same document_id automatically upserts (deletes old units and creates new ones)."""
    bank_id = f"test_upsert_{datetime.now(timezone.utc).timestamp()}"

    try:
        document_id = "meeting-002"

        # First version
        units_v1 = await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Google.",
            context="Initial",
            document_id=document_id,
            request_context=request_context,
        )

        # Get document stats
        doc_v1 = await memory.get_document(document_id, bank_id, request_context=request_context)
        count_v1 = doc_v1["memory_unit_count"]

        # Update with different content (automatic upsert when same document_id is provided)
        units_v2 = await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Microsoft. Bob works at Apple.",
            context="Updated",
            document_id=document_id,
            request_context=request_context,
        )

        # Get updated document stats
        doc_v2 = await memory.get_document(document_id, bank_id, request_context=request_context)
        count_v2 = doc_v2["memory_unit_count"]

        # Verify old units were replaced
        assert "Microsoft" in doc_v2["original_text"]
        assert doc_v2["updated_at"] > doc_v1["created_at"]

        # Different unit IDs (old ones deleted, new ones created)
        assert set(units_v1).isdisjoint(set(units_v2))

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_document_deletion(memory, request_context):
    """Test that deleting a document cascades to memory units."""
    bank_id = f"test_delete_{datetime.now(timezone.utc).timestamp()}"

    try:
        document_id = "meeting-003"

        # Create document
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Google.",
            context="Test",
            document_id=document_id,
            request_context=request_context,
        )

        # Verify it exists
        doc = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc is not None
        assert doc["memory_unit_count"] > 0

        # Delete document
        result = await memory.delete_document(document_id, bank_id, request_context=request_context)
        assert result["document_deleted"] == 1
        assert result["memory_units_deleted"] > 0

        # Verify it's gone
        doc_after = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc_after is None

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_memory_without_document(memory, request_context):
    """Test that memories can still be created without document tracking."""
    bank_id = f"test_no_doc_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Create memory without document_id (backward compatibility)
        units = await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Google.",
            context="Test",
            request_context=request_context,
        )

        assert len(units) > 0

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
