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


@pytest.mark.asyncio
async def test_document_persisted_with_zero_facts(memory, request_context):
    """
    Test that documents are persisted even when zero facts are extracted.

    This is a regression test for issue #324 where documents with no extractable
    facts were reported as disappearing from the system.
    """
    bank_id = f"test_zero_facts_{datetime.now(timezone.utc).timestamp()}"

    try:
        document_id = "doc-zero-facts"

        # Retain content that produces zero facts (gibberish/random characters)
        units = await memory.retain_async(
            bank_id=bank_id,
            content="xyzabc123 !!!### @@@ $$$",  # Random characters unlikely to produce facts
            context="Test zero facts",
            document_id=document_id,
            request_context=request_context,
        )

        # Should return empty unit list (no facts extracted)
        assert len(units) == 0, "Should extract zero facts from gibberish content"

        # But document should still be persisted and retrievable
        doc = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc is not None, "Document should be persisted even with zero facts"
        assert doc["id"] == document_id
        assert doc["bank_id"] == bank_id
        assert doc["memory_unit_count"] == 0, "Should have zero memory units"
        assert len(doc["original_text"]) > 0, "Should have non-zero text length"
        assert "xyzabc123" in doc["original_text"], "Should contain original content"

        # Document should also appear in list
        docs_list = await memory.list_documents(
            bank_id=bank_id,
            search_query=None,
            limit=100,
            offset=0,
            request_context=request_context,
        )
        assert docs_list["total"] == 1, "Document should appear in list"
        assert any(d["id"] == document_id for d in docs_list["items"]), "Document should be in items"

        listed_doc = next(d for d in docs_list["items"] if d["id"] == document_id)
        assert listed_doc["memory_unit_count"] == 0, "Listed document should show zero memory units"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_document_persisted_with_zero_facts_batch(memory, request_context):
    """
    Test that documents are persisted with zero facts in batch retain operations.

    This tests the async batch code path to ensure it also handles zero facts correctly.
    """
    bank_id = f"test_zero_facts_batch_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Mix of content: some produces facts, some produces zero facts
        contents = [
            {
                "content": "Alice works at Google",
                "document_id": "doc-with-facts",
            },
            {
                "content": "!@# $$$ %%% ^^^ &&& ***",  # Gibberish - zero facts expected
                "document_id": "doc-zero-facts",
            },
        ]

        unit_ids = await memory.retain_batch_async(
            bank_id=bank_id,
            contents=contents,
            request_context=request_context,
        )

        # First content should produce facts, second should not
        assert len(unit_ids[0]) > 0, "First content should produce facts"
        assert len(unit_ids[1]) == 0, "Second content should produce zero facts"

        # Both documents should be persisted
        doc_with_facts = await memory.get_document("doc-with-facts", bank_id, request_context=request_context)
        assert doc_with_facts is not None
        assert doc_with_facts["memory_unit_count"] > 0

        doc_zero_facts = await memory.get_document("doc-zero-facts", bank_id, request_context=request_context)
        assert doc_zero_facts is not None, "Document with zero facts should be persisted"
        assert doc_zero_facts["memory_unit_count"] == 0, "Should have zero memory units"
        assert "!@#" in doc_zero_facts["original_text"]

        # Both should appear in list
        docs_list = await memory.list_documents(
            bank_id=bank_id,
            search_query=None,
            limit=100,
            offset=0,
            request_context=request_context,
        )
        assert docs_list["total"] == 2, "Both documents should appear in list"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_document_persisted_with_zero_facts_async_submit(memory, request_context):
    """
    Test that documents are persisted with zero facts in fire-and-forget async retain.

    This tests the submit_async_retain (background task) code path to ensure it also
    handles zero facts correctly.
    """
    import asyncio

    bank_id = f"test_zero_facts_async_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Submit async retain with gibberish content
        result = await memory.submit_async_retain(
            bank_id=bank_id,
            contents=[
                {
                    "content": "!@# $$$ %%% ^^^ &&& ***",  # Gibberish - zero facts expected
                    "document_id": "doc-async-zero-facts",
                }
            ],
            request_context=request_context,
        )

        operation_id = result["operation_id"]
        assert operation_id is not None, "Should return operation_id"

        # Wait for background task to complete
        max_wait = 60  # 60 seconds max
        wait_interval = 0.5
        elapsed = 0

        while elapsed < max_wait:
            await asyncio.sleep(wait_interval)
            elapsed += wait_interval

            # Check if document exists
            doc = await memory.get_document(
                "doc-async-zero-facts", bank_id, request_context=request_context
            )
            if doc is not None:
                break

        # Document should be persisted even with zero facts
        assert doc is not None, "Document should be persisted after async task completes"
        assert doc["id"] == "doc-async-zero-facts"
        assert doc["memory_unit_count"] == 0, "Should have zero memory units"
        assert "!@#" in doc["original_text"]

        # Document should appear in list
        docs_list = await memory.list_documents(
            bank_id=bank_id,
            search_query=None,
            limit=100,
            offset=0,
            request_context=request_context,
        )
        assert docs_list["total"] == 1, "Document should appear in list"
        assert any(d["id"] == "doc-async-zero-facts" for d in docs_list["items"])

        listed_doc = next(d for d in docs_list["items"] if d["id"] == "doc-async-zero-facts")
        assert listed_doc["memory_unit_count"] == 0, "Listed document should show zero memory units"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
