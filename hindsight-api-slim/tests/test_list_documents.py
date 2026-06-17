"""
Tests for list_documents pagination and tags filtering.
"""

from datetime import datetime, timezone

import pytest


async def _retain_doc(memory, bank_id, document_id, tags, request_context):
    """Helper to retain a document with given tags. Uses gibberish content to avoid LLM
    fact extraction (documents are persisted even with zero facts)."""
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": f"xyzabc123 !@# $$$ {document_id}"}],
        document_id=document_id,
        document_tags=tags or None,
        request_context=request_context,
    )


@pytest.mark.asyncio
async def test_list_documents_offset_pagination(memory, request_context):
    """offset parameter returns the correct slice of documents."""
    bank_id = f"test_list_docs_offset_{datetime.now(timezone.utc).timestamp()}"

    try:
        for i in range(4):
            await _retain_doc(memory, bank_id, f"doc-{i:02d}", [], request_context)

        # All documents, ordered by created_at DESC → doc-03, doc-02, doc-01, doc-00
        all_docs = await memory.list_documents(bank_id=bank_id, limit=10, offset=0, request_context=request_context)
        assert all_docs["total"] == 4
        assert len(all_docs["items"]) == 4
        all_ids = [d["id"] for d in all_docs["items"]]

        # offset=2 should skip the first two and return the remaining two
        page2 = await memory.list_documents(bank_id=bank_id, limit=10, offset=2, request_context=request_context)
        assert page2["total"] == 4  # total is always the full count
        assert len(page2["items"]) == 2
        assert [d["id"] for d in page2["items"]] == all_ids[2:]

        # offset beyond total returns empty items but correct total
        beyond = await memory.list_documents(bank_id=bank_id, limit=10, offset=10, request_context=request_context)
        assert beyond["total"] == 4
        assert beyond["items"] == []

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_list_documents_tags_filter_any_strict(memory, request_context):
    """tags filter with any_strict returns only tagged documents that match."""
    bank_id = f"test_list_docs_tags_{datetime.now(timezone.utc).timestamp()}"

    try:
        await _retain_doc(memory, bank_id, "doc-alpha", ["team-a"], request_context)
        await _retain_doc(memory, bank_id, "doc-beta", ["team-b"], request_context)
        await _retain_doc(memory, bank_id, "doc-both", ["team-a", "team-b"], request_context)
        await _retain_doc(memory, bank_id, "doc-untagged", [], request_context)

        # any_strict: only docs with at least one of the given tags, untagged excluded
        result = await memory.list_documents(
            bank_id=bank_id,
            tags=["team-a"],
            tags_match="any_strict",
            request_context=request_context,
        )
        ids = {d["id"] for d in result["items"]}
        assert ids == {"doc-alpha", "doc-both"}
        assert result["total"] == 2

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_list_documents_tags_filter_any_includes_untagged(memory, request_context):
    """tags filter with 'any' mode includes untagged documents."""
    bank_id = f"test_list_docs_tags_any_{datetime.now(timezone.utc).timestamp()}"

    try:
        await _retain_doc(memory, bank_id, "doc-tagged", ["team-a"], request_context)
        await _retain_doc(memory, bank_id, "doc-other", ["team-b"], request_context)
        await _retain_doc(memory, bank_id, "doc-untagged", [], request_context)

        result = await memory.list_documents(
            bank_id=bank_id,
            tags=["team-a"],
            tags_match="any",
            request_context=request_context,
        )
        ids = {d["id"] for d in result["items"]}
        # "any" includes untagged + matching tagged
        assert "doc-tagged" in ids
        assert "doc-untagged" in ids
        assert "doc-other" not in ids

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_list_documents_tags_filter_all_strict(memory, request_context):
    """tags filter with all_strict returns only docs that have ALL the specified tags."""
    bank_id = f"test_list_docs_tags_all_{datetime.now(timezone.utc).timestamp()}"

    try:
        await _retain_doc(memory, bank_id, "doc-a-only", ["team-a"], request_context)
        await _retain_doc(memory, bank_id, "doc-a-and-b", ["team-a", "team-b"], request_context)
        await _retain_doc(memory, bank_id, "doc-untagged", [], request_context)

        result = await memory.list_documents(
            bank_id=bank_id,
            tags=["team-a", "team-b"],
            tags_match="all_strict",
            request_context=request_context,
        )
        ids = {d["id"] for d in result["items"]}
        assert ids == {"doc-a-and-b"}

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_list_documents_no_tags_filter_returns_all(memory, request_context):
    """When no tags filter is specified, all documents are returned."""
    bank_id = f"test_list_docs_no_tags_{datetime.now(timezone.utc).timestamp()}"

    try:
        await _retain_doc(memory, bank_id, "doc-tagged", ["team-a"], request_context)
        await _retain_doc(memory, bank_id, "doc-untagged", [], request_context)

        result = await memory.list_documents(
            bank_id=bank_id,
            tags=None,
            request_context=request_context,
        )
        ids = {d["id"] for d in result["items"]}
        assert ids == {"doc-tagged", "doc-untagged"}

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_list_documents_tags_and_search_query_combined(memory, request_context):
    """tags filter and q (search_query) can be combined."""
    bank_id = f"test_list_docs_tags_q_{datetime.now(timezone.utc).timestamp()}"

    try:
        await _retain_doc(memory, bank_id, "report-2024", ["team-a"], request_context)
        await _retain_doc(memory, bank_id, "report-2025", ["team-b"], request_context)
        await _retain_doc(memory, bank_id, "summary-2024", ["team-a"], request_context)

        result = await memory.list_documents(
            bank_id=bank_id,
            search_query="report",
            tags=["team-a"],
            tags_match="any_strict",
            request_context=request_context,
        )
        ids = {d["id"] for d in result["items"]}
        assert ids == {"report-2024"}

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
