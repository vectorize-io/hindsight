"""Regression coverage for cross-bank chunk identifier collisions."""

import pytest


@pytest.mark.asyncio
async def test_retain_keeps_chunks_isolated_when_bank_and_document_boundaries_overlap(memory, request_context):
    first_bank = "a"
    first_document = "b_c"
    second_bank = "a_b"
    second_document = "c"

    try:
        await memory.retain_async(
            bank_id=first_bank,
            content="FIRST_BANK_CONTENT Alice works at Acme.",
            document_id=first_document,
            request_context=request_context,
        )
        await memory.retain_async(
            bank_id=second_bank,
            content="SECOND_BANK_CONTENT Bob works at Beta.",
            document_id=second_document,
            request_context=request_context,
        )

        first_chunks = await memory.list_document_chunks(
            first_bank,
            first_document,
            request_context=request_context,
        )
        second_chunks = await memory.list_document_chunks(
            second_bank,
            second_document,
            request_context=request_context,
        )

        assert first_chunks["total"] == 1
        assert second_chunks["total"] == 1
        assert first_chunks["items"][0]["chunk_text"].startswith("FIRST_BANK_CONTENT")
        assert second_chunks["items"][0]["chunk_text"].startswith("SECOND_BANK_CONTENT")
        assert first_chunks["items"][0]["chunk_id"] != second_chunks["items"][0]["chunk_id"]
    finally:
        await memory.delete_bank(first_bank, request_context=request_context)
        await memory.delete_bank(second_bank, request_context=request_context)
