"""
Tests for the bank-level `last_write_at` timestamp returned by list_banks.

`last_document_at` is document ingestion time, so appending to (or re-retaining)
a long-lived document leaves it frozen while the bank is still being written to.
`last_write_at` must track the actual write.
"""

from datetime import datetime, timezone

import pytest


async def _retain(memory, bank_id, document_id, content, request_context):
    """Retain content into a document. Gibberish avoids LLM fact extraction noise."""
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": content}],
        document_id=document_id,
        request_context=request_context,
    )


async def _bank_entry(memory, bank_id, request_context):
    page = await memory.list_banks(search_query=bank_id, request_context=request_context)
    return next(b for b in page["banks"] if b["bank_id"] == bank_id)


def _ts(value: str | None) -> datetime:
    assert value is not None
    return datetime.fromisoformat(value)


@pytest.mark.asyncio
async def test_last_write_at_advances_when_existing_document_is_rewritten(memory, request_context):
    """Re-retaining an existing document moves last_write_at but not last_document_at."""
    bank_id = f"test_last_write_{datetime.now(timezone.utc).timestamp()}"

    try:
        await _retain(memory, bank_id, "session-1", "xyzabc123 !@# first slice", request_context)
        after_first = await _bank_entry(memory, bank_id, request_context)
        # First ingestion: the document was created and written at the same time.
        assert _ts(after_first["last_write_at"]) >= _ts(after_first["last_document_at"])

        await _retain(memory, bank_id, "session-1", "xyzabc123 !@# second slice", request_context)
        after_second = await _bank_entry(memory, bank_id, request_context)

        # Ingestion time is frozen — no new document appeared.
        assert after_second["last_document_at"] == after_first["last_document_at"]
        # ...but the bank was written to, and last_write_at says so.
        assert _ts(after_second["last_write_at"]) > _ts(after_first["last_write_at"])
        assert _ts(after_second["last_write_at"]) > _ts(after_second["last_document_at"])

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_last_write_at_is_null_for_empty_bank(memory, request_context):
    """A bank with no documents and no facts has never been written to."""
    bank_id = f"test_last_write_empty_{datetime.now(timezone.utc).timestamp()}"

    try:
        await memory._ensure_bank_exists(bank_id, request_context)
        entry = await _bank_entry(memory, bank_id, request_context)
        assert entry["last_write_at"] is None
        assert entry["last_document_at"] is None

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_banks_are_ordered_by_last_write(memory, request_context):
    """The bank written to most recently sorts first, even if its document is older."""
    stamp = datetime.now(timezone.utc).timestamp()
    older_bank = f"test_last_write_order_a_{stamp}"
    newer_bank = f"test_last_write_order_b_{stamp}"

    try:
        await _retain(memory, older_bank, "doc-a", "xyzabc123 !@# alpha", request_context)
        await _retain(memory, newer_bank, "doc-b", "xyzabc123 !@# beta", request_context)
        # Rewrite the older bank's existing document: no new document, but it is now
        # the most recently written bank.
        await _retain(memory, older_bank, "doc-a", "xyzabc123 !@# alpha revised", request_context)

        page = await memory.list_banks(search_query="test_last_write_order_", request_context=request_context)
        ordered = [b["bank_id"] for b in page["banks"] if b["bank_id"] in (older_bank, newer_bank)]
        assert ordered == [older_bank, newer_bank]

    finally:
        await memory.delete_bank(older_bank, request_context=request_context)
        await memory.delete_bank(newer_bank, request_context=request_context)
