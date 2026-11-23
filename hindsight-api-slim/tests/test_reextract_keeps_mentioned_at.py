"""#6783: re-extracting a document must not reset every fact's mentioned_at to now.

A full re-extraction deletes the document's facts and extracts them again. Content the
caller sent with no timestamp is dated "now", so a reprocess made facts first stated
months ago look as recent as the latest correction. A chunk whose text was already
stored is now extracted under the date it was first retained on; new text keeps "now".
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api.engine.schema import fq_store_table

_OLD = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
_FIRST = "Alice works at Google. Bob works at Meta."
_SECOND = "Carol moved to Berlin and now works at SAP."


def _is_old(d: datetime) -> bool:
    # Extraction spaces a chunk's facts a few milliseconds apart to keep their order.
    return abs(d - _OLD) < timedelta(seconds=1)


@pytest.fixture(scope="session")
def db_url():
    """Isolated pg0 instance so this module never touches the dev database."""
    return "pg0://hs6783:55783"


async def _backdate(memory, bank_id: str, document_id: str) -> None:
    async with memory._pool.acquire() as conn:
        await conn.execute(
            f"UPDATE {fq_store_table('memory_units')} SET mentioned_at = $1 WHERE bank_id = $2 AND document_id = $3",
            _OLD,
            bank_id,
            document_id,
        )


async def _dates_by_text(memory, bank_id: str, document_id: str, request_context) -> dict[str, datetime]:
    result = await memory.list_memory_units(
        bank_id, document_id=document_id, limit=100, request_context=request_context
    )
    return {u["text"]: datetime.fromisoformat(u["mentioned_at"]) for u in result["items"]}


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_reprocess_keeps_first_retained_date(memory, request_context):
    bank_id = f"test_6783_reprocess_{datetime.now(UTC).timestamp()}"
    try:
        await memory.retain_batch_async(
            bank_id=bank_id, contents=[{"content": _FIRST, "document_id": "doc"}], request_context=request_context
        )
        await _backdate(memory, bank_id, "doc")

        result = await memory.reprocess_document(bank_id=bank_id, document_id="doc", request_context=request_context)
        assert result is not None
        await asyncio.sleep(0.5)

        dates = await _dates_by_text(memory, bank_id, "doc", request_context)
        assert dates
        assert all(_is_old(d) for d in dates.values()), dates
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_full_reextract_dates_new_text_now(memory, request_context):
    """Unchanged text keeps its old date; text the document did not have before is dated now."""
    bank_id = f"test_6783_new_text_{datetime.now(UTC).timestamp()}"
    try:
        await memory.retain_batch_async(
            bank_id=bank_id, contents=[{"content": _FIRST, "document_id": "doc"}], request_context=request_context
        )
        await _backdate(memory, bank_id, "doc")

        before = datetime.now(UTC) - timedelta(minutes=1)
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": _FIRST, "document_id": "doc", "force_reextract": True},
                {"content": _SECOND, "document_id": "doc", "force_reextract": True},
            ],
            request_context=request_context,
        )

        dates = await _dates_by_text(memory, bank_id, "doc", request_context)
        old = [d for t, d in dates.items() if "Alice" in t or "Bob" in t]
        new = [d for t, d in dates.items() if "Carol" in t]
        assert old and new, dates
        assert all(_is_old(d) for d in old), dates
        assert all(d > before for d in new), dates
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
