"""Offset pagination of the memory list must neither repeat nor skip a row.

The default order is ``mentioned_at DESC NULLS LAST, created_at DESC``. Facts
written in one batch share both timestamps, and Postgres may return tied rows in
a different order from one query to the next (a top-N sort under LIMIT is not
stable), so pages read at increasing offsets overlapped and skipped rows. ``id``
as the last sort key makes the order total.
"""

import uuid
from datetime import UTC, datetime

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import embedding_processing

# Seeds `memory_units` directly — see test_list_memory_units_filters.py.
pytestmark = pytest.mark.memory_backend_incompatible

ROWS = 60
PAGE = 7


@pytest.mark.asyncio
async def test_offset_pages_of_tied_rows_cover_the_list_exactly_once(
    memory: MemoryEngine, request_context: RequestContext
):
    bank_id = f"test-stable-order-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    at = datetime.now(UTC)
    try:
        emb = (await embedding_processing.generate_embeddings_batch(memory.embeddings, ["tied"]))[0]
        seeded = [uuid.uuid4() for _ in range(ROWS)]
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            # Every row ties on both timestamps, as one consolidation batch does.
            await conn.executemany(
                """
                INSERT INTO memory_units (
                    id, bank_id, text, fact_type, embedding, event_date,
                    mentioned_at, created_at, updated_at, consolidated_at
                )
                VALUES ($1, $2, $3, 'observation', $4::vector, $5, $5, $5, $5, $5)
                """,
                [(mid, bank_id, f"tied {i}", str(emb), at) for i, mid in enumerate(seeded)],
            )

        paged: list[str] = []
        for offset in range(0, ROWS, PAGE):
            page = await memory.list_memory_units(bank_id, limit=PAGE, offset=offset, request_context=request_context)
            assert page["total"] == ROWS
            paged.extend(item["id"] for item in page["items"])

        assert len(paged) == len(set(paged)), "a row appeared on two pages"
        assert sorted(paged) == sorted(str(m) for m in seeded), "a row appeared on no page"
        # Ties fall back to id, so the walk is the same every time.
        assert paged == sorted(paged)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
