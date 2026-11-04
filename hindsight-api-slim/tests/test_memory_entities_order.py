"""Opening one memory lists its entities in the same order as the memory list.

Neither read sorted a fact's entities, so Postgres returned them in whatever
order it found the rows and the two reads disagreed at random (#6271). Both now
sort them by name.
"""

import uuid
from datetime import UTC, datetime

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import embedding_processing

# Seeds `memory_units` directly — see test_list_memory_units_filters.py.
pytestmark = pytest.mark.memory_backend_incompatible

# Linked in reverse name order, so rows read back in insertion order are unsorted.
NAMES = ["Zoe", "Mallory", "Bob", "Alice"]


@pytest.mark.asyncio
async def test_detail_and_listing_return_entities_sorted_by_name(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test-entity-order-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        emb = (await embedding_processing.generate_embeddings_batch(memory.embeddings, ["fact"]))[0]
        unit_id = uuid.uuid4()
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, event_date, mentioned_at)
                VALUES ($1, $2, 'Zoe, Mallory, Bob and Alice met', 'world', $3::vector, $4, $4)
                """,
                unit_id,
                bank_id,
                str(emb),
                datetime.now(UTC),
            )
            for name in NAMES:
                entity_id = await conn.fetchval(
                    "INSERT INTO entities (bank_id, canonical_name) VALUES ($1, $2) RETURNING id", bank_id, name
                )
                await conn.execute("INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)", unit_id, entity_id)

        listing = await memory.list_memory_units(bank_id, limit=10, offset=0, request_context=request_context)
        detail = await memory.get_memory_unit(bank_id, str(unit_id), request_context=request_context)

        assert listing["items"][0]["entities"] == ", ".join(sorted(NAMES))
        assert detail["entities"] == sorted(NAMES)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
