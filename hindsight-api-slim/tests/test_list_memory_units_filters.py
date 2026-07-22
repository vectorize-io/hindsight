"""Tests for the ``list_memory_units`` ingest-age filter: ``created_before``.

Lets maintenance-loop callers (retention sweeps, bulk maintenance) select units
by ingest age through the engine instead of hand-writing SQL against
``memory_units``. Filtering runs against a real Postgres via the ``memory``
fixture.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import embedding_processing


async def _ensure_bank(memory: MemoryEngine, bank_id: str, request_context: RequestContext) -> None:
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)


async def _seed_unit(
    conn,
    memory: MemoryEngine,
    bank_id: str,
    text: str,
    *,
    created_at: datetime,
) -> str:
    """Insert a live memory unit with an explicit ingest timestamp."""
    mem_id = uuid.uuid4()
    emb = await embedding_processing.generate_embeddings_batch(memory.embeddings, [text])
    await conn.execute(
        """
        INSERT INTO memory_units (
            id, bank_id, text, fact_type, embedding, event_date,
            created_at, updated_at, consolidated_at
        )
        VALUES ($1, $2, $3, 'experience', $4::vector, NOW(), $5, $5, $5)
        """,
        mem_id,
        bank_id,
        text,
        str(emb[0]),
        created_at,
    )
    return str(mem_id)


def _ids(result: dict) -> set[str]:
    return {item["id"] for item in result["items"]}


@pytest.mark.asyncio
async def test_created_before_filter(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test-lmu-created-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    now = datetime.now(UTC)
    old = now - timedelta(days=30)

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        old_id = await _seed_unit(conn, memory, bank_id, "old fact", created_at=old)
        _new_id = await _seed_unit(conn, memory, bank_id, "new fact", created_at=now)

    res = await memory.list_memory_units(
        bank_id, created_before=now - timedelta(days=1), request_context=request_context
    )
    assert _ids(res) == {old_id}
    assert res["total"] == 1

    await memory.delete_bank(bank_id, request_context=request_context)
