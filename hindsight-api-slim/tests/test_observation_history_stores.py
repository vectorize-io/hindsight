"""Observation history resolves memory data through either memories store."""

import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.consolidation.consolidator import _append_observation_history, _ObservationHistorySnapshot
from hindsight_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hindsight_api.engine.memories import get_memories, set_memories
from hindsight_api.engine.memories.base import StoredMemory
from hindsight_api.engine.memory_engine import MemoryEngine
from tests.test_memories_extension import InMemoryMemories
from tests.test_retain_same_document_concurrency import _StubEmbeddings


@pytest.fixture(scope="session")
def embeddings() -> _StubEmbeddings:
    return _StubEmbeddings()


@pytest.fixture(scope="session")
def cross_encoder() -> RRFPassthroughCrossEncoder:
    return RRFPassthroughCrossEncoder()


@pytest.fixture
def restore_store() -> Iterator[None]:
    original = get_memories()
    yield
    set_memories(original)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
@pytest.mark.parametrize("store_owned", [False, True], ids=["postgres", "external"])
async def test_history_resolves_sources_from_the_memory_store(
    memory: MemoryEngine, request_context: RequestContext, restore_store: None, store_owned: bool
) -> None:
    bank_id = f"test-history-store-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    original = get_memories()
    store = InMemoryMemories() if store_owned else original
    first_id, second_id, deleted_id, obs_id = [str(uuid.uuid4()) for _ in range(4)]
    rows = [
        StoredMemory(unit_id=first_id, text="first source", fact_type="world", context="first context"),
        StoredMemory(unit_id=second_id, text="second source", fact_type="experience", context=""),
        StoredMemory(
            unit_id=obs_id,
            text="current observation",
            fact_type="observation",
            source_memory_ids=[first_id, second_id, deleted_id],
        ),
    ]
    try:
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            if isinstance(store, InMemoryMemories):
                store.rows.update({row.unit_id: row for row in rows})
                assert await conn.fetchval("SELECT count(*) FROM memory_units WHERE bank_id = $1", bank_id) == 0
            else:
                for row in rows:
                    await conn.execute(
                        """INSERT INTO memory_units
                           (id, bank_id, text, fact_type, context, event_date, source_memory_ids)
                           VALUES ($1, $2, $3, $4, $5, $6, $7::uuid[])""",
                        uuid.UUID(row.unit_id),
                        bank_id,
                        row.text,
                        row.fact_type,
                        row.context,
                        datetime.now(timezone.utc),
                        [uuid.UUID(sid) for sid in row.source_memory_ids],
                    )
        set_memories(store)
        assert await memory.get_observation_history(bank_id, obs_id, request_context) == []
        assert await memory.get_observation_history(bank_id, first_id, request_context) == []
        assert await memory.get_observation_history(bank_id, str(uuid.uuid4()), request_context) is None

        async with pool.acquire() as conn:
            for previous_text, new_ids in [("before first", [first_id]), ("before second", [second_id, deleted_id])]:
                await _append_observation_history(
                    conn,
                    bank_id,
                    obs_id,
                    _ObservationHistorySnapshot(
                        previous_text=previous_text,
                        previous_tags=["tag"],
                        previous_occurred_start=None,
                        previous_occurred_end=None,
                        previous_mentioned_at=None,
                        new_source_memory_ids=new_ids,
                    ),
                    5,
                )
        history = await memory.get_observation_history(bank_id, obs_id, request_context)
        assert history is not None
        assert [entry["previous_text"] for entry in history] == ["before first", "before second"]
        assert [fact["id"] for fact in history[0]["source_facts"]] == [first_id]
        facts = history[1]["source_facts"]
        assert facts == [
            {"id": first_id, "text": "first source", "type": "world", "context": "first context", "is_new": False},
            {"id": second_id, "text": "second source", "type": "experience", "context": None, "is_new": True},
            {"id": deleted_id, "text": None, "type": None, "context": None, "is_new": True},
        ]
        if isinstance(store, InMemoryMemories):
            assert "get_memories" in store.calls
    finally:
        set_memories(original)
        await memory.delete_bank(bank_id, request_context=request_context)
