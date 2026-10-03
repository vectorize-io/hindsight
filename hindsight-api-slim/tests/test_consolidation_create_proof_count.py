"""New observations count distinct surviving sources on every storage path."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.consolidation import consolidator as C
from hindsight_api.engine.memories.base import MemoriesExtension
from hindsight_api.engine.memories.pg import consolidation as pg_consolidation


def _ids():
    return uuid.uuid4(), uuid.uuid4(), uuid.uuid4()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_case", ["multiple", "duplicates", "deleted", "all_deleted", "single"])
async def test_consolidator_hands_the_store_only_surviving_sources(source_case):
    first, second, deleted = _ids()
    source_ids = {
        "multiple": [first, second],
        "duplicates": [first, second, first],
        "deleted": [first, deleted, second, deleted],
        "all_deleted": [deleted],
        "single": [first],
    }[source_case]
    live = [mid for mid in source_ids if mid != deleted]
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[{"id": mid} for mid in {first, second}])
    store = MagicMock()
    store.insert_observation = AsyncMock(return_value=uuid.uuid4())
    store.lock_live_memory_ids = AsyncMock(side_effect=lambda **kw: {str(u) for u in kw["unit_ids"] if u != deleted})
    engine = SimpleNamespace(_backend=SimpleNamespace(ops=SimpleNamespace(uses_observation_sources_table=False)))

    with patch.object(C, "get_memories", return_value=store):
        result = await C._apply_create_observation(
            conn, engine, "bank", source_ids, "A supported observation", "[0.1, 0.2]"
        )

    if not live:
        assert result == {"action": "skipped", "reason": "sources_deleted"}
        store.insert_observation.assert_not_awaited()
        return
    assert result["action"] == "created"
    assert store.insert_observation.await_args.kwargs["source_memory_ids"] == live


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vchord", "native", "pg_textsearch"])
@pytest.mark.parametrize("duplicates", [False, True])
async def test_postgres_insert_observation_proof_count(backend, duplicates):
    first, second, _ = _ids()
    source_ids = [first, second, first] if duplicates else [first, second]
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value={"id": uuid.uuid4()})
    conn.executemany = AsyncMock()
    config = SimpleNamespace(text_search_extension=backend, text_search_extension_native_language="english")

    with (
        patch.object(pg_consolidation, "get_config", return_value=config),
        patch("hindsight_api.engine.schema._is_oracle", return_value=False),
    ):
        await pg_consolidation.insert_observation(
            conn=conn,
            ops=SimpleNamespace(uses_observation_sources_table=False),
            fq_table=lambda name: name,
            bank_id="bank",
            observation_id=uuid.uuid4(),
            text="A supported observation",
            embedding="[0.1, 0.2]",
            source_memory_ids=source_ids,
            tags=[],
            event_date=None,
            occurred_start=None,
            occurred_end=None,
            mentioned_at=None,
        )

    query, *args = conn.fetchrow.await_args.args
    # Resolve the proof_count SQL value, not just the Python argument.
    values = query.split("VALUES (", 1)[1].split(")", 1)[0].split(",")
    proof_value = values[5].strip()
    actual = args[int(proof_value[1:]) - 1] if proof_value.startswith("$") else int(proof_value)
    assert actual == 2
    assert args[4] == source_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicates", [False, True])
async def test_default_store_insert_observation_proof_count(duplicates):
    first, second, _ = _ids()
    source_ids = [first, second, first] if duplicates else [first, second]
    store = MagicMock()
    store.upsert_observation = AsyncMock()

    await MemoriesExtension.insert_observation(
        store,
        conn=MagicMock(),
        ops=None,
        fq_table=lambda name: name,
        bank_id="bank",
        observation_id=uuid.uuid4(),
        text="A supported observation",
        embedding="[0.1, 0.2]",
        source_memory_ids=source_ids,
        tags=[],
        event_date=None,
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
        created_at=None,
    )

    record = store.upsert_observation.await_args.kwargs["record"]
    assert record.proof_count == 2
    assert record.source_memory_ids == [str(mid) for mid in source_ids]
