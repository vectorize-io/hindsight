"""Regression tests for consolidation of a new tagged observation scope."""

import types
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import ANY, AsyncMock, patch

from hindsight_api.engine.consolidation import consolidator as C
from hindsight_api.engine.memory_engine import MemoryEngine


@asynccontextmanager
async def _acquire_connection(_pool: object) -> AsyncIterator[object]:
    yield object()


async def test_new_tag_scope_skips_filtered_observation_recall() -> None:
    """A zero-observation scope must reach CREATE without running ANN recall."""
    memory_id = str(uuid.uuid4())
    memories = [{"id": memory_id, "text": "The user prefers basil.", "tags": ["user:alice"]}]
    create_action = C._CreateAction(text="The user prefers basil.", source_fact_ids=[memory_id])
    llm_result = C._BatchLLMResult(creates=[create_action])
    scope_probe = AsyncMock(return_value=False)
    recall = AsyncMock()
    create = AsyncMock(return_value="created")

    with (
        patch.object(C, "acquire_with_retry", _acquire_connection),
        patch.object(C, "_has_observations_for_scope", scope_probe),
        patch.object(C, "_find_related_observations", recall),
        patch.object(C, "_consolidate_batch_with_llm", AsyncMock(return_value=llm_result)),
        patch.object(C, "_effective_scope_limit", return_value=-1),
        patch.object(C, "_dedup_active", return_value=False),
        patch.object(C, "_execute_create_action", create),
    ):
        result = await C._process_memory_batch(
            pool=object(),
            memory_engine=types.SimpleNamespace(),
            llm_config=object(),
            bank_id="bank-1",
            memories=memories,
            request_context=object(),
            config=object(),
        )

    scope_probe.assert_awaited_once_with(ANY, "bank-1", ["user:alice"])
    recall.assert_not_awaited()
    create.assert_awaited_once()
    assert create.await_args.kwargs["source_fact_tags"] == ["user:alice"]
    assert result == ([{"action": "created"}], 0, False)
    assert create.await_args.kwargs["source_memory_ids"] == [memory_id]


async def test_existing_tag_scope_preserves_strict_observation_recall() -> None:
    """The guard must not change recall for a scope that already has observations."""
    memory_id = str(uuid.uuid4())
    memories = [{"id": memory_id, "text": "The user prefers basil.", "tags": ["user:alice"]}]
    create_action = C._CreateAction(text="The user prefers basil.", source_fact_ids=[memory_id])
    llm_result = C._BatchLLMResult(creates=[create_action])
    scope_probe = AsyncMock(return_value=True)
    recall_result = types.SimpleNamespace(results=[], source_facts={})
    recall = AsyncMock(return_value=recall_result)

    with (
        patch.object(C, "acquire_with_retry", _acquire_connection),
        patch.object(C, "_has_observations_for_scope", scope_probe),
        patch.object(C, "_find_related_observations", recall),
        patch.object(C, "_consolidate_batch_with_llm", AsyncMock(return_value=llm_result)),
        patch.object(C, "_effective_scope_limit", return_value=-1),
        patch.object(C, "_dedup_active", return_value=False),
        patch.object(C, "_execute_create_action", AsyncMock(return_value="created")),
    ):
        await C._process_memory_batch(
            pool=object(),
            memory_engine=types.SimpleNamespace(),
            llm_config=object(),
            bank_id="bank-1",
            memories=memories,
            request_context=object(),
            config=object(),
        )

    scope_probe.assert_awaited_once()
    recall.assert_awaited_once()
    assert recall.await_args.kwargs["tags"] == ["user:alice"]


async def test_scope_probe_uses_bounded_strict_store_scan() -> None:
    """External memory stores receive the same all-strict scope with a one-row limit."""
    store = types.SimpleNamespace(
        writes_memory_rows_in_sql=False,
        scan_memories=AsyncMock(return_value=types.SimpleNamespace(memories=[])),
    )

    with patch.object(C, "get_memories", return_value=store):
        found = await C._has_observations_for_scope(object(), "bank-1", ["user:alice"])

    assert found is False
    store.scan_memories.assert_awaited_once()
    assert store.scan_memories.await_args.kwargs["fact_types"] == ["observation"]
    assert store.scan_memories.await_args.kwargs["tags"] == ["user:alice"]
    assert store.scan_memories.await_args.kwargs["tags_match"] == "all_strict"
    assert store.scan_memories.await_args.kwargs["limit"] == 1


async def test_scope_probe_uses_sql_existence_query() -> None:
    """The default store probes existence instead of counting or sorting vectors."""
    conn = types.SimpleNamespace(fetchval=AsyncMock(return_value=1))
    store = types.SimpleNamespace(writes_memory_rows_in_sql=True)

    with patch.object(C, "get_memories", return_value=store):
        found = await C._has_observations_for_scope(conn, "bank-1", ["user:alice"])

    assert found is True
    query, bank_id, tags = conn.fetchval.await_args.args
    assert "SELECT 1" in query
    assert "fact_type = 'observation'" in query
    assert "tags @> $2::varchar[]" in query
    assert "LIMIT 1" in query
    assert bank_id == "bank-1"
    assert tags == ["user:alice"]


async def test_scope_probe_executes_against_postgres(memory: MemoryEngine, request_context) -> None:
    """The existence query returns quickly and keeps strict containment semantics in PostgreSQL."""
    bank_id = f"test-empty-scope-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    try:
        async with memory._pool.acquire() as conn:
            assert await C._has_observations_for_scope(conn, bank_id, ["user:alice"]) is False
            await conn.execute(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, tags, proof_count)
                VALUES ($1, $2, $3, 'observation', $4, 1)
                """,
                uuid.uuid4(),
                bank_id,
                "Alice prefers basil.",
                ["user:alice", "project:garden"],
            )
            assert await C._has_observations_for_scope(conn, bank_id, ["user:alice"]) is True
            assert await C._has_observations_for_scope(conn, bank_id, ["user:bob"]) is False
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
