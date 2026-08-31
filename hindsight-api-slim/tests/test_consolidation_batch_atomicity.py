"""Every write from one consolidation LLM response commits or rolls back together (#3876).

Consolidation used to write each action the moment it was decided, on its own connection:
deletes first (to free observation slots), then updates, then creates, and the
``consolidated_at`` stamps for the source facts later still. A batch whose DELETE landed
and whose replacement CREATE then failed left the observation gone with nothing in its
place — and the sources stamped consolidated, which is the exclusion predicate for pending
consolidation, so nothing ever rebuilt it. The operation reported ``completed``.

The reporter of #3876 saw exactly that with a small self-hosted model whose consolidation
responses intermittently failed schema validation: batches logged success while the bank's
observations drained away.

These tests pin the contract:

1. A batch that fails partway through applying its actions leaves NOTHING behind — the
   delete it had already decided is rolled back with the rest.
2. A batch that fails is not recorded as consolidated, so its facts stay pending and the
   next round rebuilds what it could not write.
3. The happy path still commits both the observation writes and the stamps.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation import consolidator as consolidator_module
from hindsight_api.engine.consolidation.consolidator import (
    _ConsolidationBatchResponse,
    _CreateAction,
    _DedupOutcome,
    _DeleteAction,
    _UpdateAction,
    run_consolidation_job,
)
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.engine.response_models import MemoryFact
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult, ScoredResult


def test_retrieval_watermark_round_trips_through_public_memory_fact():
    """Every backend's shared retrieval model preserves the destructive CAS watermark."""
    watermark = datetime(2026, 9, 1, tzinfo=timezone.utc)
    retrieval = RetrievalResult.from_db_row(
        {"id": str(uuid.uuid4()), "text": "Observation", "fact_type": "observation", "updated_at": watermark}
    )
    serialized = ScoredResult(candidate=MergedCandidate(retrieval=retrieval, rrf_score=1.0)).to_dict()
    assert serialized["updated_at"] == watermark
    serialized["updated_at"] = serialized["updated_at"].isoformat()
    fact = MemoryFact(**serialized)

    assert fact.updated_at == watermark.isoformat()


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


def _override_config(memory: MemoryEngine, **overrides):
    raw = _get_raw_config()
    fake = type(raw)(**{**{f: getattr(raw, f) for f in raw.__dataclass_fields__}, **overrides})
    return patch.object(memory._config_resolver, "resolve_full_config", return_value=fake)


def _llm(callback):
    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")
    mock_llm.set_response_callback(callback)
    wrapper = MagicMock()
    wrapper.with_config.return_value = mock_llm
    return wrapper


def _create_one_per_fact(text: str | None = None):
    """Emit one CREATE per fact id found in the prompt."""

    def callback(messages, scope):
        if scope != "consolidation":
            return _ConsolidationBatchResponse()
        prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
        return _ConsolidationBatchResponse(
            creates=[
                _CreateAction(text=text or f"Observation about fact {fid[:8]}", source_fact_ids=[fid])
                for fid in fact_ids
            ]
        )

    return callback


async def _insert_memory(conn, bank_id: str, text: str, tags: list[str]) -> uuid.UUID:
    mem_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, tags, observation_scopes, created_at)
        VALUES ($1, $2, $3, 'experience', $4, $5::jsonb, now())
        """,
        mem_id,
        bank_id,
        text,
        tags,
        json.dumps(None),
    )
    return mem_id


async def _observations(memory: MemoryEngine, bank_id: str) -> list[str]:
    async with memory._pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT text FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation' ORDER BY text",
            bank_id,
        )
    return [r["text"] for r in rows]


async def _pending_facts(memory: MemoryEngine, bank_id: str) -> list[str]:
    async with memory._pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT text FROM memory_units WHERE bank_id = $1 AND fact_type = 'experience' "
            "AND consolidated_at IS NULL AND consolidation_failed_at IS NULL ORDER BY text",
            bank_id,
        )
    return [r["text"] for r in rows]


async def _run(memory: MemoryEngine, bank_id: str, request_context, callback, **config_overrides):
    original = memory._consolidation_llm_config
    memory._consolidation_llm_config = _llm(callback)
    try:
        with (
            _override_config(
                memory,
                consolidation_llm_batch_size=4,
                consolidation_llm_parallelism=1,
                **config_overrides,
            ),
            patch.object(memory, "submit_async_consolidation"),
        ):
            return await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)
    finally:
        memory._consolidation_llm_config = original


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_failed_create_rolls_back_the_delete_it_was_replacing(memory: MemoryEngine, request_context):
    """The classic #3876 shape: DELETE + CREATE in one response, the CREATE blows up.

    Before the fix the delete was already committed on its own connection and the
    observation was gone for good.
    """
    bank_id = f"atomic-del-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])

        await _run(memory, bank_id, request_context, _create_one_per_fact("Alice lives in Berlin"))
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]

        async with memory._pool.acquire() as conn:
            obs_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'", bank_id
            )
            await _insert_memory(conn, bank_id, "Alice moved to Munich", ["user:alice"])

        def delete_and_create(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            return _ConsolidationBatchResponse(
                creates=[_CreateAction(text="Alice lives in Munich", source_fact_ids=fact_ids)],
                deletes=[_DeleteAction(observation_id=str(obs_id))],
            )

        with patch.object(
            consolidator_module,
            "_apply_create_observation",
            new=AsyncMock(side_effect=RuntimeError("write failed mid-batch")),
        ):
            with pytest.raises(RuntimeError, match="write failed mid-batch"):
                await _run(memory, bank_id, request_context, delete_and_create)

        # The delete rolled back with the failed create: the observation is still there.
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
        # And the fact that batch consumed is still pending, so the next round retries it.
        assert await _pending_facts(memory, bank_id) == ["Alice moved to Munich"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_failed_batch_does_not_stamp_consolidated_at(memory: MemoryEngine, request_context):
    """``consolidated_at`` commits with the observations, never on its own.

    A stamp written for a batch whose writes were lost would exclude those facts from
    pending consolidation forever — the silent half of the data loss in #3876.
    """
    bank_id = f"atomic-stamp-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            for i in range(3):
                await _insert_memory(conn, bank_id, f"Alice fact {i}", ["user:alice"])

        with patch.object(
            consolidator_module,
            "_apply_create_observation",
            new=AsyncMock(side_effect=RuntimeError("write failed mid-batch")),
        ):
            with pytest.raises(RuntimeError, match="write failed mid-batch"):
                await _run(memory, bank_id, request_context, _create_one_per_fact())

        assert await _observations(memory, bank_id) == []
        assert await _pending_facts(memory, bank_id) == ["Alice fact 0", "Alice fact 1", "Alice fact 2"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_successful_batch_commits_observations_and_stamps(memory: MemoryEngine, request_context):
    """Guard on the rollback tests: the happy path still writes both halves."""
    bank_id = f"atomic-ok-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])

        result = await _run(memory, bank_id, request_context, _create_one_per_fact("Alice lives in Berlin"))

        assert result["status"] == "completed"
        assert result["observations_created"] == 1
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
        assert await _pending_facts(memory, bank_id) == []
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_rejected_replacement_source_keeps_observation_and_marks_batch_failed(
    memory: MemoryEngine, request_context
):
    """An out-of-batch replacement cannot accompany a destructive action."""
    bank_id = f"atomic-ref-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])

        await _run(memory, bank_id, request_context, _create_one_per_fact("Alice lives in Berlin"))
        async with memory._pool.acquire() as conn:
            obs_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'", bank_id
            )
            source_id = await _insert_memory(conn, bank_id, "Alice moved to Munich", ["user:alice"])

        foreign_id = str(uuid.uuid4())

        def invalid_replacement(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            return _ConsolidationBatchResponse(
                creates=[_CreateAction(text="Alice lives in Munich", source_fact_ids=[foreign_id])],
                deletes=[_DeleteAction(observation_id=str(obs_id))],
            )

        result = await _run(memory, bank_id, request_context, invalid_replacement)

        assert result["memories_failed"] == 1
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
        async with memory._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT consolidated_at, consolidation_failed_at FROM memory_units WHERE id = $1", source_id
            )
        assert row["consolidated_at"] is None
        assert row["consolidation_failed_at"] is not None
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_replacement_skipped_during_write_does_not_delete(memory: MemoryEngine, request_context):
    """A replacement that becomes inapplicable aborts its destructive response."""
    bank_id = f"atomic-skip-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])

        await _run(memory, bank_id, request_context, _create_one_per_fact("Alice lives in Berlin"))
        async with memory._pool.acquire() as conn:
            obs_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'", bank_id
            )
            await _insert_memory(conn, bank_id, "Alice moved to Munich", ["user:alice"])

        def delete_and_create(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            return _ConsolidationBatchResponse(
                creates=[_CreateAction(text="Alice lives in Munich", source_fact_ids=fact_ids)],
                deletes=[_DeleteAction(observation_id=str(obs_id))],
            )

        with patch.object(consolidator_module, "_apply_create_action", new=AsyncMock(return_value="skipped")):
            result = await _run(memory, bank_id, request_context, delete_and_create)

        assert result["memories_failed"] == 1
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_replacement_can_free_and_reuse_a_full_scope_slot(memory: MemoryEngine, request_context):
    """At capacity, a one-for-one DELETE+CREATE replacement still makes progress."""
    bank_id = f"atomic-cap-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])

        await _run(
            memory,
            bank_id,
            request_context,
            _create_one_per_fact("Alice lives in Berlin"),
            max_observations_per_scope=1,
        )
        async with memory._pool.acquire() as conn:
            obs_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'", bank_id
            )
            await _insert_memory(conn, bank_id, "Alice moved to Munich", ["user:alice"])

        def replacement(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            return _ConsolidationBatchResponse(
                creates=[_CreateAction(text="Alice lives in Munich", source_fact_ids=fact_ids)],
                deletes=[_DeleteAction(observation_id=str(obs_id))],
            )

        result = await _run(
            memory,
            bank_id,
            request_context,
            replacement,
            max_observations_per_scope=1,
        )

        assert result["memories_failed"] == 0
        assert result["observations_created"] == 1
        assert await _observations(memory, bank_id) == ["Alice lives in Munich"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_duplicate_delete_targets_cannot_mint_capacity(memory: MemoryEngine, request_context):
    """One row can fund at most one replacement CREATE."""
    bank_id = f"atomic-dupcap-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])
        await _run(
            memory,
            bank_id,
            request_context,
            _create_one_per_fact("Alice lives in Berlin"),
            max_observations_per_scope=1,
        )
        async with memory._pool.acquire() as conn:
            obs_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'", bank_id
            )
            await _insert_memory(conn, bank_id, "Alice moved again", ["user:alice"])

        def duplicate_deletes(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            return _ConsolidationBatchResponse(
                creates=[
                    _CreateAction(text="Replacement one", source_fact_ids=fact_ids),
                ],
                deletes=[
                    _DeleteAction(observation_id=str(obs_id)),
                    _DeleteAction(observation_id=str(obs_id)),
                ],
            )

        result = await _run(
            memory,
            bank_id,
            request_context,
            duplicate_deletes,
            max_observations_per_scope=1,
        )

        assert result["memories_failed"] == 1
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_destructive_capacity_overflow_rejects_whole_response(memory: MemoryEngine, request_context):
    """A replacement CREATE cannot be silently truncated and its source stamped."""
    bank_id = f"atomic-overcap-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])
        await _run(
            memory,
            bank_id,
            request_context,
            _create_one_per_fact("Alice lives in Berlin"),
            max_observations_per_scope=1,
        )
        async with memory._pool.acquire() as conn:
            obs_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'", bank_id
            )
            await _insert_memory(conn, bank_id, "Alice moved again", ["user:alice"])

        def overflowing_replacement(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            return _ConsolidationBatchResponse(
                creates=[
                    _CreateAction(text="Replacement one", source_fact_ids=fact_ids),
                    _CreateAction(text="Replacement two", source_fact_ids=fact_ids),
                ],
                deletes=[_DeleteAction(observation_id=str(obs_id))],
            )

        result = await _run(
            memory,
            bank_id,
            request_context,
            overflowing_replacement,
            max_observations_per_scope=1,
        )

        assert result["memories_failed"] == 1
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_destructive_batch_locks_dedup_target_in_global_order(memory: MemoryEngine, request_context):
    """A semantic fold target joins the same deterministic witness lock set."""
    bank_id = f"atomic-dedup-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice fact A", ["user:alice"])
        await _run(memory, bank_id, request_context, _create_one_per_fact("Observation A"))
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice fact B", ["user:alice"])
        await _run(memory, bank_id, request_context, _create_one_per_fact("Observation B"))
        async with memory._pool.acquire() as conn:
            delete_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND text = 'Observation A'", bank_id
            )
            target = await conn.fetchrow(
                "SELECT id, updated_at FROM memory_units WHERE bank_id = $1 AND text = 'Observation B'", bank_id
            )
            await _insert_memory(conn, bank_id, "Alice moved to Munich", ["user:alice"])

        def replacement(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            return _ConsolidationBatchResponse(
                creates=[_CreateAction(text="Replacement observation", source_fact_ids=fact_ids)],
                deletes=[_DeleteAction(observation_id=str(delete_id))],
            )

        outcome = _DedupOutcome(
            best_id=str(target["id"]),
            merged_text="Observation B enriched",
            should_merge=True,
            best_text="Observation B",
            best_updated_at=target["updated_at"],
        )
        lock_order: list[str] = []
        original_match = consolidator_module._observation_matches_witness

        async def record_match(conn, bank_id, observation_id, expected_text, expected_updated_at):
            lock_order.append(observation_id)
            return await original_match(conn, bank_id, observation_id, expected_text, expected_updated_at)

        with (
            patch.object(consolidator_module, "_dedup_active", return_value=True),
            patch.object(consolidator_module, "_dedup_adjudicate", new=AsyncMock(return_value=outcome)),
            patch.object(consolidator_module, "_observation_matches_witness", new=record_match),
        ):
            result = await _run(memory, bank_id, request_context, replacement)

        assert result["memories_failed"] == 0
        assert lock_order == sorted(lock_order)
        assert set(lock_order) == {str(delete_id), str(target["id"])}
        assert await _observations(memory, bank_id) == ["Observation B enriched"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.parametrize("store_owned", [False, True])
@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_dedup_fold_without_explicit_delete_uses_atomic_contract(
    memory: MemoryEngine, request_context, store_owned: bool
):
    """A mutating semantic fold is CAS-locked, or fail-closed for an external store."""
    bank_id = f"atomic-dedupcas-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    store_patch = None
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])
        await _run(memory, bank_id, request_context, _create_one_per_fact("Alice lives in Berlin"))
        async with memory._pool.acquire() as conn:
            target = await conn.fetchrow(
                "SELECT id, updated_at FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'", bank_id
            )
            await _insert_memory(conn, bank_id, "Alice confirmed the move", ["user:alice"])

        if store_owned:
            store_patch = patch.object(consolidator_module.get_memories(), "store_owned_for", return_value=True)

        def create_only(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            if store_patch is not None:
                store_patch.start()
            return _ConsolidationBatchResponse(
                creates=[_CreateAction(text="Alice remains in Berlin", source_fact_ids=fact_ids)]
            )

        outcome = _DedupOutcome(
            best_id=str(target["id"]),
            merged_text="Alice lives in Berlin",
            should_merge=True,
            best_text="Alice lives in Berlin",
            best_updated_at=target["updated_at"],
        )
        lock_order: list[str] = []
        original_match = consolidator_module._observation_matches_witness

        async def record_match(conn, bank_id, observation_id, expected_text, expected_updated_at):
            lock_order.append(observation_id)
            return await original_match(conn, bank_id, observation_id, expected_text, expected_updated_at)

        try:
            with (
                patch.object(consolidator_module, "_dedup_active", return_value=True),
                patch.object(consolidator_module, "_dedup_adjudicate", new=AsyncMock(return_value=outcome)),
                patch.object(consolidator_module, "_observation_matches_witness", new=record_match),
            ):
                result = await _run(memory, bank_id, request_context, create_only)
        finally:
            if store_patch is not None:
                store_patch.stop()

        if store_owned:
            assert result["memories_failed"] == 1
            assert lock_order == []
        else:
            assert result["memories_failed"] == 0
            assert lock_order == [str(target["id"])]
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_exact_duplicate_create_remains_successful_noop(memory: MemoryEngine, request_context):
    """The intentional exact-duplicate drop still consumes a valid source fact."""
    bank_id = f"atomic-dup-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])

        await _run(memory, bank_id, request_context, _create_one_per_fact("Alice lives in Berlin"))
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice confirmed the move", ["user:alice"])

        result = await _run(memory, bank_id, request_context, _create_one_per_fact("Alice lives in Berlin"))

        assert result["memories_failed"] == 0
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
        assert await _pending_facts(memory, bank_id) == []
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_delete_plus_exact_duplicate_create_keeps_existing_observation(memory: MemoryEngine, request_context):
    """A verbatim replacement cancels its paired delete instead of needing a free slot."""
    bank_id = f"atomic-dupdel-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])

        await _run(memory, bank_id, request_context, _create_one_per_fact("Alice lives in Berlin"))
        async with memory._pool.acquire() as conn:
            obs_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'", bank_id
            )
            await _insert_memory(conn, bank_id, "Alice confirmed the move", ["user:alice"])

        def duplicate_replacement(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            return _ConsolidationBatchResponse(
                creates=[_CreateAction(text="Alice lives in Berlin", source_fact_ids=fact_ids)],
                deletes=[_DeleteAction(observation_id=str(obs_id))],
            )

        result = await _run(memory, bank_id, request_context, duplicate_replacement)

        assert result["memories_failed"] == 0
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
        assert await _pending_facts(memory, bank_id) == []
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_duplicate_delete_cancellation_keeps_batch_atomic(memory: MemoryEngine, request_context):
    """Folding one delete/create pair cannot weaken another replacement action."""
    bank_id = f"atomic-dupatomic-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])
        await _run(memory, bank_id, request_context, _create_one_per_fact("Alice lives in Berlin"))
        async with memory._pool.acquire() as conn:
            obs_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'", bank_id
            )
            await _insert_memory(conn, bank_id, "Alice confirmed the move", ["user:alice"])

        def duplicate_and_novel(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            return _ConsolidationBatchResponse(
                creates=[
                    _CreateAction(text="Alice lives in Berlin", source_fact_ids=fact_ids),
                    _CreateAction(text="Alice also works remotely", source_fact_ids=fact_ids),
                ],
                deletes=[_DeleteAction(observation_id=str(obs_id))],
            )

        with patch.object(consolidator_module, "_apply_create_action", new=AsyncMock(return_value="skipped")):
            result = await _run(memory, bank_id, request_context, duplicate_and_novel)

        assert result["memories_failed"] == 1
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.parametrize("mutation", ["rewrite", "delete"])
@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_duplicate_noop_revalidates_target_before_stamping(memory: MemoryEngine, request_context, mutation: str):
    """A stale duplicate target cannot turn the paired CREATE into a false success."""
    bank_id = f"atomic-duprace-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])

        await _run(memory, bank_id, request_context, _create_one_per_fact("Alice lives in Berlin"))
        async with memory._pool.acquire() as conn:
            obs_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'", bank_id
            )
            source_id = await _insert_memory(conn, bank_id, "Alice confirmed the move", ["user:alice"])

        def duplicate_replacement(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            return _ConsolidationBatchResponse(
                creates=[_CreateAction(text="Alice lives in Berlin", source_fact_ids=fact_ids)],
                deletes=[_DeleteAction(observation_id=str(obs_id))],
            )

        original_match = consolidator_module._observation_matches_witness
        mutated = False

        async def mutate_then_match(conn, bank_id, observation_id, expected_text, expected_updated_at):
            nonlocal mutated
            if not mutated:
                mutated = True
                async with memory._pool.acquire() as other_conn:
                    if mutation == "rewrite":
                        await other_conn.execute(
                            "UPDATE memory_units SET context = $1, updated_at = now() WHERE id = $2",
                            "concurrently enriched",
                            obs_id,
                        )
                    else:
                        await other_conn.execute("DELETE FROM memory_units WHERE id = $1", obs_id)
            return await original_match(conn, bank_id, observation_id, expected_text, expected_updated_at)

        with patch.object(consolidator_module, "_observation_matches_witness", new=mutate_then_match):
            result = await _run(memory, bank_id, request_context, duplicate_replacement)

        assert result["memories_failed"] == 1
        async with memory._pool.acquire() as conn:
            source = await conn.fetchrow(
                "SELECT consolidated_at, consolidation_failed_at FROM memory_units WHERE id = $1", source_id
            )
        assert source["consolidated_at"] is None
        assert source["consolidation_failed_at"] is not None
        if mutation == "rewrite":
            assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
        else:
            assert await _observations(memory, bank_id) == []
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_non_destructive_invalid_create_keeps_skip_semantics(memory: MemoryEngine, request_context):
    """Hard rejection is limited to responses that can destroy observations."""
    bank_id = f"atomic-nondel-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])

        def invalid_create(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            return _ConsolidationBatchResponse(
                creates=[_CreateAction(text="Unusable", source_fact_ids=[str(uuid.uuid4())])]
            )

        result = await _run(memory, bank_id, request_context, invalid_create)

        assert result["memories_failed"] == 0
        assert await _observations(memory, bank_id) == []
        assert await _pending_facts(memory, bank_id) == []
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_update_and_delete_same_observation_is_rejected(memory: MemoryEngine, request_context):
    """A successful update must not be erased by a later delete in the same response."""
    bank_id = f"atomic-overlap-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])

        await _run(memory, bank_id, request_context, _create_one_per_fact("Alice lives in Berlin"))
        async with memory._pool.acquire() as conn:
            obs_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'", bank_id
            )
            await _insert_memory(conn, bank_id, "Alice moved to Munich", ["user:alice"])

        def update_and_delete(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            return _ConsolidationBatchResponse(
                updates=[
                    _UpdateAction(text="Alice lives in Munich", observation_id=str(obs_id), source_fact_ids=fact_ids)
                ],
                deletes=[_DeleteAction(observation_id=str(obs_id))],
            )

        result = await _run(memory, bank_id, request_context, update_and_delete)

        assert result["memories_failed"] == 1
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_changed_delete_target_is_not_removed(memory: MemoryEngine, request_context):
    """A concurrent rewrite invalidates the recalled-state witness for DELETE."""
    bank_id = f"atomic-race-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])

        await _run(memory, bank_id, request_context, _create_one_per_fact("Alice lives in Berlin"))
        async with memory._pool.acquire() as conn:
            obs_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'", bank_id
            )
            await _insert_memory(conn, bank_id, "Alice moved to Munich", ["user:alice"])

        def delete_and_create(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            return _ConsolidationBatchResponse(
                creates=[_CreateAction(text="Alice lives in Munich", source_fact_ids=fact_ids)],
                deletes=[_DeleteAction(observation_id=str(obs_id))],
            )

        original_match = consolidator_module._observation_matches_witness
        mutated = False

        async def concurrent_rewrite(conn, bank_id, observation_id, expected_text, expected_updated_at):
            nonlocal mutated
            if not mutated:
                mutated = True
                async with memory._pool.acquire() as other_conn:
                    await other_conn.execute(
                        "UPDATE memory_units SET context = $1, updated_at = now() WHERE id = $2",
                        "concurrently enriched",
                        obs_id,
                    )
            return await original_match(conn, bank_id, observation_id, expected_text, expected_updated_at)

        with patch.object(consolidator_module, "_observation_matches_witness", new=concurrent_rewrite):
            result = await _run(memory, bank_id, request_context, delete_and_create)

        assert result["memories_failed"] == 1
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
        async with memory._pool.acquire() as conn:
            assert (
                await conn.fetchval("SELECT context FROM memory_units WHERE id = $1", obs_id) == "concurrently enriched"
            )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_changed_update_target_aborts_destructive_batch(memory: MemoryEngine, request_context):
    """A stale UPDATE replacement cannot overwrite a concurrent rewrite before DELETE."""
    bank_id = f"atomic-updrace-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice fact A", ["user:alice"])
        await _run(memory, bank_id, request_context, _create_one_per_fact("Observation A"))

        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice fact B", ["user:alice"])
        await _run(memory, bank_id, request_context, _create_one_per_fact("Observation B"))

        async with memory._pool.acquire() as conn:
            update_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND text = 'Observation A'", bank_id
            )
            delete_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND text = 'Observation B'", bank_id
            )
            await _insert_memory(conn, bank_id, "Alice replacement fact", ["user:alice"])

        def update_and_delete(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            return _ConsolidationBatchResponse(
                updates=[
                    _UpdateAction(
                        text="Replacement observation",
                        observation_id=str(update_id),
                        source_fact_ids=fact_ids,
                    )
                ],
                deletes=[_DeleteAction(observation_id=str(delete_id))],
            )

        original_match = consolidator_module._observation_matches_witness
        mutated = False
        lock_order: list[str] = []

        async def mutate_update_target(conn, bank_id, observation_id, expected_text, expected_updated_at):
            nonlocal mutated
            lock_order.append(observation_id)
            if observation_id == str(update_id) and not mutated:
                mutated = True
                async with memory._pool.acquire() as other_conn:
                    await other_conn.execute(
                        "UPDATE memory_units SET context = $1, updated_at = now() WHERE id = $2",
                        "concurrently enriched",
                        update_id,
                    )
            return await original_match(conn, bank_id, observation_id, expected_text, expected_updated_at)

        with patch.object(consolidator_module, "_observation_matches_witness", new=mutate_update_target):
            result = await _run(memory, bank_id, request_context, update_and_delete)

        assert result["memories_failed"] == 1
        assert lock_order == sorted(lock_order)
        assert await _observations(memory, bank_id) == ["Observation A", "Observation B"]
        async with memory._pool.acquire() as conn:
            assert await conn.fetchval("SELECT context FROM memory_units WHERE id = $1", update_id) == (
                "concurrently enriched"
            )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_delete_only_batch_fails_closed_for_store_owned_bank(memory: MemoryEngine, request_context):
    """External deletes are not attempted without a version-checked CAS primitive."""
    bank_id = f"atomic-store-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])

        await _run(memory, bank_id, request_context, _create_one_per_fact("Alice lives in Berlin"))
        async with memory._pool.acquire() as conn:
            obs_id = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'", bank_id
            )
            await _insert_memory(conn, bank_id, "Alice moved to Munich", ["user:alice"])

        store = consolidator_module.get_memories()
        store_owned_patch = patch.object(store, "store_owned_for", return_value=True)

        def delete_only(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            store_owned_patch.start()
            return _ConsolidationBatchResponse(deletes=[_DeleteAction(observation_id=str(obs_id))])

        try:
            result = await _run(memory, bank_id, request_context, delete_only)
        finally:
            store_owned_patch.stop()

        assert result["memories_failed"] == 1
        assert await _observations(memory, bank_id) == ["Alice lives in Berlin"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
