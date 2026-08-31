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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation import consolidator as consolidator_module
from hindsight_api.engine.consolidation.consolidator import (
    _ConsolidationBatchResponse,
    _CreateAction,
    _DeleteAction,
    run_consolidation_job,
)
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM


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


async def _run(memory: MemoryEngine, bank_id: str, request_context, callback):
    original = memory._consolidation_llm_config
    memory._consolidation_llm_config = _llm(callback)
    try:
        with (
            _override_config(memory, consolidation_llm_batch_size=4, consolidation_llm_parallelism=1),
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
