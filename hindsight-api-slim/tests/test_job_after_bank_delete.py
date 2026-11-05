"""A background job never recreates, or writes into, a bank deleted while it ran (#6344).

Before the fix a consolidation that finished after a delete got the bank created again (and, for
a store that owns its storage, its storage too) and wrote its old results into it.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.bank_life import BankDeletedError, bind_job_bank
from hindsight_api.engine.consolidation.consolidator import _CreateAction
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM
from tests.consolidation_answers import batch_fact_ids, fact_answer


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


async def _bank_exists(memory: MemoryEngine, bank_id: str) -> bool:
    async with memory._pool.acquire() as conn:
        return await conn.fetchval("SELECT 1 FROM banks WHERE bank_id = $1", bank_id) is not None


async def _unit_count(memory: MemoryEngine, bank_id: str) -> int:
    async with memory._pool.acquire() as conn:
        return await conn.fetchval("SELECT count(*) FROM memory_units WHERE bank_id = $1", bank_id)


def _create_llm():
    def callback(messages, scope):
        if scope != "consolidation":
            return fact_answer()
        prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        fact_ids = batch_fact_ids(prompt)
        return fact_answer(creates=[_CreateAction(text="Alice lives in Berlin", source_fact_ids=fact_ids)])

    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")
    mock_llm.set_response_callback(callback)
    wrapper = MagicMock()
    wrapper.with_config.return_value = mock_llm
    return wrapper


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
@pytest.mark.parametrize("recreate", [False, True], ids=["deleted", "deleted-and-recreated"])
async def test_consolidation_stops_when_its_bank_is_deleted_mid_run(
    memory: MemoryEngine, request_context, recreate: bool
):
    bank_id = f"del-mid-run-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    async with memory._pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, observation_scopes, created_at) "
            "VALUES ($1, $2, 'Alice moved to Berlin', 'experience', $3, $4::jsonb, now())",
            uuid.uuid4(),
            bank_id,
            ["user:alice"],
            json.dumps(None),
        )

    store = get_memories()
    original_begin = store.begin_write_batch
    # On Postgres the foreign keys already stop these writes; a store that owns its storage has
    # none, and recreates the bank on the write batch's commit. So that commit must not happen.
    committed: list[str] = []

    async def delete_then_begin(*, bank_id: str):
        # The LLM has answered; the delete lands right before the job writes what it decided. Run
        # in a fresh context, as an API request is: inside the job's, the recreate is refused.
        async def user_requests():
            await memory.delete_bank(bank_id, request_context=request_context)
            if recreate:
                await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

        await asyncio.get_running_loop().create_task(user_requests(), context=contextvars.Context())
        batch = await original_begin(bank_id=bank_id)
        original_commit = batch.commit

        async def commit():
            committed.append(bank_id)
            await original_commit()

        batch.commit = commit
        return batch

    original_llm = memory._consolidation_llm_config
    memory._consolidation_llm_config = _create_llm()
    try:
        with (
            patch.object(store, "begin_write_batch", side_effect=delete_then_begin),
            patch.object(memory, "submit_async_consolidation"),
        ):
            # Dropped, not failed and not retried: no exception reaches the worker.
            await memory.execute_task({"type": "consolidation", "bank_id": bank_id})

        assert committed == []
        assert await _bank_exists(memory, bank_id) is recreate
        # Nothing from before the delete came back: no observation, in either bank life.
        assert await _unit_count(memory, bank_id) == 0
    finally:
        memory._consolidation_llm_config = original_llm
        if await _bank_exists(memory, bank_id):
            await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_retain_job_for_a_deleted_bank_does_not_recreate_it(memory: MemoryEngine, request_context):
    bank_id = f"del-before-run-{uuid.uuid4().hex[:8]}"
    await memory.execute_task(
        {"type": "batch_retain", "bank_id": bank_id, "contents": [{"content": "Alice moved to Berlin"}]}
    )
    assert not await _bank_exists(memory, bank_id)


@pytest.mark.asyncio
async def test_bound_job_cannot_create_its_bank_after_a_delete(memory: MemoryEngine, request_context):
    bank_id = f"del-ensure-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    with await bind_job_bank(memory._pool, bank_id):
        # Still alive: the ensure is the usual no-op.
        assert await memory._ensure_bank_exists(bank_id, request_context) is False
        await memory.delete_bank(bank_id, request_context=request_context)
        with pytest.raises(BankDeletedError):
            await memory._ensure_bank_exists(bank_id, request_context)
    assert not await _bank_exists(memory, bank_id)
