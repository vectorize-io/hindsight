"""A drain whose last batch is exactly the round cap must still refresh.

A round that processes exactly ``max_memories_per_round`` units used to leave
``hit_round_limit`` set even when the queue was empty, skip the refresh, and
re-queue a no-op follow-up that saw no new memories and never fanned out.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation.consolidator import run_consolidation_job
from hindsight_api.engine.memory_engine import MemoryEngine


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


def _make_config(**overrides):
    raw = _get_raw_config()
    return type(raw)(
        **{
            **{f: getattr(raw, f) for f in raw.__dataclass_fields__},
            **overrides,
        }
    )


async def _seed_facts(memory: MemoryEngine, bank_id: str, n: int) -> None:
    """Insert exactly n unconsolidated world units (retain extracts many facts)."""
    async with memory._pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at)
            VALUES ($1, $2, $3, 'world', '{}', now(), now())
            """,
            [(uuid.uuid4(), bank_id, f"Fact {i}: the user did activity number {i}.") for i in range(n)],
        )


async def _insert_refreshable_mm(memory: MemoryEngine, bank_id: str) -> str:
    mm_id = f"mm-{uuid.uuid4().hex}"
    async with memory._pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO mental_models
              (id, bank_id, subtype, name, source_query, content, tags, trigger, last_refreshed_at)
            VALUES ($1, $2, 'pinned', 'coalesce model', 'what changed', 'body', '{}',
                    '{"refresh_after_consolidation": true}'::jsonb, now() - interval '1 day')
            """,
            mm_id,
            bank_id,
        )
    return mm_id


@pytest.mark.asyncio
async def test_chained_rounds_refresh_once_when_last_batch_is_exact_cap(memory: MemoryEngine, request_context):
    bank_id = f"test-coalesce-chain-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    await _seed_facts(memory, bank_id, 6)
    await _insert_refreshable_mm(memory, bank_id)

    refresh_calls: list[str] = []

    async def _record(*, bank_id, mental_model_id, request_context, skip_if_in_flight=False):
        refresh_calls.append(mental_model_id)
        return {"operation_id": str(uuid.uuid4())}

    fake_config = _make_config(consolidation_max_memories_per_round=3)
    with (
        patch.object(memory._config_resolver, "resolve_full_config", return_value=fake_config),
        patch.object(memory, "submit_async_refresh_mental_model", side_effect=_record),
        patch.object(memory, "submit_async_consolidation") as mock_requeue,
    ):
        first = await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)
        assert first["status"] == "completed"
        assert first.get("mental_models_refreshed", 0) == 0
        assert refresh_calls == []
        assert mock_requeue.call_count == 1
        carried = mock_requeue.call_args.kwargs.get("pending_refresh_tags")

        second = await run_consolidation_job(
            memory_engine=memory,
            bank_id=bank_id,
            request_context=request_context,
            pending_refresh_tags=carried,
        )

    assert second["status"] == "completed"
    assert second.get("mental_models_refreshed", 0) == 1
    assert len(refresh_calls) == 1, f"chained rounds must fan out once, got {refresh_calls}"

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_single_round_still_refreshes_once(memory: MemoryEngine, request_context):
    bank_id = f"test-coalesce-single-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    await _seed_facts(memory, bank_id, 3)
    await _insert_refreshable_mm(memory, bank_id)

    refresh_calls: list[str] = []

    async def _record(*, bank_id, mental_model_id, request_context, skip_if_in_flight=False):
        refresh_calls.append(mental_model_id)
        return {"operation_id": str(uuid.uuid4())}

    fake_config = _make_config(consolidation_max_memories_per_round=10)
    with (
        patch.object(memory._config_resolver, "resolve_full_config", return_value=fake_config),
        patch.object(memory, "submit_async_refresh_mental_model", side_effect=_record),
        patch.object(memory, "submit_async_consolidation") as mock_requeue,
    ):
        result = await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

    assert result["status"] == "completed"
    mock_requeue.assert_not_called()
    assert result.get("mental_models_refreshed", 0) == 1
    assert len(refresh_calls) == 1

    await memory.delete_bank(bank_id, request_context=request_context)
