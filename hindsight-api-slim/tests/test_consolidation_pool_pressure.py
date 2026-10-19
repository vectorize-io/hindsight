"""Consolidation must not starve the DB pool or outlive auto-consolidation (issue #5590).

- Each related-observation recall holds one DB connection: a batch fans out one recall
  per fact, so the default recall budget of 4 multiplied pool use and timed out user
  recalls on a small pool.
- A round-limited chain stops re-queuing once ``enable_auto_consolidation`` is off,
  unless an explicit ``/consolidate`` call started it.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation import run_consolidation_job
from hindsight_api.engine.memory_engine import MemoryEngine


def _make_config(**overrides):
    raw = _get_raw_config()
    return type(raw)(**{**{f: getattr(raw, f) for f in raw.__dataclass_fields__}, **overrides})


async def _bank_with_backlog(memory: MemoryEngine, request_context, size: int) -> str:
    bank_id = f"test-pool-pressure-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    no_obs = _make_config(enable_observations=False)
    with patch.object(memory._config_resolver, "resolve_full_config", return_value=no_obs):
        for i in range(size):
            await memory.retain_async(
                bank_id=bank_id,
                content=f"Fact {i}: the user did activity number {i} on day {i}.",
                request_context=request_context,
            )
    return bank_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auto_enabled, caller_requested, expect_requeue",
    [
        (True, False, True),
        (False, False, False),
        (False, True, True),
    ],
)
async def test_round_limit_requeue_honours_auto_consolidation(
    memory: MemoryEngine, request_context, auto_enabled, caller_requested, expect_requeue
):
    bank_id = await _bank_with_backlog(memory, request_context, size=6)
    config = _make_config(
        enable_observations=True,
        enable_auto_consolidation=auto_enabled,
        consolidation_max_memories_per_round=2,
    )
    original_recall = memory.recall_async
    recall_budgets: list[int | None] = []

    async def spy_recall(*args, **kwargs):
        recall_budgets.append(kwargs.get("_connection_budget"))
        return await original_recall(*args, **kwargs)

    submit = AsyncMock(return_value={"operation_id": "x"})
    try:
        with (
            patch.object(memory._config_resolver, "resolve_full_config", return_value=config),
            patch.object(memory, "submit_async_consolidation", submit),
            patch.object(memory, "recall_async", side_effect=spy_recall),
        ):
            await run_consolidation_job(
                memory_engine=memory,
                bank_id=bank_id,
                request_context=request_context,
                caller_requested=caller_requested,
            )

        assert submit.await_count == (1 if expect_requeue else 0)
        if expect_requeue:
            assert submit.await_args.kwargs["caller_requested"] is caller_requested
        assert recall_budgets and all(b == 1 for b in recall_budgets), recall_budgets
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
