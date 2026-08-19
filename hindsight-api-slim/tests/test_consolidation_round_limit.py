"""Integration tests for consolidation_max_memories_per_round config."""

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


@pytest.mark.asyncio
async def test_round_limit_caps_processed_memories(memory: MemoryEngine, request_context):
    """When max_memories_per_round is set, consolidation processes at most that many memories
    and re-submits itself for the remaining backlog."""
    bank_id = f"test-round-limit-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    # Disable consolidation during retain so we build up a backlog
    fake_config_no_obs = _make_config(enable_observations=False)

    with patch.object(memory._config_resolver, "resolve_full_config", return_value=fake_config_no_obs):
        for i in range(6):
            await memory.retain_async(
                bank_id=bank_id,
                content=f"Fact number {i}: The user enjoys activity {i} on weekends.",
                request_context=request_context,
            )

    # Verify we have unconsolidated memories
    # consolidation_state='pending' is the read API's name for exactly this predicate:
    # not consolidated, not failed, and a source fact type.
    unconsolidated = (
        await memory.list_memory_units(
            bank_id=bank_id, consolidation_state="pending", limit=1, request_context=request_context
        )
    )["total"]
    assert unconsolidated >= 6, f"Expected at least 6 unconsolidated memories, got {unconsolidated}"

    # Run consolidation with a round limit of 3
    round_limit = 3
    fake_config = _make_config(consolidation_max_memories_per_round=round_limit)

    with (
        patch.object(memory._config_resolver, "resolve_full_config", return_value=fake_config),
        patch.object(memory, "submit_async_consolidation") as mock_requeue,
    ):
        result = await run_consolidation_job(
            memory_engine=memory,
            bank_id=bank_id,
            request_context=request_context,
        )

    assert result["status"] == "completed"
    assert result["memories_processed"] <= round_limit

    # Must have re-queued consolidation for remaining work
    # This bank's memories are untagged, so the accumulated refresh-tag union is empty
    # and the re-queue carries pending_refresh_tags=None (#3411).
    mock_requeue.assert_called_once_with(
        bank_id=bank_id,
        request_context=request_context,
        observation_scopes=None,
        pending_refresh_tags=None,
    )

    # This bank has no mental models, so nothing is refreshed on any round.
    # (Refresh now runs every round, not just the final one — see #3411.)
    assert result.get("mental_models_refreshed", 0) == 0

    # Verify some memories are still unconsolidated
    still_unconsolidated = (
        await memory.list_memory_units(
            bank_id=bank_id, consolidation_state="pending", limit=1, request_context=request_context
        )
    )["total"]
    assert still_unconsolidated > 0, "Some memories should still be unconsolidated after hitting round limit"

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_unlimited_round_processes_all(memory: MemoryEngine, request_context):
    """When max_memories_per_round is 0 (unlimited), all memories are processed without re-queue."""
    bank_id = f"test-unlimited-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    # Disable consolidation during retain
    fake_config_no_obs = _make_config(enable_observations=False)

    with patch.object(memory._config_resolver, "resolve_full_config", return_value=fake_config_no_obs):
        for i in range(4):
            await memory.retain_async(
                bank_id=bank_id,
                content=f"Fact {i}: The user visited city {i} last year.",
                request_context=request_context,
            )

    # Run consolidation with unlimited round (0)
    fake_config = _make_config(consolidation_max_memories_per_round=0)

    with (
        patch.object(memory._config_resolver, "resolve_full_config", return_value=fake_config),
        patch.object(memory, "submit_async_consolidation") as mock_requeue,
    ):
        result = await run_consolidation_job(
            memory_engine=memory,
            bank_id=bank_id,
            request_context=request_context,
        )

    assert result["status"] == "completed"
    # Should NOT re-queue
    mock_requeue.assert_not_called()

    # All memories should be consolidated
    still_unconsolidated = (
        await memory.list_memory_units(
            bank_id=bank_id, consolidation_state="pending", limit=1, request_context=request_context
        )
    )["total"]
    assert still_unconsolidated == 0

    await memory.delete_bank(bank_id, request_context=request_context)
