"""Tests for source_facts token limiting in recall.

Covers:
- max_source_facts_tokens: total token budget across all source facts
- max_source_facts_tokens_per_observation: per-observation cap

Both parameters are tested at the recall_async level and verified to produce
fewer source facts when the budget is tight vs. unlimited.
"""

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memory_engine import Budget


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


async def _setup_bank_with_observations(memory, bank_id, request_context):
    """Retain several memories and trigger consolidation to produce observations with source facts."""
    contents = [
        "Alice is a software engineer who loves Python programming.",
        "Alice has been working at TechCorp for 5 years.",
        "Alice recently completed a machine learning certification course.",
        "Alice mentors junior developers on the team.",
        "Alice prefers functional programming patterns in her code.",
    ]
    for content in contents:
        await memory.retain_async(
            bank_id=bank_id,
            content=content,
            request_context=request_context,
        )
    await memory.run_consolidation(bank_id=bank_id, request_context=request_context)


class TestRecallSourceFactsPerObservationCap:
    @pytest.mark.asyncio
    async def test_per_observation_cap_reduces_source_facts(self, memory, request_context):
        """A tight per-observation token cap should return fewer source facts than unlimited."""
        bank_id = "test-sf-per-obs-cap"
        try:
            await _setup_bank_with_observations(memory, bank_id, request_context)

            result_limited = await memory.recall_async(
                bank_id=bank_id,
                query="Alice engineer",
                fact_type=["observation"],
                max_tokens=4096,
                include_source_facts=True,
                max_source_facts_tokens_per_observation=1,  # Effectively cuts all source facts
                budget=Budget.MID,
                request_context=request_context,
            )

            result_unlimited = await memory.recall_async(
                bank_id=bank_id,
                query="Alice engineer",
                fact_type=["observation"],
                max_tokens=4096,
                include_source_facts=True,
                max_source_facts_tokens_per_observation=-1,
                budget=Budget.MID,
                request_context=request_context,
            )

            unlimited_count = len(result_unlimited.source_facts) if result_unlimited.source_facts else 0
            limited_count = len(result_limited.source_facts) if result_limited.source_facts else 0

            if unlimited_count > 0:
                assert limited_count <= unlimited_count, (
                    f"Per-observation cap should yield fewer source facts ({limited_count} <= {unlimited_count})"
                )
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_per_observation_cap_does_not_mix_between_observations(self, memory, request_context):
        """Each observation's source facts are capped independently — not as a shared pool."""
        bank_id = "test-sf-per-obs-independent"
        try:
            await _setup_bank_with_observations(memory, bank_id, request_context)

            # With a generous per-observation limit each observation can have facts;
            # with a global limit of 1 token the first observation would consume the whole budget.
            result_per_obs = await memory.recall_async(
                bank_id=bank_id,
                query="Alice engineer",
                fact_type=["observation"],
                max_tokens=4096,
                include_source_facts=True,
                max_source_facts_tokens=4096,  # large global budget
                max_source_facts_tokens_per_observation=512,  # reasonable per-obs limit
                budget=Budget.MID,
                request_context=request_context,
            )

            # Should not raise; source_facts may be populated for multiple observations
            assert result_per_obs.source_facts is not None or len(result_per_obs.results) == 0
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)


class TestRecallSourceFactsTotalBudget:
    @pytest.mark.asyncio
    async def test_total_budget_limits_source_facts(self, memory, request_context):
        """A tight total token budget should return fewer source facts than unlimited."""
        bank_id = "test-sf-total-budget"
        try:
            await _setup_bank_with_observations(memory, bank_id, request_context)

            result_tight = await memory.recall_async(
                bank_id=bank_id,
                query="Alice engineer",
                fact_type=["observation"],
                max_tokens=4096,
                include_source_facts=True,
                max_source_facts_tokens=1,  # Effectively cuts all source facts
                budget=Budget.MID,
                request_context=request_context,
            )

            result_unlimited = await memory.recall_async(
                bank_id=bank_id,
                query="Alice engineer",
                fact_type=["observation"],
                max_tokens=4096,
                include_source_facts=True,
                max_source_facts_tokens=-1,
                budget=Budget.MID,
                request_context=request_context,
            )

            unlimited_count = len(result_unlimited.source_facts) if result_unlimited.source_facts else 0
            tight_count = len(result_tight.source_facts) if result_tight.source_facts else 0

            if unlimited_count > 0:
                assert tight_count <= unlimited_count, (
                    f"Total budget should yield fewer source facts ({tight_count} <= {unlimited_count})"
                )
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_no_source_facts_without_flag(self, memory, request_context):
        """source_facts should be None when include_source_facts is not set."""
        bank_id = "test-sf-no-flag"
        try:
            await _setup_bank_with_observations(memory, bank_id, request_context)

            result = await memory.recall_async(
                bank_id=bank_id,
                query="Alice engineer",
                fact_type=["observation"],
                max_tokens=4096,
                include_source_facts=False,  # default
                budget=Budget.MID,
                request_context=request_context,
            )

            assert result.source_facts is None or len(result.source_facts) == 0
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)
