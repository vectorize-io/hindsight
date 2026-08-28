"""Recall never answers a match with nothing just because max_tokens is small (issue #3688).

The packing rules themselves are covered by tests/test_fact_budget_selection.py;
this walks the real recall path to check the budget is wired to it.
"""

import pytest

from hindsight_api.engine.memory_engine import Budget


@pytest.mark.asyncio
async def test_tight_budget_returns_the_top_fact_whole(memory, request_context):
    bank_id = "test-recall-token-budget"

    try:
        await memory.retain_async(
            bank_id=bank_id,
            content=(
                "Priya leads the payments platform team in Berlin. "
                "She migrated the settlement service from Kafka to Pulsar last spring. "
                "Her team owns the reconciliation pipeline and the payout scheduler. "
                "Priya reviews every schema change to the ledger tables herself."
            ),
            context="team notes",
            request_context=request_context,
        )

        generous = await memory.recall_async(
            bank_id=bank_id,
            query="what does Priya work on",
            max_tokens=4096,
            budget=Budget.MID,
            request_context=request_context,
        )
        assert len(generous.results) > 1, "fixture should yield several facts to spend a budget on"

        # Every fact is individually over this budget — before #3688 the whole
        # answer came back empty, which reads as "this bank knows nothing".
        tight = await memory.recall_async(
            bank_id=bank_id,
            query="what does Priya work on",
            max_tokens=1,
            budget=Budget.MID,
            request_context=request_context,
        )
        assert len(tight.results) == 1
        assert tight.results[0].id == generous.results[0].id, "the surviving fact is the top-ranked one"
        assert tight.results[0].text == generous.results[0].text, "the fact is returned whole, not clipped"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
