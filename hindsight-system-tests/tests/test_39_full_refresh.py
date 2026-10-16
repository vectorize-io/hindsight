"""A delta model can be rewritten from scratch without stopping being a delta model.

Delta refreshes edit a document in place, which keeps it stable — and, over many
edits, lets it drift from what a fresh write would say. A full refresh clears that
drift: once on demand (`mode: "full"` on the refresh) or on its own schedule
(`trigger.full_refresh_cron`). The seam worth pinning is that the one-off full run
and the model's standing delta mode meet without either overwriting the other: the
content is regenerated, the trigger still says delta, and the schedule survives.

The stub makes "it really ran in full mode" observable: a delta refresh would ask
for the delta-merge step, which no rule here answers, so the story fails if the
override is dropped on the way to the worker.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

SOURCE_QUERY = "Where does Alice live?"
FIRST = "Alice lives in Berlin."
REWRITTEN = "Alice lives in Berlin, where she renewed her lease."
WEEKLY = "0 3 * * 0"


@pytest.fixture
async def delta_model(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=FIRST)
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    created = await client.mental_models.create_mental_model(
        bank_id,
        {
            "name": "Alice housing",
            "source_query": SOURCE_QUERY,
            "trigger": {"mode": "delta", "full_refresh_cron": WEEKLY},
        },
    )
    await settled(bank_id)
    return created.mental_model_id


async def test_the_first_write_of_a_delta_model_counts_as_a_full_refresh(client, bank_id, delta_model):
    """With nothing to edit, a delta model's first refresh writes the whole
    document — and the schedule's clock has to know that, or the first scheduled
    full refresh would fire for a model that was just written from scratch."""
    model = await client.mental_models.get_mental_model(bank_id, delta_model, detail="full")

    assert model.content.strip() == FIRST
    assert model.last_full_refreshed_at is not None
    assert model.trigger.mode == "delta"
    assert model.trigger.full_refresh_cron == WEEKLY


async def test_a_full_refresh_rewrites_a_delta_model_and_leaves_it_delta(
    client, llm, bank_id, delta_model, settled
):
    before = await client.mental_models.get_mental_model(bank_id, delta_model, detail="full")

    llm.reset()
    reflect_loop(llm, answer=REWRITTEN)
    await client.arefresh_mental_model(bank_id, delta_model, mode="full")
    await settled(bank_id)

    after = await client.mental_models.get_mental_model(bank_id, delta_model, detail="full")
    assert after.content.strip() == REWRITTEN
    assert after.last_full_refreshed_at > before.last_full_refreshed_at
    # A one-off: the stored trigger is exactly what it was.
    assert after.trigger.mode == "delta"
    assert after.trigger.full_refresh_cron == WEEKLY
