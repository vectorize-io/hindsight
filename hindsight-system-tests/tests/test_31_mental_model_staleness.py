"""A mental model knows when the bank has moved on without it.

The promise is "always current, without asking", and the machinery behind it is
a watermark: `last_memory_seen_at` records how far through the bank the last
refresh read. A write later than that watermark makes the model stale; a refresh
advances it.

Both halves have to work or the feature inverts. A watermark that never advances
leaves a model permanently stale — every scheduled sweep refreshes it again, at
full cost, forever. A staleness check that never fires leaves a model
permanently *fresh* — it stops updating, keeps answering confidently, and gets
further from the truth with every retain. The second is the dangerous one,
because nothing about it looks wrong.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

FIRST_ANSWER = "Alice lives in Berlin."
SECOND_ANSWER = "Alice lives in Berlin and plays the cello."


@pytest.fixture
async def model(client, llm, bank_id, settled):
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("extract_facts", contains="cello").returns(
        extracted(fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=FIRST_ANSWER)

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    created = await client.mental_models.create_mental_model(
        bank_id, {"name": "Housing", "source_query": "Where does Alice live?"}
    )
    await settled(bank_id)
    return created.mental_model_id


async def _read(client, bank: str, model_id: str):
    return await client.mental_models.get_mental_model(bank, model_id, detail="full")


async def test_a_freshly_refreshed_model_is_not_stale(client, bank_id, model):
    current = await _read(client, bank_id, model)

    assert current.content.strip() == FIRST_ANSWER
    assert current.is_stale is False
    assert current.last_memory_seen_at is not None, "a refresh that read the bank must record how far it got"


async def test_a_later_write_makes_it_stale(client, bank_id, model, settled):
    """The signal that drives every scheduled refresh. If it never fires, the
    model quietly stops updating while still answering with confidence."""
    await client.aretain(bank_id=bank_id, content="Alice plays cello.")
    await settled(bank_id)

    current = await _read(client, bank_id, model)
    assert current.is_stale is True


async def test_going_stale_does_not_change_the_answer(client, bank_id, model, settled):
    """Stale means "worth refreshing", not "wrong". The old answer keeps serving
    until a refresh replaces it — blanking it on staleness would leave a gap
    every time anything was retained."""
    await client.aretain(bank_id=bank_id, content="Alice plays cello.")
    await settled(bank_id)

    current = await _read(client, bank_id, model)
    assert current.content.strip() == FIRST_ANSWER


async def test_the_watermark_does_not_move_until_a_refresh_reads_the_bank(client, bank_id, model, settled):
    """A write is not a read. The watermark records what the *model* has seen, so
    it must not drift forward just because the bank grew."""
    before = (await _read(client, bank_id, model)).last_memory_seen_at

    await client.aretain(bank_id=bank_id, content="Alice plays cello.")
    await settled(bank_id)

    assert (await _read(client, bank_id, model)).last_memory_seen_at == before


async def test_refreshing_takes_in_the_new_facts_and_clears_the_flag(client, llm, bank_id, model, settled):
    """The loop closing: refresh, new answer, watermark advanced, no longer stale.

    A watermark that failed to advance here would leave the model stale forever
    and make every scheduled sweep redo the same work at full cost.
    """
    await client.aretain(bank_id=bank_id, content="Alice plays cello.")
    await settled(bank_id)
    watermark_before = (await _read(client, bank_id, model)).last_memory_seen_at

    llm.reset()
    reflect_loop(llm, answer=SECOND_ANSWER)
    await client.mental_models.refresh_mental_model(bank_id, model)
    await settled(bank_id)

    current = await _read(client, bank_id, model)
    assert current.content.strip() == SECOND_ANSWER
    assert current.is_stale is False
    assert current.last_memory_seen_at > watermark_before


async def test_the_previous_answer_is_kept_in_history(client, llm, bank_id, model, settled):
    """A model that rewrites itself in place is unauditable — "it used to say
    something else" needs to be answerable."""
    llm.reset()
    reflect_loop(llm, answer=SECOND_ANSWER)
    await client.mental_models.refresh_mental_model(bank_id, model)
    await settled(bank_id)

    history = await client.mental_models.get_mental_model_history(bank_id, model)
    assert any(entry["previous_content"].strip() == FIRST_ANSWER for entry in history)
