"""A mental model is a standing answer that maintains itself.

The pitch is that the answer is already written: instead of reasoning over
memories at question time, a bank keeps a document per recurring question and
refreshes it as the facts move. So the properties worth pinning are the ones a
caller depends on — the model exists before it has content, a refresh replaces
that placeholder with a real answer drawn from the bank, and re-reading it is
cheap because nothing is recomputed.

The refresh runs the full reflect loop in the worker, which is why these stories
drive it (see `reflect.py`) rather than stubbing a single call: a mental model
that never climbs the search ladder is not reading the bank at all.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

ANSWER = "Alice lives in Berlin and has renewed her lease."
SOURCE_QUERY = "Where does Alice live?"


@pytest.fixture
async def bank_with_facts(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice renewed her Berlin lease", who="Alice", entities=["Alice", "Berlin"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=ANSWER)
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin. Alice renewed her Berlin lease.")
    await settled(bank_id)
    return bank_id


async def _create(client, bank: str, **kwargs) -> str:
    payload = {"name": "Alice housing", "source_query": SOURCE_QUERY, **kwargs}
    created = await client.mental_models.create_mental_model(bank, payload)
    return created.mental_model_id


async def test_a_new_model_exists_before_it_has_an_answer(client, bank_with_facts):
    """Creation is immediate; the answer is background work. The model is
    addressable in between, holding a visible placeholder rather than pretending
    to have content — a caller can tell "not written yet" from "nothing to say".
    """
    model_id = await _create(client, bank_with_facts)

    model = await client.mental_models.get_mental_model(bank_with_facts, model_id, detail="full")
    assert model.id == model_id
    assert model.name == "Alice housing"
    assert model.source_query == SOURCE_QUERY
    assert model.last_memory_seen_at is None, "nothing has been read yet"


async def test_a_refresh_writes_an_answer_drawn_from_the_bank(client, bank_with_facts, settled):
    model_id = await _create(client, bank_with_facts)
    await settled(bank_with_facts)

    model = await client.mental_models.get_mental_model(bank_with_facts, model_id, detail="full")
    assert model.content.strip() == ANSWER


async def test_the_model_records_that_it_has_read_the_bank(client, bank_with_facts, settled):
    """`last_refreshed_at` says when it ran; `last_memory_seen_at` says how far
    through the bank it got. They are separate because a refresh that ran and read
    nothing is a different state from one that is up to date — the distinction
    staleness checks depend on."""
    model_id = await _create(client, bank_with_facts)
    await settled(bank_with_facts)

    model = await client.mental_models.get_mental_model(bank_with_facts, model_id, detail="full")
    assert model.last_refreshed_at is not None
    assert model.is_stale is False


async def test_reading_a_model_does_not_recompute_it(client, bank_with_facts, settled, llm):
    """The whole economic argument: the answer is already written, so reading it
    is a database read. If a `get` triggered the reflect loop, the feature would
    just be recall with extra steps."""
    model_id = await _create(client, bank_with_facts)
    await settled(bank_with_facts)

    # Any further reflect turn would now go unscripted and fail the test, because
    # the rulebook is not reset mid-test — which is precisely the assertion.
    llm.reset()

    first = await client.mental_models.get_mental_model(bank_with_facts, model_id, detail="full")
    second = await client.mental_models.get_mental_model(bank_with_facts, model_id, detail="full")

    assert first.content == second.content == f"{ANSWER}\n"


async def test_a_model_is_listed_with_the_bank(client, bank_with_facts, settled):
    model_id = await _create(client, bank_with_facts)
    await settled(bank_with_facts)

    listing = await client.mental_models.list_mental_models(bank_with_facts)

    assert [m.id for m in listing.items] == [model_id]
    assert [m.name for m in listing.items] == ["Alice housing"]


async def test_deleting_a_model_leaves_the_facts_alone(client, bank_with_facts, settled):
    """A mental model is derived, like an observation: throwing it away must not
    reach the memories it was written from."""
    model_id = await _create(client, bank_with_facts)
    await settled(bank_with_facts)

    await client.mental_models.delete_mental_model(bank_with_facts, model_id)

    assert (await client.mental_models.list_mental_models(bank_with_facts)).items == []
    memories = await client.memory.list_memories(bank_with_facts, limit=100)
    assert sorted(m.text for m in memories.items) == sorted(
        ["Alice moved to Berlin | Involving: Alice", "Alice renewed her Berlin lease | Involving: Alice"]
    )
