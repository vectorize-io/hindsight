"""Consolidation triage: the easy facts settled without the consolidation LLM.

With a decision model set up for consolidation (and ``consolidation_use_decision_model``
on, its default), consolidation asks it, per new fact, how it relates to the
observations it would be consolidated against. A restatement becomes evidence for its
observation, a new entity or attribute becomes an observation of its own, chatter is
dropped — none of them costs a consolidation
call. Only a fact that changes what an observation says reaches the LLM.

The verdicts are scripted (``stubs.triage``); what these stories pin is what the
server does with each one, and that nothing reaches the LLM it did not need to —
an unstubbed consolidation call fails the story.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import extracted, fact, observes

pytestmark = pytest.mark.asyncio


async def _observations(client, bank: str) -> list:
    memories = await client.memory.list_memories(bank, limit=100)
    return [m for m in memories.items if m.fact_type == "observation"]


async def _with_triage(client, bank: str, **updates) -> None:
    await client.banks.update_bank_config(bank, {"updates": {"consolidation_use_decision_model": True, **updates}})


async def test_new_facts_and_restatements_never_reach_the_llm(client, llm, stubs, bank_id, settled):
    await _with_triage(client, bank_id)
    stubs.triage.verdicts.update({"moved to Berlin": "new", "lives in Berlin": "restates"})

    llm.on_step("extract_facts", contains="Alice moved to Berlin.").returns(
        extracted(fact("Alice moved to Berlin", entities=["Alice", "Berlin"]))
    )
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)
    [observation] = await _observations(client, bank_id)
    assert observation.text == "Alice moved to Berlin", "a new fact is recorded as stated"

    llm.on_step("extract_facts", contains="Alice lives in Berlin.").returns(
        extracted(fact("Alice lives in Berlin", entities=["Alice", "Berlin"]))
    )
    await client.aretain(bank_id=bank_id, content="Alice lives in Berlin.")
    await settled(bank_id)
    [observation] = await _observations(client, bank_id)
    assert observation.text == "Alice moved to Berlin", "a restatement leaves the text alone"
    assert observation.proof_count == 2, "and becomes its evidence"

    assert llm.prompts_for("consolidate") == []


async def test_a_new_fact_already_stated_word_for_word_is_evidence(client, llm, stubs, bank_id, settled):
    """Recorded verbatim, it would be dropped as a duplicate create and back nothing."""
    await _with_triage(client, bank_id)
    stubs.triage.verdicts["moved to Berlin"] = "new"
    llm.on_step("extract_facts").returns(extracted(fact("Alice moved to Berlin", entities=["Alice", "Berlin"])))

    for _ in range(2):
        await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
        await settled(bank_id)

    [observation] = await _observations(client, bank_id)
    assert observation.proof_count == 2


async def test_chatter_is_dropped_without_a_call(client, llm, stubs, bank_id, settled):
    await _with_triage(client, bank_id)
    stubs.triage.verdicts["coffee"] = "chatter"
    llm.on_step("extract_facts").returns(extracted(fact("The user went for a coffee", entities=["user"])))

    await client.aretain(bank_id=bank_id, content="brb, coffee")
    await settled(bank_id)

    assert await _observations(client, bank_id) == []
    assert llm.prompts_for("consolidate") == []


async def test_a_change_still_goes_to_the_llm(client, llm, stubs, bank_id, settled):
    """Anything that affects what an observation says needs the rewrite — the stub's
    default verdict, so this is also what an unsure judgment does."""
    await _with_triage(client, bank_id)
    llm.on_step("extract_facts").returns(extracted(fact("Alice moved to Lisbon", entities=["Alice", "Lisbon"])))
    llm.on_step("consolidate").answers_with(observes("Alice lives in Lisbon"))

    await client.aretain(bank_id=bank_id, content="Alice moved to Lisbon.")
    await settled(bank_id)

    assert [o.text for o in await _observations(client, bank_id)] == ["Alice lives in Lisbon"]
    assert len(llm.prompts_for("consolidate")) == 1


async def test_a_fact_worded_against_the_mission_goes_to_the_llm(client, llm, stubs, bank_id, settled):
    """A mission may want a fact rephrased ("name no company"); recorded verbatim it
    would ignore that, so a fact the judgment does not clear goes to the LLM."""
    await _with_triage(client, bank_id, observations_mission="Record only industry trends. Name no company.")
    stubs.triage.verdicts["Northwind"] = "new"
    llm.on_step("extract_facts").returns(
        extracted(fact("Northwind is moving off frontier models", entities=["Northwind"]))
    )
    llm.on_step("consolidate").answers_with(observes("Companies are moving off frontier models"))

    await client.aretain(bank_id=bank_id, content="Northwind is moving off frontier models.")
    await settled(bank_id)

    assert [o.text for o in await _observations(client, bank_id)] == ["Companies are moving off frontier models"]


async def test_a_fact_the_mission_wants_as_worded_never_reaches_the_llm(client, llm, stubs, bank_id, settled):
    """The common case under a bank's own mission: the fact is on topic and breaks no
    wording rule, so it is recorded as stated without a consolidation call."""
    await _with_triage(client, bank_id, observations_mission="Keep a catalogue of services and who owns them.")
    stubs.triage.verdicts["billing-api"] = "new"
    stubs.triage.fits_mission.add("billing-api")
    llm.on_step("extract_facts").returns(
        extracted(fact("The billing-api service is owned by team Atlas", entities=["billing-api", "team Atlas"]))
    )

    await client.aretain(bank_id=bank_id, content="Team Atlas owns billing-api.")
    await settled(bank_id)

    assert [o.text for o in await _observations(client, bank_id)] == ["The billing-api service is owned by team Atlas"]
    assert llm.prompts_for("consolidate") == []
