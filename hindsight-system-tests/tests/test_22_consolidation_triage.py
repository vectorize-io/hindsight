"""Consolidation triage: the easy facts settled without the consolidation LLM.

With a decision model set up for consolidation (and ``consolidation_use_decision_model``
on, its default), consolidation asks it, per new fact, how it relates to the
observations it would be consolidated against. A restatement becomes evidence for its
observation and chatter is dropped — neither costs a consolidation call. It never writes
an observation's text: a new fact, and any fact with no observation to compare against,
reaches the LLM.

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


async def test_a_restatement_is_evidence_without_a_call(client, llm, stubs, bank_id, settled):
    await _with_triage(client, bank_id)
    stubs.triage.verdicts["lives in Berlin"] = "restates"
    llm.on_step("extract_facts", contains="Alice moved to Berlin.").returns(
        extracted(fact("Alice moved to Berlin", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").answers_with(observes("Alice lives in Berlin"))
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)
    assert len(llm.prompts_for("consolidate")) == 1, "the first fact has nothing to compare against"

    llm.on_step("extract_facts", contains="Alice lives in Berlin.").returns(
        extracted(fact("Alice lives in Berlin", entities=["Alice", "Berlin"]))
    )
    await client.aretain(bank_id=bank_id, content="Alice lives in Berlin.")
    await settled(bank_id)
    [observation] = await _observations(client, bank_id)
    assert observation.text == "Alice lives in Berlin", "a restatement leaves the text alone"
    assert observation.proof_count == 2, "and becomes its evidence"
    assert len(llm.prompts_for("consolidate")) == 1


async def test_a_new_fact_is_written_by_the_llm(client, llm, stubs, bank_id, settled):
    """Recorded as stated, it would keep its noise and skip the merge with the facts beside it."""
    await _with_triage(client, bank_id)
    stubs.triage.verdicts["Alice"] = "new"
    llm.on_step("extract_facts").returns(extracted(fact("Ugh, Alice moved to Berlin", entities=["Alice", "Berlin"])))
    llm.on_step("consolidate").answers_with(observes("Alice lives in Berlin"))

    await client.aretain(bank_id=bank_id, content="Ugh, Alice moved to Berlin.")
    await settled(bank_id)

    assert [o.text for o in await _observations(client, bank_id)] == ["Alice lives in Berlin"]


async def test_chatter_is_dropped_without_a_call(client, llm, stubs, bank_id, settled):
    await _with_triage(client, bank_id)
    stubs.triage.verdicts["coffee"] = "chatter"
    llm.on_step("extract_facts", contains="Berlin").returns(extracted(fact("The user lives in Berlin", entities=["user"])))
    llm.on_step("consolidate").answers_with(observes("The user lives in Berlin"))
    await client.aretain(bank_id=bank_id, content="I live in Berlin.")
    await settled(bank_id)

    llm.on_step("extract_facts", contains="coffee").returns(
        extracted(fact("The user went for a coffee", entities=["user"]))
    )
    await client.aretain(bank_id=bank_id, content="brb, coffee")
    await settled(bank_id)

    assert [o.text for o in await _observations(client, bank_id)] == ["The user lives in Berlin"]
    assert len(llm.prompts_for("consolidate")) == 1


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
