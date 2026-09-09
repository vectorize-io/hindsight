"""New evidence for something already observed strengthens it, rather than
growing a sibling next to it.

The alternative is worse than untidy. If every consolidation round creates a new
observation for the same recurring claim, the bank fills with near-duplicates
that all say roughly the same thing, each with a fraction of the evidence.
Recall then returns five paraphrases instead of one well-supported statement,
and `proof_count` — the signal for how strongly something is believed — stops
meaning anything, because the proof is spread across the copies.

So an update has to be a genuine merge: one observation, revised text, evidence
accumulated from both rounds.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import (
    Consolidation,
    ObservationUpdate,
    extracted,
    fact,
    fact_ids_in,
    observes,
)

pytestmark = pytest.mark.asyncio

FIRST_OBSERVATION = "Alice is settled in Berlin"
MERGED_OBSERVATION = "Alice has been settled in Berlin for two years"


async def _observations(client, bank: str) -> list[dict]:
    memories = await client.memory.list_memories(bank, limit=100)
    return [m for m in memories.items if m.fact_type == "observation"]


@pytest.fixture
async def merged_bank(client, llm, bank_id, settled) -> str:
    """Two retains a round apart; the second consolidation updates the first's
    observation rather than creating one of its own."""
    llm.on_step("extract_facts", contains="moved to Berlin").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").answers_with(observes(FIRST_OBSERVATION))

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    existing = (await _observations(client, bank_id))[0].id

    llm.reset()
    llm.on_step("extract_facts", contains="lease").returns(
        extracted(fact("Alice renewed her Berlin lease", who="Alice", entities=["Alice", "Berlin"]))
    )

    def merge(request) -> Consolidation:
        return Consolidation(
            updates=[
                ObservationUpdate(
                    text=MERGED_OBSERVATION,
                    observation_id=existing,
                    source_fact_ids=fact_ids_in(request.all_text),
                    reason="same canonical claim, more evidence",
                )
            ]
        )

    llm.on_step("consolidate").answers_with(merge)

    await client.aretain(bank_id=bank_id, content="Alice renewed her Berlin lease.")
    await settled(bank_id)
    return bank_id


async def test_the_bank_holds_one_observation_not_two(client, merged_bank):
    observations = await _observations(client, merged_bank)

    assert [o.text for o in observations] == [MERGED_OBSERVATION]


async def test_the_evidence_accumulates_across_rounds(client, merged_bank):
    """`proof_count` is the "how strongly is this believed" signal. A merge that
    replaced the evidence instead of adding to it would leave a two-round-old
    observation looking as thin as a brand new one."""
    observation = (await _observations(client, merged_bank))[0]

    assert observation.proof_count == 2
    assert len(observation.source_memory_ids) == 2


async def test_both_rounds_of_facts_are_still_there(client, merged_bank):
    memories = await client.memory.list_memories(merged_bank, limit=100)
    raw = sorted(m.text for m in memories.items if m.fact_type != "observation")

    assert raw == sorted(
        ["Alice moved to Berlin | Involving: Alice", "Alice renewed her Berlin lease | Involving: Alice"]
    )


async def test_recall_returns_the_merged_statement_once(client, merged_bank):
    """The user-visible payoff: one well-supported answer rather than two
    paraphrases competing for the same slot."""
    response = await client.arecall(bank_id=merged_bank, query="Where is Alice?", types=["observation"])

    assert [r.text for r in response.results] == [MERGED_OBSERVATION]
