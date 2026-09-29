"""When the world changes, the old observation goes away rather than coexisting
with its replacement.

Some new evidence does not refine an existing observation, it retires it. Alice
moved to Lisbon; "Alice is settled in Berlin" is not out of date at the margins,
it is no longer true of anyone.

Coexistence is the failure. Two contradictory observations both come back from a
recall, both look equally supported, and the layer built to give an agent a
settled view of the world instead hands it a disagreement to arbitrate — with no
signal about which side won. The point of an observation is to *be* the settled
view, so retiring the loser is part of the job.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import (
    Consolidation,
    Observation,
    ObservationDelete,
    extracted,
    fact,
    fact_ids_in,
    observes,
)

pytestmark = pytest.mark.asyncio

OLD_OBSERVATION = "Alice is settled in Berlin"
NEW_OBSERVATION = "Alice has moved to Lisbon"


async def _observations(client, bank: str) -> list[dict]:
    memories = await client.memory.list_memories(bank, limit=100)
    return [m for m in memories.items if m.fact_type == "observation"]


@pytest.fixture
async def superseded_bank(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts", contains="moved to Berlin").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").answers_with(observes(OLD_OBSERVATION))

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    stale = (await _observations(client, bank_id))[0].id

    llm.reset()
    llm.on_step("extract_facts", contains="Lisbon").returns(
        extracted(fact("Alice moved to Lisbon", who="Alice", entities=["Alice", "Lisbon"]))
    )

    def supersede(request) -> Consolidation:
        return Consolidation(
            creates=[
                Observation(
                    text=NEW_OBSERVATION,
                    source_fact_ids=fact_ids_in(request.all_text),
                    reason="the move is a new claim, not an amendment",
                )
            ],
            deletes=[ObservationDelete(observation_id=stale, reason="Alice no longer lives in Berlin")],
        )

    llm.on_step("consolidate").answers_with(supersede)

    await client.aretain(bank_id=bank_id, content="Alice moved to Lisbon.")
    await settled(bank_id)
    return bank_id


async def test_only_the_current_observation_remains(client, superseded_bank):
    observations = await _observations(client, superseded_bank)

    assert [o.text for o in observations] == [NEW_OBSERVATION]


async def test_a_recall_does_not_return_both_sides_of_the_contradiction(client, superseded_bank):
    """The user-visible failure: an agent asking where Alice lives must not be
    handed Berlin and Lisbon with equal confidence."""
    response = await client.arecall(bank_id=superseded_bank, query="Where does Alice live?", types=["observation"])

    texts = [r.text for r in response.results]
    assert NEW_OBSERVATION in texts
    assert OLD_OBSERVATION not in texts


async def test_retiring_an_observation_does_not_retire_its_evidence(client, superseded_bank):
    """The Berlin *fact* is still true — Alice did move there once. Only the
    synthesis drawn from it stopped being current, so the historical record has
    to survive the observation that summarised it.
    """
    memories = await client.memory.list_memories(superseded_bank, limit=100)
    raw = sorted(m.text for m in memories.items if m.fact_type != "observation")

    assert raw == sorted(["Alice moved to Berlin | Involving: Alice", "Alice moved to Lisbon | Involving: Alice"])
