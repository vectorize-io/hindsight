"""Recall can return the synthesis, the evidence, or both without saying the same
thing twice.

By default a recall returns whatever ranks — observation and raw facts alike —
and that is often duplication: the observation says Alice is settled in Berlin,
and right beneath it sit the two facts it was drawn from, saying the same thing
in more words. For an agent working to a token budget that is the worst outcome:
paying three times for one claim.

`prefer_observations` resolves it by dropping any fact a returned observation was
consolidated from — superseding rather than filtering, which is why it only
affects facts that specific observation actually cites. And when the evidence is
wanted after all, `include_source_facts` returns it as a separate section keyed
by id, so the answer stays one claim with its provenance attached rather than
four competing rows.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import extracted, fact, observes

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"

OBSERVATION = "Alice is settled in Berlin"
BERLIN = "Alice moved to Berlin | Involving: Alice"
LEASE = "Alice renewed her Berlin lease | Involving: Alice"


@pytest.fixture
async def observed_bank(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice renewed her Berlin lease", who="Alice", entities=["Alice", "Berlin"]),
        )
    )
    llm.on_step("consolidate").answers_with(observes(OBSERVATION))
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin. Alice renewed her Berlin lease.")
    await settled(bank_id)
    return bank_id


async def test_by_default_the_synthesis_and_its_evidence_both_come_back(client, observed_bank):
    """The duplication the next test removes — worth pinning so the default is a
    deliberate choice rather than an accident."""
    response = await client.arecall(bank_id=observed_bank, query=QUERY)

    assert sorted(r.text for r in response.results) == sorted([OBSERVATION, BERLIN, LEASE])


async def test_preferring_observations_drops_the_facts_they_superseded(client, observed_bank):
    """One claim, once. The two facts are not filtered out for being irrelevant —
    they are dropped because the observation above them already says what they
    say, and it cites them."""
    response = await client.arecall(bank_id=observed_bank, query=QUERY, prefer_observations=True)

    assert [r.text for r in response.results] == [OBSERVATION]


async def test_the_evidence_is_still_reachable_as_provenance(client, observed_bank):
    """Superseded is not hidden. The observation names its sources, and
    `include_source_facts` hydrates them into their own section — so an agent can
    show its work without paying for the facts twice in the ranked results."""
    response = await client.arecall(
        bank_id=observed_bank, query=QUERY, types=["observation"], include_source_facts=True
    )

    observation = response.results[0]
    assert observation.text == OBSERVATION
    assert len(observation.source_fact_ids) == 2

    # Keyed by id so a caller can resolve a citation without scanning a list.
    assert set(response.source_facts) == set(observation.source_fact_ids)
    assert sorted(f.text for f in response.source_facts.values()) == sorted([BERLIN, LEASE])
    assert response.source_facts_truncated is False


async def test_source_facts_are_absent_unless_asked_for(client, observed_bank):
    """They cost tokens to hydrate, so nothing pays for them by default."""
    response = await client.arecall(bank_id=observed_bank, query=QUERY, types=["observation"])

    assert response.source_facts is None
