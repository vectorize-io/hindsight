"""A consolidation that fails costs you the synthesis, never the facts.

Consolidation is the one background job that both *reads* and *deletes*: it
merges facts into observations and retires observations that no longer hold. A
failure partway through is therefore the most dangerous moment in the system —
the moment where a bank could end up with the deletions applied and the creates
missing, or with facts marked consumed by a round that never produced anything.

The correct behaviour is boring and that is the point. The model returns
nonsense, the round produces nothing, every fact is exactly where it was, and
the failure is *visible* — because a consolidation that fails silently leaves a
bank that looks fine and quietly stops synthesising.
"""

from __future__ import annotations

import asyncio

import pytest

from hindsight_system_tests.payloads import extracted, fact, observes

pytestmark = pytest.mark.asyncio

BERLIN = "Alice moved to Berlin | Involving: Alice"
LEASE = "Alice renewed her Berlin lease | Involving: Alice"


async def _facts_and_observations(client, bank: str) -> tuple[list[str], list[str]]:
    memories = await client.memory.list_memories(bank, limit=100)
    raw = sorted(m.text for m in memories.items if m.fact_type != "observation")
    observations = sorted(m.text for m in memories.items if m.fact_type == "observation")
    return raw, observations


@pytest.fixture
async def failed_round(client, llm, bank_id, settled) -> str:
    """A retain whose consolidation gets an unparseable answer."""
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice renewed her Berlin lease", who="Alice", entities=["Alice", "Berlin"]),
        )
    )
    llm.on_step("consolidate").returns_text("I'm afraid I can't do that.")

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin. Alice renewed her Berlin lease.")
    await settled(bank_id)
    return bank_id


async def test_no_fact_is_lost_to_a_failed_round(client, failed_round):
    """The unrecoverable failure, guarded. Facts are the source material; a round
    that consumed them and produced nothing cannot be undone."""
    raw, observations = await _facts_and_observations(client, failed_round)

    assert raw == sorted([BERLIN, LEASE])
    assert observations == [], "a failed round must not leave a half-written observation"


async def test_the_facts_still_answer_recalls(client, failed_round):
    """Not merely present in a listing — still reachable. A fact marked consumed
    by a round that failed would vanish from search while remaining in the table.
    """
    response = await client.arecall(bank_id=failed_round, query="Where does Alice live?")

    assert sorted(r.text for r in response.results) == sorted([BERLIN, LEASE])


async def test_the_operation_does_not_wedge(client, failed_round):
    """The round finishes. A consolidation left mid-flight blocks every later one
    for the bank, so one bad model answer would stop synthesis permanently."""
    stuck = await client.operations.list_operations(failed_round, status="processing", limit=100)
    assert stuck.operations == []

    pending = await client.operations.list_operations(failed_round, status="pending", limit=100)
    assert pending.operations == []


async def test_the_failure_is_visible_in_the_bank_stats(client, failed_round):
    """Silence would be the worst outcome: a bank that looks healthy and has
    quietly stopped consolidating. `failed_consolidation` counts facts still
    waiting for a synthesis that failed — a backlog gauge, not a call counter."""
    stats = await client.banks.get_agent_stats(failed_round)

    assert stats.failed_consolidation > 0


async def test_recovery_is_explicit_and_clears_the_backlog(client, llm, failed_round):
    """The failure is transient, not terminal — but recovery has to be asked for.

    An ordinary `trigger_consolidation` does *not* pick these facts back up: they
    are claimed as failed, and a scheduled round leaves them alone rather than
    re-feeding a poison input forever. `recover_consolidation` is the deliberate
    "try them again" door, and without it one bad model answer would mean those
    facts are never synthesised.

    The gauge is asserted in the same test rather than a second one: recovery
    runs on a maintenance sweep and takes ~15s to land, so splitting them would
    pay that wait twice to check two halves of one outcome.
    """
    llm.reset()
    llm.on_step("consolidate").answers_with(observes("Alice is settled in Berlin"))

    result = await client.banks.recover_consolidation(failed_round)
    # Counted per stranded fact, not per round.
    assert result.retried_count == 2

    # Polling for the outcome rather than for an idle bank: the work is enqueued
    # by a later sweep, so in between the bank is quiet and misleadingly "settled".
    observations: list[str] = []
    for _ in range(60):
        _, observations = await _facts_and_observations(client, failed_round)
        if observations:
            break
        await asyncio.sleep(1)

    assert observations == ["Alice is settled in Berlin"]

    raw, _ = await _facts_and_observations(client, failed_round)
    assert raw == sorted([BERLIN, LEASE])

    # `failed_consolidation` counts facts still waiting, so it has to come back
    # down — a gauge that only ever climbs says nothing about whether the bank
    # recovered.
    stats = await client.banks.get_agent_stats(failed_round)
    assert stats.failed_consolidation == 0
