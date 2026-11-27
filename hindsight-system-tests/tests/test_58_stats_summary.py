"""A bank's window, summed up by the server.

`GET /banks/{id}/stats/summary` is what a year-in-numbers view reads instead of
listing every memory: per-hour counts, the documents and the coding agents that
wrote them, the names most memories mention, and the first memory. Each piece is
an aggregate over the same window, so a window that misses the bank's activity
must come back empty on every one of them — a count that ignored the window
would show up there, not in the busy case.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from hindsight_client_api.exceptions import ApiException

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def bank_with_two_agents(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("extract_facts", contains="cello").returns(
        extracted(fact("Alice plays cello with Bob", who="Alice", entities=["Alice", "Bob"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(
        bank_id=bank_id, content="Alice moved to Berlin.", document_id="chat-1", tags=["harness:codex"]
    )
    await client.aretain(
        bank_id=bank_id, content="Alice plays cello with Bob.", document_id="chat-2", tags=["harness:claude-code"]
    )
    await settled(bank_id)
    return bank_id


async def test_the_summary_adds_up_the_window(client, bank_with_two_agents):
    now = datetime.now(timezone.utc)
    summary = await client.banks.get_stats_summary(
        bank_with_two_agents, start_date=now - timedelta(days=1), end_date=now + timedelta(days=1)
    )

    # Both facts were mentioned now: one or two UTC hours (the retains may straddle one).
    assert sum(h.world + h.experience + h.observation for h in summary.hours) == 2
    assert all(datetime.fromisoformat(h.time).minute == 0 for h in summary.hours)
    assert summary.documents == 2
    assert sorted((h.name, h.count) for h in summary.harnesses) == [("claude-code", 1), ("codex", 1)]
    # Alice is in both facts, so she leads; the rest tie at one and order by name.
    assert [(e.name, e.count) for e in summary.entities] == [("Alice", 2), ("Berlin", 1), ("Bob", 1)]
    assert summary.first_memory is not None
    assert summary.first_memory.text.startswith("Alice moved to Berlin")
    # Consolidation found nothing to observe, so there is no observation to call the strongest.
    assert summary.strongest_observation is None


async def test_a_window_before_the_activity_is_empty(client, bank_with_two_agents):
    now = datetime.now(timezone.utc)
    summary = await client.banks.get_stats_summary(
        bank_with_two_agents, start_date=now - timedelta(days=300), end_date=now - timedelta(days=1)
    )

    assert summary.hours == []
    assert summary.documents == 0
    assert summary.harnesses == []
    assert summary.entities == []
    assert summary.first_memory is None
    assert summary.strongest_observation is None


async def test_a_window_longer_than_a_year_is_refused(client, bank_with_two_agents):
    now = datetime.now(timezone.utc)
    with pytest.raises(ApiException) as refused:
        await client.banks.get_stats_summary(bank_with_two_agents, start_date=now - timedelta(days=800), end_date=now)
    assert refused.value.status == 400
