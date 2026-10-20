"""An absurd date must not take the bank down with it.

Recency decay is an exponential over "days ago". A fact dated in the future
makes that negative, and the naive form raises `OverflowError` — which does not
degrade the ranking of one fact, it fails the *entire recall*, for every query,
until someone finds the row. That is the failure this guards: a single bad date,
extracted from one careless document, denies service to the whole bank.

Dates like these are not hypothetical. Models hallucinate year 9999, documents
carry placeholder timestamps, and a clock skew is enough to put an event
milliseconds into the future.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "What do we know about Alice?"

NORMAL = "Alice moved to Berlin | Involving: Alice"
FAR_FUTURE = "Alice will attend the centenary gala | Involving: Alice"
DISTANT_PAST = "Alice's family archive begins | Involving: Alice"


async def test_a_far_future_date_does_not_deny_the_whole_bank(client, llm, bank_id, settled):
    """The regression guard. The future-dated fact makes `days_ago` negative; the
    clamp must happen before the exponential, not after it.

    Note the assertion is that *every* fact comes back — the ordinary one
    included. A recall that returns only the sane fact would mean the bad row was
    dropped; a recall that raises would mean it poisoned the query. Neither is
    acceptable.
    """
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact(
                "Alice will attend the centenary gala",
                who="Alice",
                entities=["Alice"],
                occurred_start="9999-12-31T23:59:59Z",
                occurred_end="9999-12-31T23:59:59Z",
            ),
        )
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin. Alice will attend the centenary gala.")
    await settled(bank_id)

    response = await client.arecall(bank_id=bank_id, query=QUERY)

    assert sorted(r.text for r in response.results) == sorted([NORMAL, FAR_FUTURE])


async def test_a_distant_past_date_ranks_rather_than_crashes(client, llm, bank_id, settled):
    """The other end of the range. A very old event decays, but decaying to a
    small number is ranking; raising is not."""
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact(
                "Alice's family archive begins",
                who="Alice",
                entities=["Alice"],
                occurred_start="0001-01-01T00:00:00Z",
                occurred_end="0001-01-01T00:00:00Z",
            ),
        )
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin. The family archive begins long ago.")
    await settled(bank_id)

    response = await client.arecall(bank_id=bank_id, query=QUERY, trace=True)

    assert sorted(r.text for r in response.results) == sorted([NORMAL, DISTANT_PAST])
    # Every recency score stays inside the unit interval it is defined on. An
    # overflow or a sign slip escapes that range long before it raises, so this
    # catches the same bug one step earlier.
    for entry in response.trace["reranked"]:
        recency = entry["score_components"]["recency"]
        assert 0.0 <= recency <= 1.0, f"recency out of range for {entry['text']!r}: {recency}"


async def test_an_open_ended_range_still_recalls_and_consolidates(client, llm, bank_id, settled):
    """#5820. A model writing "still ongoing" as the largest possible date — on the first
    fact of a batch, which gets no ordering offset — used to be stored as Postgres
    `infinity`. That reads back without a timezone next to an aware start, so the temporal
    arm raised on the midpoint and failed every recall whose window reached the fact; the
    recall consolidation runs failed the same way, on every retry. The placeholder is now an
    unknown bound: the fact keeps its real start and is found in that window.
    """
    llm.on_step("extract_facts").returns(
        extracted(
            fact(
                "Alice started rebuilding the bridge",
                who="Alice",
                entities=["Alice"],
                fact_kind="event",  # a conversation fact's dates are discarded — see test_07
                occurred_start="2025-01-16T00:00:00Z",
                occurred_end="9999-12-31T23:59:59.999999Z",
            ),
        )
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="Alice started rebuilding the bridge in January 2025; still going.")
    await settled(bank_id)

    response = await client.arecall(
        bank_id=bank_id,
        query=QUERY,
        temporal_window={"start": "2025-01-01T00:00:00Z", "end": "2025-01-31T23:59:59Z"},
    )

    assert [r.text for r in response.results] == ["Alice started rebuilding the bridge | Involving: Alice"]
    stats = await client.banks.get_agent_stats(bank_id)
    assert stats.failed_consolidation == 0
