"""When something happened, when you were told, and the flag that connects them.

Hindsight keeps two clocks per fact. `occurred_start`/`occurred_end` say when the
event happened; `mentioned_at` says when the material stating it was written. A
document retained today about a 2019 event carries both.

The connection between them is `fact_kind`, and it is load-bearing in a way
nothing warns you about: the pipeline keeps `occurred_start`/`occurred_end` only
for a fact marked `event`, and **silently discards them** for a `conversation`.
A date on a conversation fact is not an error, not a warning — it is just gone,
and the fact then behaves as though it happened at ingest time. Both halves of
that are asserted below, because the failure is invisible from the outside.

Once a date does survive, it drives recency: an old event decays, a fact with no
date does not. `query_timestamp` moves the "now" that decay is measured from,
which is what makes an as-of-then question answerable.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "What happened with Alice?"

OLD_EVENT = "Alice moved to Berlin | Involving: Alice"
RECENT_EVENT = "Alice adopted a cat | Involving: Alice"
UNDATED = "Alice mentioned liking jazz | Involving: Alice"


@pytest.fixture
async def bank_with_two_eras(client, llm, bank_id, settled) -> str:
    """Three facts retained in one breath: two datable events seven years apart,
    and one conversational remark carrying a date it is not entitled to keep."""
    llm.on_step("extract_facts").returns(
        extracted(
            fact(
                "Alice moved to Berlin",
                who="Alice",
                entities=["Alice", "Berlin"],
                fact_kind="event",
                occurred_start="2019-03-01T00:00:00Z",
                occurred_end="2019-03-01T00:00:00Z",
            ),
            fact(
                "Alice adopted a cat",
                who="Alice",
                entities=["Alice"],
                fact_kind="event",
                occurred_start="2026-08-01T00:00:00Z",
                occurred_end="2026-08-01T00:00:00Z",
            ),
            # Same date, but conversational — the date must not survive.
            fact(
                "Alice mentioned liking jazz",
                who="Alice",
                entities=["Alice"],
                occurred_start="2019-03-01T00:00:00Z",
            ),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin in 2019, adopted a cat, likes jazz.")
    await settled(bank_id)
    return bank_id


def _as_datetime(value: str) -> datetime:
    """The API serialises timestamps as ISO strings, not datetimes."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


async def _recency_by_text(client, bank: str, **kwargs) -> dict[str, float]:
    response = await client.arecall(bank_id=bank, query=QUERY, trace=True, **kwargs)
    return {entry["text"]: entry["score_components"]["recency"] for entry in response.trace["reranked"]}


async def test_only_an_event_fact_keeps_its_date(client, bank_with_two_eras):
    """The silent-discard contract, both directions in one assertion.

    The jazz fact was handed exactly the same `occurred_start` as the Berlin one
    and comes back with none, because it is a `conversation`. This is why
    `test_01` sees `occurred_start is None` for a fact whose `when` said "2021".
    """
    response = await client.arecall(bank_id=bank_with_two_eras, query=QUERY)
    by_text = {r.text: r for r in response.results}

    assert _as_datetime(by_text[OLD_EVENT].occurred_start).year == 2019
    assert _as_datetime(by_text[RECENT_EVENT].occurred_start).year == 2026
    assert by_text[UNDATED].occurred_start is None


async def test_the_ingest_clock_runs_regardless_of_the_event_clock(client, bank_with_two_eras):
    """All three arrived in the same retain, seconds ago — including the one that
    happened in 2019. `mentioned_at` tracks ingest, never the event."""
    response = await client.arecall(bank_id=bank_with_two_eras, query=QUERY)

    now = datetime.now(timezone.utc)
    for result in response.results:
        assert abs((now - _as_datetime(result.mentioned_at)).total_seconds()) < 300, (
            f"{result.text!r} must be mentioned_at ingest time, not its event date"
        )


async def test_recency_decays_from_the_event_date(client, bank_with_two_eras):
    """An old event is old, even though we learned about it a second ago.

    Ordering rather than exact values: the decay is measured against wall-clock
    now, so the numbers move every day the suite runs, but the ranking between
    them does not. The undated fact sits at the top because a fact with no event
    date is treated as current.
    """
    recency = await _recency_by_text(client, bank_with_two_eras)

    assert recency[OLD_EVENT] < recency[RECENT_EVENT] < recency[UNDATED]
    assert recency[UNDATED] == pytest.approx(1.0, abs=1e-6)


async def test_query_timestamp_moves_the_now_that_decay_is_measured_from(client, bank_with_two_eras):
    """Asking as of 2020 makes the 2019 move recent and the 2026 cat undecayed —
    the anchor is what makes a historical question answerable at all.

    Note it re-ranks rather than filters: everything still comes back.
    """
    now_anchored = await _recency_by_text(client, bank_with_two_eras)
    then_anchored = await _recency_by_text(client, bank_with_two_eras, query_timestamp="2020-01-01T00:00:00")

    # Seven years closer to the anchor, so it decays less.
    assert then_anchored[OLD_EVENT] > now_anchored[OLD_EVENT]
    # And an event still in the anchor's future cannot be stale at all.
    assert then_anchored[RECENT_EVENT] == pytest.approx(1.0, abs=1e-6)

    assert set(then_anchored) == {OLD_EVENT, RECENT_EVENT, UNDATED}


async def test_the_trace_reports_the_anchor_the_query_actually_used(client, llm, bank_id, settled):
    """A trace exists to explain a ranking, so it reports the inputs that produced
    it — including the anchor, not the moment the trace was built.

    It used to stamp `datetime.now(UTC)` (#4217), which pointed anyone debugging
    "why did my 2019 memory rank low when I asked as of 2020?" at today's date
    and the wrong conclusion.
    """
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    response = await client.arecall(
        bank_id=bank_id, query="Where does Alice live?", query_timestamp="2020-01-01T00:00:00", trace=True
    )

    assert response.trace["query"]["timestamp"].startswith("2020-01-01")
