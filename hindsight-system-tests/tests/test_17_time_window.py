"""Listing a bank by time, on the clock you actually meant.

Story 07 establishes that Hindsight keeps two clocks per fact: `mentioned_at`
tracks when the material arrived, `occurred_start`/`occurred_end` when the event
happened. This story is what that separation is *for* on the read side — "show me
everything from March 2019" and "show me everything ingested this morning" are
different questions, and until #4349 the list endpoints could express neither.
A caller had to page the whole bank and filter client-side.

The seam worth testing is the one between the two stories: every fact below is
retained in a single breath, seconds ago, so ingest time cannot distinguish them.
Only the event clock can — which means a window that came back with the wrong set
would be indistinguishable from a window that ignored `time_field` entirely.

The other half is the undated fact. Ordering by a column a row has no value for
cannot place that row, so it is dropped rather than parked at one end. That makes
`total` count the window and not the bank, which is the number every paginating
client pages on.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

BERLIN = "Alice moved to Berlin"
CAT = "Alice adopted a cat"
JAZZ = "Alice mentioned liking jazz"

DOCUMENT_ID = "alice-timeline"


@pytest.fixture
async def bank_with_two_eras(client, llm, bank_id, settled) -> str:
    """Two datable events seven years apart, plus one remark with no event date.

    All three arrive in the same retain, so they share an ingest time.
    """
    llm.on_step("extract_facts").returns(
        extracted(
            fact(
                BERLIN,
                who="Alice",
                entities=["Alice", "Berlin"],
                fact_kind="event",
                occurred_start="2019-03-01T00:00:00Z",
                occurred_end="2019-03-01T00:00:00Z",
            ),
            fact(
                CAT,
                who="Alice",
                entities=["Alice"],
                fact_kind="event",
                occurred_start="2026-08-01T00:00:00Z",
                occurred_end="2026-08-01T00:00:00Z",
            ),
            # Conversational: the pipeline keeps no event date for it (story 07).
            fact(JAZZ, who="Alice", entities=["Alice"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(
        bank_id=bank_id,
        content="Alice moved to Berlin in 2019, adopted a cat, likes jazz.",
        document_id=DOCUMENT_ID,
    )
    await settled(bank_id)
    return bank_id


def _texts(listing) -> set[str]:
    # Facts come back with their entity suffix appended; compare on the stem.
    return {item.text.split(" | ")[0] for item in listing.items}


async def test_the_event_clock_and_the_ingest_clock_answer_differently(client, bank_with_two_eras):
    """The same bank, the same instant, two different answers — which is the whole
    point of naming the axis."""
    by_event = await client.memory.list_memories(
        bank_with_two_eras,
        time_field="occurred_start",
        start_date="2019-01-01T00:00:00Z",
        end_date="2020-01-01T00:00:00Z",
        limit=100,
    )
    assert _texts(by_event) == {BERLIN}
    assert by_event.total == 1, "total counts the window, not the bank"

    # Everything was ingested seconds ago, so the ingest clock keeps all three —
    # including the 2019 event the event clock just isolated.
    now = datetime.now(timezone.utc)
    by_ingest = await client.memory.list_memories(
        bank_with_two_eras,
        time_field="created_at",
        start_date=(now - timedelta(hours=1)).isoformat(),
        limit=100,
    )
    assert _texts(by_ingest) == {BERLIN, CAT, JAZZ}
    assert by_ingest.total == 3


async def test_a_fact_with_no_date_on_that_axis_is_left_out(client, bank_with_two_eras):
    """Dropped, not parked at one end — and absent from `total` with it.

    The default listing still carries it, so this is the window's doing and not
    the fact having gone missing.
    """
    dated = await client.memory.list_memories(bank_with_two_eras, time_field="occurred_start", limit=100)
    assert JAZZ not in _texts(dated)
    assert dated.total == len(dated.items) == 2

    everything = await client.memory.list_memories(bank_with_two_eras, limit=100)
    assert JAZZ in _texts(everything)


async def test_an_empty_window_answers_empty_on_a_bank_that_is_not(client, bank_with_two_eras):
    """`total: 0` on a bank that plainly is not empty.

    Documented, and asserted here so it stays deliberate: the alternative —
    falling back to ingest time for the undated rows, the way the timeseries
    endpoint buckets — would answer a question nobody asked.
    """
    listing = await client.memory.list_memories(bank_with_two_eras, time_field="mentioned_at", limit=100)
    assert listing.total > 0, "mentioned_at is set at ingest, so this axis is populated"

    # The window itself can still be empty without the bank being empty.
    ancient = await client.memory.list_memories(
        bank_with_two_eras,
        time_field="mentioned_at",
        end_date="2000-01-01T00:00:00Z",
        limit=100,
    )
    assert ancient.total == 0
    assert ancient.items == []


async def test_documents_take_the_same_window(client, bank_with_two_eras):
    """The document listing gained the same parameters, on its own two axes."""
    now = datetime.now(timezone.utc)

    recent = await client.documents.list_documents(
        bank_with_two_eras,
        time_field="created_at",
        start_date=(now - timedelta(hours=1)).isoformat(),
        limit=100,
    )
    assert [d.id for d in recent.items] == [DOCUMENT_ID]
    assert recent.total == 1

    past = await client.documents.list_documents(
        bank_with_two_eras,
        time_field="created_at",
        end_date="2020-01-01T00:00:00Z",
        limit=100,
    )
    assert past.total == 0
    assert past.items == []
