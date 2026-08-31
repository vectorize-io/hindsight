"""Coarse dates are extracted as full period spans (issue #3893).

When the text states only a year or only a month, the extracted fact must carry
that whole period in ``occurred_start``/``occurred_end`` rather than collapsing
to a single instant. Recall depends on this: ``_spans_calendar_period`` reads the
span back to tell a coarse date apart from a precise one, and a collapsed date is
indistinguishable from an exact one — it then gets aged from the period's first
day (or, worse, stamped with the ingest date) and the recency boost can outrank a
strictly more relevant memory.

Real-LLM test: whether the model follows the coarse-date instruction cannot be
simulated with MockLLM. The assertions are structural rather than judged because
the contract is a numeric span, and it is asserted through the same production
helper recall uses, so the two halves of the fix cannot drift apart.
"""

from datetime import datetime

import pytest
from dateutil import parser as date_parser

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from hindsight_api.engine.search.reranking import _spans_calendar_period

pytestmark = pytest.mark.hs_llm_core

# Reference date deliberately inside 2026 so the "current year" case below is a
# real regression guard: the model used to resolve "in 2026" to this date.
EVENT_DATE = datetime(2026, 8, 30)


async def _dated_spans(text: str) -> list[tuple[datetime, datetime]]:
    """Extract `text` and return the (start, end) pair of every dated fact."""
    facts, _, _ = await extract_facts_from_text(
        text=text,
        event_date=EVENT_DATE,
        llm_config=LLMConfig.from_env(),
        agent_name="user",
        config=_get_raw_config(),
    )
    spans = []
    for f in facts:
        if f.occurred_start and f.occurred_end:
            start = date_parser.isoparse(str(f.occurred_start))
            end = date_parser.isoparse(str(f.occurred_end))
            spans.append((start.replace(tzinfo=None), end.replace(tzinfo=None)))
    return spans


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,year",
    [
        ("The user graduated from university in 2015.", 2015),
        ("The user visited Japan in 2023.", 2023),
        # The #3893 case: a coarse date in the *current* year (relative to the
        # event date) must still span the year instead of collapsing onto it.
        ("In 2026 the user attended the Hangzhou Open Source Summit.", 2026),
    ],
)
async def test_year_only_date_spans_the_whole_year(text: str, year: int):
    spans = await _dated_spans(text)
    assert spans, f"expected a dated fact from: {text}"
    matching = [(s, e) for s, e in spans if s.year == year and _spans_calendar_period(s, e)]
    assert matching, (
        f"expected a fact spanning the whole of {year}, got {spans}. "
        "A collapsed coarse date reads as an exact date during recall scoring."
    )
    start, end = matching[0]
    assert (start.month, start.day) == (1, 1)
    assert end.month == 12


@pytest.mark.asyncio
async def test_month_only_date_spans_the_whole_month():
    spans = await _dated_spans("The user moved to Berlin in March 2026.")
    assert spans, "expected a dated fact"
    matching = [(s, e) for s, e in spans if (s.year, s.month) == (2026, 3) and _spans_calendar_period(s, e)]
    assert matching, f"expected a fact spanning all of March 2026, got {spans}"
    start, end = matching[0]
    assert start.day == 1
    assert end.month == 3


@pytest.mark.asyncio
async def test_exact_date_is_not_widened_into_a_period():
    """The instruction must not push precise dates into spans — that would make
    every dated memory look coarse and disable the recency signal wholesale."""
    spans = await _dated_spans("On August 24, 2026, the user switched the UI to the light theme.")
    assert spans, "expected a dated fact"
    same_day = [(s, e) for s, e in spans if s.date() == e.date() == datetime(2026, 8, 24).date()]
    assert same_day, f"expected a same-day span for an exact date, got {spans}"
    assert not _spans_calendar_period(*same_day[0])
