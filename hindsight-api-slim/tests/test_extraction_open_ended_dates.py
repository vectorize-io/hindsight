"""Open-ended events are extracted without placeholder years (#5820).

A model writing "still ongoing" as year 9999 (or "since forever" as year 1) produced a memory
Postgres stores as infinity, which broke temporal recall and consolidation. The parser drops such
values regardless (``test_temporal_infinite_dates.py``); this checks the prompt keeps the model
from emitting them in the first place, on the raw strings it returns.

Real LLM (hs_llm_core). The assertion is structural — a year in the model's own output — so it
is checked directly rather than judged.
"""

from datetime import UTC, datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text

pytestmark = pytest.mark.hs_llm_core

OPEN_ENDED = [
    "The construction of the bridge began in May 2023 and is still ongoing, with no completion date announced.",
    "My gym membership started on 2024-01-01 and never expires.",
    "Since forever, my family has spent every Christmas in Bologna.",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", OPEN_ENDED)
async def test_open_ended_event_has_no_placeholder_year(text):
    facts, _, _ = await extract_facts_from_text(
        text, datetime(2026, 10, 20, tzinfo=UTC), LLMConfig.from_env(), _get_raw_config()
    )
    assert facts
    for fact in facts:
        for value in (fact.occurred_start, fact.occurred_end):
            assert not (value and value[:4] in ("0001", "9999")), f"placeholder date {value!r} in {fact.fact!r}"
