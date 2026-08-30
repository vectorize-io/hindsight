"""Real-provider guard for the coarse-date extraction prompt."""

from datetime import datetime, timezone

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


@pytest.mark.asyncio
async def test_year_only_event_does_not_gain_an_invented_month_or_day():
    source = "用户在2026年杭州开源峰会分享了时间感知记忆的主题。"
    facts, _chunks, _usage = await extract_facts_from_text(
        text=source,
        event_date=datetime(2026, 8, 30, tzinfo=timezone.utc),
        llm_config=LLMConfig.from_env(),
        agent_name="test-agent",
        context="峰会记录",
        config=_get_raw_config(),
    )

    assert facts, "The extractor should return at least one fact for the summit statement"
    summary = "\n".join(
        f"fact={fact.fact!r}; occurred_start={fact.occurred_start!r}; "
        f"occurred_end={fact.occurred_end!r}; occurred_precision={fact.occurred_precision!r}"
        for fact in facts
    )
    await assert_meets_criteria(
        response=summary,
        criteria=(
            "At least one extracted fact captures the Hangzhou open-source summit talk as occurring "
            "sometime in 2026 and keeps occurred_precision='year'. Its fact text must not claim that "
            "the source specified January 1, any other month/day, or a day of week. A concrete "
            "occurred_start storage value is acceptable only when the accompanying precision remains 'year'."
        ),
        context=f"The source states only the year, with no month or day: {source}",
    )
