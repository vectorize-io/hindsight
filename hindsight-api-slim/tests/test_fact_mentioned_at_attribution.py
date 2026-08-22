"""Real-LLM coverage for per-fact source timestamp attribution (issue #2550)."""

import dataclasses
import json
from datetime import datetime, timezone

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


@pytest.mark.asyncio
async def test_source_message_time_is_distinct_from_event_time() -> None:
    """Facts inherit their own record time, while described event dates stay occurred dates."""
    records = [
        {
            "timestamp": "2026-07-05T09:15:00Z",
            "role": "user",
            "content": "PostgreSQL is my favorite database.",
        },
        {
            "timestamp": "2026-07-10T18:30:00+02:00",
            "role": "user",
            "content": "Svelte is my favorite frontend framework.",
        },
        {
            "timestamp": "2026-07-12T08:00:00Z",
            "role": "user",
            "content": "The deployment occurred at 2022-04-03T12:00:00Z.",
        },
    ]
    text = "\n".join(json.dumps(record) for record in records)
    config = dataclasses.replace(
        _get_raw_config(),
        retain_extraction_mode="concise",
        retain_extract_causal_links=False,
    )

    facts, _chunks, _usage = await extract_facts_from_text(
        text=text,
        event_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
        llm_config=LLMConfig.from_env(),
        agent_name="test-agent",
        context="JSONL session where each record's timestamp is its source write time",
        config=config,
    )

    assert facts, "Should extract at least one fact"
    facts_summary = "\n".join(
        f"- mentioned_at={fact.mentioned_at}; occurred_start={fact.occurred_start}; {fact.fact}" for fact in facts
    )
    await assert_meets_criteria(
        response=facts_summary,
        criteria=(
            "The PostgreSQL preference is attributed to source time 2026-07-05T09:15:00Z; "
            "the Svelte preference is attributed to source time 2026-07-10T18:30:00+02:00; "
            "and the deployment fact is attributed to source-write time 2026-07-12T08:00:00Z. "
            "The request-level Event Date of July 1 is not used for those timestamped records. "
            "The preferences do not treat their source timestamps as event occurrence dates. "
            "For the deployment fact, occurred_start is 2022-04-03T12:00:00Z or that complete event timestamp is "
            "clearly retained in the fact text, while mentioned_at remains 2026-07-12T08:00:00Z. "
            "The event timestamp 2022-04-03T12:00:00Z must not be promoted to mentioned_at even though it is also "
            "a complete timezone-bearing timestamp visible in the same source record. Equivalent timezone "
            "normalization is acceptable."
        ),
        context=(
            "Each JSONL record contains its own source timestamp. mentioned_at means when the record directly stating "
            "the fact was written; occurred_start means when the described event happened. The deployment record "
            "intentionally contains both kinds of complete timestamp to test semantic attribution beyond the "
            "source-presence guard."
        ),
        msg=f"Source timestamps must be attributed per fact without replacing event time. Facts: {facts_summary}",
    )
