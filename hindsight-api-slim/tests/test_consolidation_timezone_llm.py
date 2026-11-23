"""#6441: an observation keeps the local date of a fact, not the UTC one.

A fact at 07:10 on 8 Nov in Shanghai is 23:10 on 7 Nov in UTC. Shown the UTC time, every run on
gemini-2.5-flash-lite wrote "on 7 November" into the observation; shown the bank's zone, "on 8
November". Which date the model writes is prompt-following, so a judge grades it.
``test_prompt_timezone.py`` pins what the prompt shows.
"""

import dataclasses
import uuid
from datetime import UTC, datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation.consolidator import _consolidate_batch_with_llm
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


@pytest.mark.asyncio
async def test_observation_text_uses_the_bank_local_date():
    config = dataclasses.replace(_get_raw_config(), default_timezone="Asia/Shanghai")
    when = datetime(2026, 11, 7, 23, 10, tzinfo=UTC)  # 07:10 on 8 November in Shanghai
    fact = {
        "id": uuid.uuid4(),
        "text": "Ming said the model download reached 57% when he checked it this morning.",
        "tags": [],
        "occurred_start": when,
        "occurred_end": when,
        "mentioned_at": when,
    }
    result = await _consolidate_batch_with_llm(
        llm_config=LLMConfig.from_env(),
        memories=[fact],
        union_observations=[],
        union_source_facts={},
        config=config,
    )
    assert result.failed is False
    texts = [c.text for c in result.creates] + [u.text for u in result.updates]
    assert texts, "the fact produced no observation"
    await assert_meets_criteria(
        response="\n".join(f"- {t}" for t in texts),
        criteria=(
            "No observation says the event happened on 7 November. If an observation states a date "
            "or a time of day, it is 8 November and/or about 07:10 in the morning (local time)."
        ),
        context=(
            "The fact happened at 07:10 on 8 November 2026 local time in Asia/Shanghai, which is "
            "23:10 on 7 November in UTC. The bank's timezone is Asia/Shanghai."
        ),
    )
