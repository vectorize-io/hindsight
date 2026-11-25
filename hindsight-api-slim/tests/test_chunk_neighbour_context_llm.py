"""A chunk extracted with its neighbours reads a terse reply against them, and takes no facts from them (#6645).

Real-LLM half of test_chunk_neighbour_context.py: whether the model follows the CONTEXT ONLY rule.
"""

import dataclasses
import json
from datetime import datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

_BEFORE = json.dumps(
    {
        "role": "assistant",
        "content": (
            "I found two flats. A) Via Po 12: two bedrooms, €1,450 a month, fourth floor with a lift. "
            "B) San Salvario, Via Nizza 88: one bedroom plus a study, €1,250 a month, no lift. "
            "Separately: your dentist moved your check-up to 3 December."
        ),
    }
)
_CHUNK = json.dumps({"role": "user", "content": "the cheaper one, book a visit for Saturday"})


@pytest.mark.asyncio
async def test_a_reply_is_resolved_from_the_context_and_no_fact_comes_from_it():
    config = dataclasses.replace(_get_raw_config(), retain_chunk_context_tokens=300)
    facts, _, _ = await extract_facts_from_text(
        text=_CHUNK,
        event_date=datetime(2026, 11, 16),
        llm_config=LLMConfig.from_env(),
        config=config,
        context="conversation",
        context_before=_BEFORE,
    )
    await assert_meets_criteria(
        response="\n".join(f"- [{f.fact_type}] {f.fact}" for f in facts),
        criteria=(
            "1) Some fact names the flat the user chose — San Salvario / Via Nizza 88 / €1,250 — rather than "
            "only 'the cheaper one'. Facts may be split; the visit may be in a separate fact. "
            "2) No fact mentions the dentist or a check-up: that was only context, not the user's message."
        ),
        context=f"Context shown before the chunk (read-only): {_BEFORE}\nChunk: {_CHUNK}",
    )
