"""The fact-by-fact consolidation prompt gets a real model to keep observations true.

The prompt asks for one entry per new fact, judging every observation
that mentions it. Whether a model judges a lookalike name as a different subject, ends every
observation a job change makes untrue (not only the closest), and leaves a count alone for a
named member is prompt-following that MockLLM cannot simulate — so this runs the real batch
call on one batch holding all three, and judges the writes it turns into.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation.consolidator import _consolidate_batch_with_llm
from hindsight_api.engine.response_models import MemoryFact
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


@pytest.mark.asyncio
async def test_real_model_ends_what_a_fact_makes_untrue_and_leaves_lookalikes_alone():
    earlier = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    now = datetime.now(timezone.utc)
    observations = {
        str(uuid.uuid4()): text
        for text in (
            "Alice works at Google.",
            "Alice commutes to the Google office in Zurich three days a week.",
            "billing-sink listens on port 8336.",
            "The user has 2 dogs.",
        )
    }
    pool = [
        MemoryFact(id=oid, text=t, fact_type="observation", mentioned_at=earlier) for oid, t in observations.items()
    ]
    facts = (
        "Alice left Google in April and now works at Microsoft.",
        "ledger-sink listens on port 8444.",
        "The user has a dog named Rex.",
    )
    memories = [{"id": uuid.uuid4(), "text": t, "tags": [], "mentioned_at": now} for t in facts]

    result = await _consolidate_batch_with_llm(
        llm_config=LLMConfig.from_env(),
        memories=memories,
        union_observations=pool,
        union_source_facts={},
        config=_get_raw_config(),
        closest={str(m["id"]): list(observations) for m in memories},
    )

    assert result.failed is False, "the consolidation batch call failed outright"
    writes = "\n".join(
        [f"- CREATE: {c.text}" for c in result.creates]
        + [f"- UPDATE of '{observations.get(u.observation_id, '?')}' -> '{u.text}'" for u in result.updates]
        + [f"- DELETE of '{observations.get(d.observation_id, '?')}'" for d in result.deletes]
        # Listed explicitly: with only the writes, the judge read a new ledger-sink observation as
        # an update of the untouched billing-sink one.
        + [
            f"- UNCHANGED: '{text}'"
            for oid, text in observations.items()
            if oid not in {u.observation_id for u in result.updates} | {d.observation_id for d in result.deletes}
        ]
    )
    await assert_meets_criteria(
        response=writes or "(no writes)",
        criteria=(
            "1. Both Alice observations — her job at Google AND her commute to the Google office — are "
            "updated so neither says she is still at Google (a past-tense record is fine), and an "
            "observation says she works at Microsoft. "
            "2. The billing-sink observation is not updated or deleted; ledger-sink's port 8444 is "
            "recorded in a new observation. "
            "3. The '2 dogs' observation is not changed to another count; Rex is recorded in its own observation."
        ),
        context=("Existing observations: " + "; ".join(observations.values()) + ". New facts: " + " ".join(facts)),
    )
