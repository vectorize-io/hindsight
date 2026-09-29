"""The consolidation prompt gets a real model to emit a well-formed DELETE (#4152).

``deletes[].observation_id`` is required, and before this the prompt never showed
the shape of a delete entry: both worked examples ended in ``"deletes": []`` and the
`deletes` field rule said only *when* to delete. A model that answered with a
reason-only delete had its whole response rejected — creates and updates included.

Whether a given model omits the field is exactly the kind of prompt-following
behaviour MockLLM cannot simulate, so this runs the real batch call. Note what it
can and cannot pin: that a delete is *emitted at all* is a model judgement (rule 7
tells it to be conservative), so the strict assertion is conditional — every delete
that IS emitted must carry a real observation id. The unconditional part is that
the response validated at all, which is the failure #4152 reported.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation.consolidator import _consolidate_batch_with_llm
from hindsight_api.engine.response_models import MemoryFact
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


@pytest.mark.asyncio
async def test_real_model_emits_a_delete_that_names_its_target():
    """A fact that outright supersedes an existing observation."""
    # MemoryFact carries its id and timestamps as plain strings (they come off a
    # JSON row), so build them as strings rather than UUID/datetime objects.
    stale_obs_id = str(uuid.uuid4())
    fact_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    observation = MemoryFact(
        id=stale_obs_id,
        text="Bob is on the beta waitlist, waiting for an invite.",
        fact_type="observation",
        mentioned_at=now.isoformat(),
    )
    memories = [
        {
            "id": fact_id,
            "text": (
                "The beta waitlist was deleted entirely and replaced with open signup; "
                "nobody is on a waitlist any more and the waitlist page now 404s."
            ),
            "tags": [],
            "occurred_start": None,
            "occurred_end": None,
            "mentioned_at": now,
        }
    ]

    # The real batch call: it RAISES on a response that fails schema validation,
    # so reaching the asserts at all is the #4152 regression check.
    result = await _consolidate_batch_with_llm(
        llm_config=LLMConfig.from_env(),
        memories=memories,
        union_observations=[observation],
        union_source_facts={},
        config=_get_raw_config(),
    )

    assert result.failed is False, "the consolidation batch call failed outright"

    # Conditional by design — see the module docstring. What must never happen is a
    # delete that names nothing, or one pointing at an id the prompt never showed.
    for delete in result.deletes:
        assert _UUID_RE.match(delete.observation_id), (
            f"delete carries a malformed observation_id: {delete.observation_id!r}"
        )
        assert delete.observation_id == stale_obs_id, (
            f"delete targets an id that was not in the prompt: {delete.observation_id!r}"
        )

    actions_summary = "\n".join(
        [f"- CREATE: {c.text}" for c in result.creates]
        + [f"- UPDATE {u.observation_id}: {u.text}" for u in result.updates]
        + [f"- DELETE {d.observation_id}: {d.reason}" for d in result.deletes]
    )
    await assert_meets_criteria(
        response=actions_summary or "(no actions)",
        criteria=(
            "The actions do not leave the claim 'Bob is on the beta waitlist' standing as a "
            "current fact: the waitlist observation is either deleted or rewritten to say the "
            "waitlist no longer exists. Any DELETE names an observation id."
        ),
        context=(
            "A consolidation model was given one existing observation, 'Bob is on the beta "
            f"waitlist, waiting for an invite.' (id {stale_obs_id}), and one new fact saying the "
            "beta waitlist was deleted entirely and replaced with open signup."
        ),
    )
