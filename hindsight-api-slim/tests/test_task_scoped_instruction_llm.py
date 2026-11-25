"""An order for the task at hand is stored as a decision, not a standing preference (#6770).

With no retain mission, "for this UI use the small model, it's just a prototype" came back as
"User prefers the small model", and recall served that preference into unrelated sessions. The
concise guidelines now say to keep what an instruction was for. The control keeps the fix from
over-reaching: a rule the user marks as lasting must stay a lasting rule.

This guards the behaviour; the regression signal is the `scope` category of the retain system
evals. Called directly like this, gemma-4-26b-a4b already words the first case as an instruction
without the fix, while the same input retained through the server came back as "User wants...".
"""

import json
from datetime import datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


async def _extract(user: str, assistant: str) -> str:
    text = json.dumps([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    facts, _, _ = await extract_facts_from_text(
        text=text,
        event_date=datetime(2026, 11, 17, 10, 0),
        llm_config=LLMConfig.from_env(),
        agent_name=None,
        config=_get_raw_config(),
    )
    return "\n".join(f"- [{f.fact_type}] {f.fact}" for f in facts)


@pytest.mark.asyncio
async def test_an_order_for_one_task_is_a_decision_not_a_preference():
    summary = await _extract(
        "for this UI use the small model, not the big one, it's just a prototype",
        "OK, the prototype UI will call the small model.",
    )
    await assert_meets_criteria(
        response=summary,
        criteria=(
            "Some fact records that the user chose the small model for the prototype UI, and no fact words it "
            "as a preference of the user ('prefers', 'likes', 'wants') rather than a decision for that UI."
        ),
        context="The user gave an instruction for one prototype UI, not a lasting preference.",
    )


@pytest.mark.asyncio
async def test_a_rule_marked_as_lasting_stays_a_rule():
    summary = await _extract(
        "from now on, always use the small model for prototypes",
        "Got it, small model for every prototype going forward.",
    )
    await assert_meets_criteria(
        response=summary,
        criteria=(
            "Some fact records, as a standing rule, that the user wants the small model used for prototypes, "
            "and no fact limits it to a single task, prototype or date."
        ),
        context="The user said 'from now on, always', marking the rule as lasting.",
    )
