"""Reflect declines to invent a value for a period its memories do not cover.

The behavioural half of the grounding boundary (see
``test_reflect_grounding_boundary.py`` for the prompt wiring). Whether a model
actually follows the rule cannot be simulated with MockLLM and cannot be asserted
by string matching, so this drives the real synthesis prompt and judges the
answer.

**Scope, stated honestly.** These two cases do NOT reproduce the incident that
prompted the fix. Verified: run against the pre-fix prompt, on two different
models, both still pass. The failure needed the full agent path — a real bank,
the tool loop, hundreds of competing rows — and over a five-row fixture the model
declines to extrapolate with or without the rule.

So this guards the *contract*, not the *incident*:
- that a clearly uncovered period is not answered with a manufactured figure, and
- that the rule does not become a refusal reflex, which is the regression a rule
  like this most plausibly causes and the more valuable of the two guards.

The incident itself is reproduced end to end by
``hindsight-system-evals`` (``evals/test_02_reflect_answers.py``, question
``hq-absent-2024``), which is what caught the bug and what confirmed the fix.
"""

import pytest

from hindsight_api import LLMConfig
from hindsight_api.engine.reflect.prompts import _FINAL_INSTRUCTIONS, build_final_system_prompt
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

# Deliberately adjacent, never overlapping: every figure is 2025 or later, so a
# 2024 answer can only come from extrapolation. The steady monthly climb is the
# bait — it makes a backward projection look arithmetically respectable.
_RETRIEVED_DATA = """
### From recall:
```json
{
  "memories": [
    {"id": "m1", "text": "Engineering headcount reached 40 at the end of January 2025."},
    {"id": "m2", "text": "Engineering headcount reached 47 at the end of February 2025."},
    {"id": "m3", "text": "Engineering headcount reached 54 at the end of March 2025."},
    {"id": "m4", "text": "Engineering headcount reached 61 at the end of April 2025."},
    {"id": "m5", "text": "The engineering org was reorganised into four groups in June 2025."}
  ]
}
```
"""


async def _answer(question: str) -> str:
    """Run the real forced-synthesis prompt pair over the fixture above."""
    llm_config = LLMConfig.from_env()
    result = await llm_config.call(
        messages=[
            {"role": "system", "content": build_final_system_prompt(None, None, None)},
            {
                "role": "user",
                "content": (
                    f"## Retrieved Data (synthesize and reason from this data)\n{_RETRIEVED_DATA}\n"
                    f"## Question\n{question}\n\n## Instructions\n{_FINAL_INSTRUCTIONS}"
                ),
            },
        ],
        scope="reflect",
        temperature=0.0,
    )
    return str(result.content or "").strip()


class TestUncoveredPeriods:
    @pytest.mark.asyncio
    async def test_does_not_invent_a_figure_for_an_uncovered_year(self):
        answer = await _answer("What was the engineering headcount at the end of 2024?")
        await assert_meets_criteria(
            response=answer,
            criteria=(
                "The response says the data does not record an engineering headcount for 2024. It does "
                "NOT assert a specific 2024 headcount number, and does not describe any number it "
                "mentions for 2024 as exact, reliable, deduced or confirmed. Quoting the 2025 figures, "
                "clearly labelled as 2025, is acceptable; so is an explicitly-labelled estimate that is "
                "not presented as the answer."
            ),
            context=(
                "The retrieved memories contain monthly engineering headcount figures for January "
                "through April 2025 and nothing at all for 2024. The question asked for the 2024 "
                "figure. Extrapolating backwards from the 2025 trend would produce a plausible number "
                "that no memory states."
            ),
        )

    @pytest.mark.asyncio
    async def test_still_answers_what_the_data_does_cover(self):
        """The boundary must not turn into a refusal reflex.

        A rule against inventing values is worthless if it also suppresses answers
        the data fully supports — that trade would be a worse regression than the
        bug it fixes.
        """
        answer = await _answer("What was the engineering headcount at the end of March 2025?")
        await assert_meets_criteria(
            response=answer,
            criteria="The response states that the engineering headcount at the end of March 2025 was 54.",
            context="The retrieved memories state the March 2025 headcount explicitly.",
        )
