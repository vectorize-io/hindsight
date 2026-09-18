"""A fact must not borrow metadata the text stated about something else (#4457).

`when` / `where` / `who` / `why` used to be required non-null strings. Under
strict structured output every declared property is required, so a model asked
for `when` on a fact the text gives no date for had no legal way to say "not
stated" — and the most plausible string in sight is whatever the surrounding
text mentions. The reported case turned a project's target start date into the
deadline of an unrelated requirement.

Real-LLM test, because the thing under test is what the model does with the
prompt and the schema. MockLLM echoes its input, so it cannot exercise the
pressure that produces the fabrication, and the fields' nullability alone is a
structural fact covered by `test_fact_extraction_nullable_dimensions.py`. That
split is the one CLAUDE.md asks for: mechanics in a fast unit test,
model-following behaviour judged here.

Note what this can and cannot show. It passes when the model is faithful; it is
not a regression guard, because a strong model stays faithful under the old
required-non-null schema too (measured on Qwen3.6-35B under strict schema, which
passed both ways — the reported case was a 9B).
"""

from datetime import datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

# One line carries a date, the next carries a requirement with no date of its
# own. The date is not absent from the document — it is absent from the fact,
# which is the distinction the extractor has to make.
_NOTE = (
    "Fieldwork planning note.\n\n"
    "The project has a target start of September 14.\n"
    "Customer notifications must be completed before fieldwork begins.\n"
    "The equipment audit finished last quarter."
)


@pytest.mark.asyncio
async def test_a_fact_does_not_borrow_a_date_stated_about_another_subject():
    facts, _, _ = await extract_facts_from_text(
        text=_NOTE,
        event_date=datetime(2026, 9, 1),
        llm_config=LLMConfig.from_env(),
        config=_get_raw_config(),
    )

    assert len(facts) > 0, "Should extract at least one fact"

    # Each fact is rendered on its own line and judged on its own. Facts are
    # stored and retrieved separately, so what a reader could infer by putting
    # two of them side by side ("notifications precede fieldwork" + "the project
    # starts September 14") is not a fabrication by the extractor — an earlier
    # version of this criterion allowed that inference and failed on correct
    # output.
    facts_summary = "\n".join(f"- {f.fact}" for f in facts)
    await assert_meets_criteria(
        response=facts_summary,
        criteria=(
            "The fact about the customer notifications carries no date of its own: it does not say "
            "the notifications are due on, by, or around September 14, and gives them no other "
            "specific date or deadline. Judge that fact alone — do not combine it with the separate "
            "fact about the project's target start date, and do not treat a date that appears only "
            "in another fact as part of this one. A fact that simply says the notifications must be "
            "completed before fieldwork begins, with no date attached, meets this criterion."
        ),
        context=(
            "The source note said the project has a target start of September 14, and separately "
            "that customer notifications must be completed before fieldwork begins. The note gave "
            "the notifications no date of their own."
        ),
        msg=f"A fact must not take a date the note stated about another subject. Facts: {[f.fact for f in facts]}",
    )
