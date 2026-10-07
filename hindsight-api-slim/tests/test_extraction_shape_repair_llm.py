"""The shape correction sent on a retry actually works on a real model (#5280).

When an OpenAI-compatible relay accepts ``response_format`` and silently drops it,
the model answers in a schema of its own invention and every fact is discarded. The
retry then carries ``_shape_repair_message``, which names the keys the model used and
the keys the schema allows.

That text is a prompt, and whether a model *acts* on it is model-following behaviour —
MockLLM echoes its input, so a mock test proves only that the string was assembled
(that half is covered fast in ``test_fact_extraction_retry.py``). This one asks a real
model the question that matters: given its own drifted answer and the correction, does
it re-emit the same facts under the right keys? If it does not, the whole mechanism is
pure cost and the loop still cannot converge.

Judged, not string-matched: the model may legitimately word the fact text differently
or fill the optional dimensions, and only the key names and preserved content are the
contract.
"""

import json

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import (
    FactExtractionResponseNoCausal,
    _expected_fact_keys,
    _shape_repair_message,
)
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


# The exact shape the reporter's endpoint produced, from the issue's raw-response dump.
_DRIFTED_ANSWER = {
    "facts": [
        {"type": "person_role", "subject": "Maria", "predicate": "is", "object": "head of operations"},
        {
            "type": "action",
            "actor": "Maria",
            "action": "migrated",
            "object": "the billing service",
            "target": "Frankfurt",
            "date": "2026-10-06",
        },
    ]
}


@pytest.mark.asyncio
async def test_a_real_model_re_emits_its_facts_under_the_required_keys():
    config = _get_raw_config()
    llm_config = LLMConfig.from_env()

    expected_keys = _expected_fact_keys(FactExtractionResponseNoCausal)
    correction = _shape_repair_message(
        {k for fact in _DRIFTED_ANSWER["facts"] for k in fact},
        expected_keys,
    )
    assert correction is not None

    # Replays the failing exchange: the extraction request, the model's drifted answer,
    # then the correction as its own user turn — exactly how the retry sends it. No
    # response_format, because the endpoint this exists for ignores it; the correction
    # has to carry the shape on its own.
    result = await llm_config.call(
        messages=[
            {"role": "system", "content": "Extract significant facts from the text the user gives you."},
            {
                "role": "user",
                "content": "Maria is the head of operations. She migrated the billing service to Frankfurt on 2026-10-06.",
            },
            {"role": "assistant", "content": json.dumps(_DRIFTED_ANSWER, ensure_ascii=False)},
            {"role": "user", "content": correction},
        ],
        scope="retain_extract_facts",
        temperature=config.llm_temperature_retain,
        max_completion_tokens=2048,
    )

    await assert_meets_criteria(
        response=str(result.content),
        criteria=(
            "The response is a JSON object with a top-level 'facts' array. Every fact in it "
            "carries a non-empty 'what' key holding the statement, and uses NO keys outside: "
            f"{', '.join(expected_keys)} — in particular no 'subject', 'predicate', 'object', "
            "'actor', 'action', 'target' or 'date'. Both original facts survive: that Maria is "
            "the head of operations, and that she migrated the billing service to Frankfurt."
        ),
        context=(
            "A model had answered with facts keyed subject/predicate/object, which the extractor "
            "cannot read. It was then sent a correction naming the keys it used and the keys the "
            "schema allows, and asked to re-emit the same facts. This is that re-answer."
        ),
        msg="The shape correction did not get the model to re-emit under the schema's keys",
    )
