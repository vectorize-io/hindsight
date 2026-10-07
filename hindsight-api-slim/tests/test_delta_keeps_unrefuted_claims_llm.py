"""A delta refresh keeps a claim the new batch does not refute (#5272).

The system eval behind it (test_01 ``hq-release-prod``) builds a release page in two
waves. Wave one records that release 0.9.3 reached production on 18 March 2026; wave
two brings the other releases, including 0.9.6's *canary* promotion that same day.
Nothing in wave two says anything about 0.9.3. The ops call used to also see the
refresh's synthesis, written from wave two alone, which said "no release was deployed
to production on 18 March" — and gemini-3.1-flash-lite and qwen3.8-flash both
overwrote the page with it. The synthesis is no longer sent; this runs the real prompt
on the wave-one page the eval actually produced.

Known gap, not asserted here: on a bare one-line page ("Release 0.9.3 was deployed to
production on 2026-03-18." and nothing else) both models still infer "nothing shipped
on the 18th" from 0.9.2 (16th) and 0.9.4 (20th) bracketing the date. A prompt rule
with a neutral example did not move that (qwen3.8-flash 0/3).
"""

from __future__ import annotations

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.reflect.delta_ops import DeltaOperationList, apply_operations, request_delta_operations
from hindsight_api.engine.reflect.prompts import STRUCTURED_DELTA_SYSTEM_PROMPT, build_structured_delta_prompt
from hindsight_api.engine.reflect.structured_doc import Block, Section, StructuredDocument, render_document
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

#: Wave two of the eval corpus: the even releases, staging -> canary -> production.
_SCHEDULE = {
    "0.9.0": ("2026-02-03", "2026-02-10", "2026-02-17"),
    "0.9.2": ("2026-03-02", "2026-03-09", "2026-03-16"),
    "0.9.4": ("2026-03-06", "2026-03-13", "2026-03-20"),
    "0.9.6": ("2026-03-09", "2026-03-18", "2026-04-22"),  # canary ON the page's date
    "0.9.8": ("2026-05-04", "2026-05-11", "2026-05-18"),
}


def _wave_two() -> list[dict[str, str]]:
    facts = []
    for version, (staging, canary, production) in _SCHEDULE.items():
        for env, verb, date in (
            ("stg", "deployed to staging", staging),
            ("can", "promoted to canary", canary),
            ("prod", "deployed to production", production),
        ):
            facts.append(
                {"id": f"{version}-{env}", "text": f"Release {version} was {verb} on {date}.", "type": "world"}
            )
    return facts


async def test_a_batch_without_the_release_does_not_erase_it():
    document = StructuredDocument(
        sections=[
            Section(
                id="answer",
                heading="Production deployment on 18 March 2026",
                level=2,
                # Verbatim the wave-one page qwen3.8-flash wrote in the eval run that failed.
                blocks=[
                    Block(id="b1", text="Release 0.9.3 was deployed to production on 2026-03-18."),
                    Block(
                        id="b2",
                        text="Its path through the pipeline ahead of that promotion:\n"
                        "- Staging: 2026-03-04\n- Canary: 2026-03-11\n- Production: 2026-03-18",
                    ),
                    Block(
                        id="b3",
                        text="For context, this sits between Release 0.9.1 (production 2026-02-24) and "
                        "Release 0.9.5 (production 2026-04-15).",
                    ),
                ],
            )
        ]
    )
    user_prompt = build_structured_delta_prompt(
        current_document_json=document.model_dump_json(),
        supporting_facts=_wave_two(),
        source_query="Which release was deployed to production on 18 March 2026?",
    )
    llm = LLMConfig.from_env().with_config(_get_raw_config())

    op_list = await request_delta_operations(
        llm,
        system_prompt=STRUCTURED_DELTA_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        scope="test_delta_keeps_claim",
        response_format=DeltaOperationList,
        skip_validation=True,
        document=document,
    )
    page = render_document(apply_operations(document, op_list.operations).document)

    await assert_meets_criteria(
        response=page,
        criteria=(
            "The page still states that release 0.9.3 was deployed to production on 18 March 2026, and does "
            "NOT claim that no release reached production that day or that another release did."
        ),
        context=(
            "The page said release 0.9.3 reached production on 2026-03-18. The new facts cover releases 0.9.0, "
            "0.9.2, 0.9.4, 0.9.6 and 0.9.8 only; 0.9.6 was promoted to canary (not production) on 2026-03-18. "
            "None of them mentions 0.9.3."
        ),
    )
