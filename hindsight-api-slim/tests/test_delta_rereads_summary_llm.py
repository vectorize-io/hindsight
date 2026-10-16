"""A delta refresh corrects the opening summary when a value it restates changes (#5228).

The page is the wave-one page gemini-2.5-flash-lite wrote in the system eval
(``hindsight-system-evals/evals/test_10_page_stays_consistent.py``), verbatim: an
opening summary that names the owner and says no owner change was recorded, then
sections that say the same again. Wave two moves the interval and the owner. Without
the re-read line in the task footer the model rewrote every section but the summary,
on this exact request, every time at temperature 0 — the page then said "Brian Lee
owns it" below "Alice Morgan owns it, no change recorded".

The judge also checks the gaps the notice did NOT fill are still recorded: "rewrite
every block about the pump" would pass the first half and erase them.
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


def _section(section_id: str, heading: str, *texts: str) -> Section:
    return Section(
        id=section_id,
        heading=heading,
        level=2,
        blocks=[Block(id=f"{section_id}-{i}", text=text) for i, text in enumerate(texts)],
    )


_WAVE_ONE_PAGE = StructuredDocument(
    sections=[
        _section(
            "preamble",
            "",
            "The Blue Heron pump's maintenance is governed by manual V1, which specifies a filter replacement "
            "interval, designates Alice Morgan as the owner of maintenance, and requires operators to check the "
            "pressure gauge before each shift. No changes to the maintenance owner have been recorded since the "
            "manual's inception.",
        ),
        _section(
            "filter-replacement-interval",
            "Filter Replacement Interval",
            "The filter replacement interval for the Blue Heron pump is every 1,000 operating hours, as specified "
            "in manual V1, which became effective on October 1, 2026. Manual V1 provides only this interval and "
            "does not include details on filter type, part number, or cleaning guidance.",
        ),
        _section(
            "maintenance-ownership",
            "Maintenance Ownership",
            "Alice Morgan is identified as the owner of maintenance for the Blue Heron pump, according to manual "
            "V1. A maintenance audit conducted on October 2, 2026, confirmed that manual V1 is the only maintenance "
            "document issued and that no changes to the maintenance ownership have been recorded.",
        ),
        _section(
            "operating-checks",
            "Operating Checks",
            "Manual V1 mandates that operators of the Blue Heron pump check the pressure gauge before each shift. "
            "No other operating checks are detailed in the available information.",
        ),
        _section(
            "dated-changes",
            "Dated Changes",
            "Manual V1 for the Blue Heron pump became effective on October 1, 2026. A maintenance audit on October "
            "2, 2026, confirmed the manual's status and the recorded maintenance ownership. No other dated changes "
            "to maintenance requirements or ownership have been recorded.",
        ),
        _section("sources", "Sources", "Blue Heron pump maintenance manual V1", "Maintenance audit, October 2, 2026"),
        _section(
            "information-gaps",
            "Information Gaps",
            "The available documentation (manual V1) does not specify the filter type, part number, or cleaning "
            "guidance. Additionally, there are no electrical isolation instructions provided for filter maintenance.",
        ),
    ]
)

_NOTICE_V2 = [
    {
        "id": "v2-interval",
        "type": "world",
        "text": "Blue Heron pump maintenance notice V2, effective 2026-10-05, supplements manual V1: the filter "
        "replacement interval changes from 1,000 to 800 operating hours.",
    },
    {
        "id": "v2-owner",
        "type": "world",
        "text": "From 2026-10-05, under notice V2, Brian Lee replaces Alice Morgan as owner of Blue Heron pump "
        "maintenance.",
    },
    {
        "id": "v2-unchanged",
        "type": "world",
        "text": "Notice V2 leaves the Blue Heron pump's pressure-gauge check unchanged and adds no filter part "
        "number and no electrical isolation procedure.",
    },
]


async def test_the_opening_summary_is_updated_with_the_value_it_restates():
    user_prompt = build_structured_delta_prompt(
        current_document_json=_WAVE_ONE_PAGE.model_dump_json(),
        supporting_facts=_NOTICE_V2,
        source_query=(
            "Describe the Blue Heron pump's current maintenance requirements. Open with a short summary, then "
            "cover the filter replacement interval, who owns maintenance, the operating checks and dated changes, "
            "and end with a Sources section and an Information gaps section."
        ),
    )
    llm = LLMConfig.from_env().with_config(_get_raw_config())

    op_list = await request_delta_operations(
        llm,
        system_prompt=STRUCTURED_DELTA_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        scope="test_delta_rereads_summary",
        response_format=DeltaOperationList,
        skip_validation=True,
        document=_WAVE_ONE_PAGE,
        # The production delta call runs at temperature 0, which is what made the
        # stale summary reproducible on this request.
        temperature=0.0,
    )
    updated = apply_operations(_WAVE_ONE_PAGE, op_list.operations).document
    # The summary is judged on its own: shown the whole page, the judge drifted onto
    # unrelated details (it failed a page for its V1/V2 timeline).
    summary = updated.sections[0]
    await assert_meets_criteria(
        response="\n\n".join(block.text for block in summary.blocks),
        # Stated positively: phrased as "nowhere is X presented as current", the judge
        # failed summaries that mention 1,000 hours as the superseded value.
        criteria=(
            "The paragraph names Brian Lee as the current maintenance owner and does not claim that no owner "
            "change has been recorded. Mentioning Alice Morgan as the earlier owner is fine."
        ),
        context="The opening summary of a pump maintenance page. Ownership moved from Alice Morgan to Brian Lee.",
    )
    await assert_meets_criteria(
        response=render_document(updated),
        criteria="The page says the filter part number and the electrical isolation instructions are not documented.",
        context="A pump maintenance page. None of its sources gives a filter part number or isolation steps.",
    )
