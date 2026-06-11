"""LLM-judged test for the gray-band entity-arbitration prompt (real LLM).

The deterministic mechanics live in ``test_entity_gray_band.py``; this checks
the model-following part: given co-occurrence context, the arbiter merges a
compatible partial name and refuses an incompatible one — judged, not
hard-asserted (per project testing rules).
"""

import pytest

from hindsight_api import LLMConfig
from hindsight_api.engine.entity_arbitration import GrayBandCandidate, GrayBandCase, arbitrate_gray_band
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


async def test_arbiter_uses_context_and_name_compatibility():
    llm_config = LLMConfig.from_env()

    cases = [
        # Compatible partial name with shared context — should merge.
        GrayBandCase(
            mention="Bob",
            nearby_names=("Alice", "Project Alpha"),
            candidates=(
                GrayBandCandidate(
                    entity_id="bob-smith",
                    canonical_name="Bob Smith",
                    cooccurring_names=("Alice", "Project Alpha"),
                ),
                GrayBandCandidate(
                    entity_id="bobby-tables",
                    canonical_name="Bobby Tables",
                    cooccurring_names=("SQL injection", "xkcd"),
                ),
            ),
        ),
        # Incompatible name despite a shared social circle — should refuse.
        GrayBandCase(
            mention="Bob",
            nearby_names=("Alice",),
            candidates=(GrayBandCandidate(entity_id="rob", canonical_name="Rob", cooccurring_names=("Alice",)),),
        ),
    ]

    chosen = await arbitrate_gray_band(llm_config, cases)

    summary = (
        "case 0: mention 'Bob' nearby [Alice, Project Alpha]; "
        "candidates: [0] 'Bob Smith' (co-occurs Alice, Project Alpha), [1] 'Bobby Tables' (co-occurs SQL injection)\n"
        "case 1: mention 'Bob' nearby [Alice]; candidates: [0] 'Rob' (co-occurs Alice)\n"
        f"arbiter chose: {chosen}"
    )
    await assert_meets_criteria(
        response=summary,
        criteria=(
            "Case 0 resolves to 'bob-smith' (compatible partial name with matching context), not "
            "'bobby-tables'. Case 1 is absent from the choices (or explicitly none): 'Bob' and 'Rob' "
            "are different names, and shared acquaintances alone must not merge them."
        ),
        context=(
            "An entity-resolution arbiter decides whether an ambiguous mention refers to an existing "
            "entity. False merges corrupt both entities' histories, so incompatible names must be "
            "refused even with overlapping context."
        ),
    )
