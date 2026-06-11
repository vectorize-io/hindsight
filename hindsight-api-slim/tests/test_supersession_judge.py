"""LLM-judged test for the supersession arbitration prompt (real LLM).

The deterministic mechanics (queue, interval algebra, idempotent writes) live
in ``test_supersession.py``; this module checks the part MockLLM can't: does
the model, given the arbitration prompt, separate true contradictions from
paraphrases and unrelated facts. Per project testing rules the verdict is
evaluated by an independent judge, never hard-asserted index by index.
"""

import pytest

from hindsight_api import LLMConfig
from hindsight_api.engine.retain.supersession import _arbitrate
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


async def test_arbitration_separates_contradictions_from_paraphrases():
    llm_config = LLMConfig.from_env()

    new_fact = "Alice works at Beta Industries as a staff engineer since June 2024."
    candidates = [
        "Alice works at Acme Corporation as a software engineer.",  # contradiction (employer changed)
        "Alice is employed by Beta Industries in a staff engineering role.",  # paraphrase of the new fact
        "Alice enjoys hiking on weekends.",  # unrelated
        "Bob works at Acme Corporation.",  # different subject
    ]

    verdict, _usage = await _arbitrate(llm_config, new_fact, candidates)

    summary = (
        f"new fact: {new_fact}\n"
        + "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
        + f"\nverdict: duplicates={verdict.duplicate_indices}, contradicted={verdict.contradicted_indices}"
    )
    await assert_meets_criteria(
        response=summary,
        criteria=(
            "The verdict marks candidate [0] (Alice at Acme — the changed employer) as contradicted. "
            "Candidate [1] (a paraphrase of the new fact) is classified as duplicate, NOT contradicted. "
            "Candidates [2] (unrelated hobby) and [3] (a different person, Bob) appear in neither list."
        ),
        context=(
            "An arbitration step decides whether existing memory facts are contradicted by a newly "
            "learned fact. False contradictions hide true facts, so unrelated/different-subject facts "
            "must never be flagged."
        ),
    )
