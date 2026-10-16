"""Real-LLM checks for two reflect prompt rules: the bank mission, and a decision against a later change.

The deterministic halves are elsewhere: that the mission is restated after the question is pinned in
``test_reflect_prompt_builder.py``, and the rule's text in the expected tools prompt. What only a real
model shows is whether it follows them.

* The mission sat only at the top of the system prompt, and lost to the retrieval ladder below it: told
  to read a decision's own record before reporting it, the agent answered from a fresh page that had
  misread the decision (system eval test_09, boltons-budget-history).
* "Latest statement wins" read a later cleanup commit as replacing a decision stated with its reason.
"""

from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.reflect.agent import run_reflect_agent
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

_DECISION = {
    "id": "mem-decision",
    "content": (
        "Commit e4a4097 (2026-03-02): 'MAX_ATTEMPTS is 7 — measured: the upstream recovers within 7 "
        "attempts in 99.8% of outages. Do not round it up to 8 or 10.'"
    ),
    "tags": ["source:git"],
    "mentioned_at": "2026-03-02T10:00:00Z",
}
_CLEANUP = {
    "id": "mem-cleanup",
    "content": "Commit bef5bc4 (2026-05-11): 'perf: simplify retry' — sets MAX_ATTEMPTS = 10.",
    "tags": ["source:git"],
    "mentioned_at": "2026-05-11T09:00:00Z",
}


def _functions(mental_models: list[dict], memories: list[dict]) -> dict[str, AsyncMock]:
    return {
        "search_mental_models_fn": AsyncMock(
            return_value={"count": len(mental_models), "mental_models": mental_models}
        ),
        "read_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
        "search_observations_fn": AsyncMock(return_value={"observations": []}),
        "recall_fn": AsyncMock(return_value={"memories": memories}),
        "expand_fn": AsyncMock(return_value={"memories": []}),
    }


@pytest.mark.asyncio
async def test_the_mission_sends_reflect_past_a_fresh_page_to_the_decision(llm_config):
    """A fresh page that misread the decision would answer alone; the mission says read the record."""
    page = {
        "id": "mm-retry",
        "name": "Retrier",
        "content": "The Retrier's attempts are dynamically capped at the measured budget, replacing the old limit of 7.",
        "relevance": 0.93,
        "is_stale": False,
    }
    functions = _functions([page], [_DECISION, _CLEANUP])

    result = await run_reflect_agent(
        llm_config=llm_config,
        bank_id="test-bank",
        query="What did the project decide about the Retrier's MAX_ATTEMPTS?",
        bank_profile={
            "name": "Test",
            "mission": (
                "Knowledge pages are summaries and can misread a decision's exact values. Before reporting a "
                "rule, recall the commit message or conversation that states it, and report the rule from there."
            ),
        },
        has_mental_models=True,
        budget="low",
        max_iterations=6,
        **functions,
    )

    functions["recall_fn"].assert_awaited()
    await assert_meets_criteria(
        response=result.text,
        criteria="The answer reports that the project decided MAX_ATTEMPTS is 7.",
        context="The page wrongly said the limit of 7 was replaced; commit e4a4097 states MAX_ATTEMPTS is 7.",
    )


@pytest.mark.asyncio
async def test_a_later_cleanup_does_not_replace_a_decision(llm_config):
    """A decision stated with its reason stands against a later change that decides nothing."""
    functions = _functions([], [_DECISION, _CLEANUP])

    result = await run_reflect_agent(
        llm_config=llm_config,
        bank_id="test-bank",
        query="What is the project's rule for the Retrier's MAX_ATTEMPTS?",
        bank_profile={"name": "Test", "mission": ""},
        has_mental_models=False,
        budget="low",
        max_iterations=6,
        **functions,
    )

    await assert_meets_criteria(
        response=result.text,
        criteria=(
            "The answer reports MAX_ATTEMPTS = 7 as the project's standing rule, and reports the later change "
            "to 10 as a departure from it rather than as the rule."
        ),
        context="Commit e4a4097 decided 7 with a measured reason; a later 'perf: simplify retry' commit set 10.",
    )
