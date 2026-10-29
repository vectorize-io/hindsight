"""Reflect and a page refresh on the configured provider: the cheap check that it works at all.

Every provider makes reflect's tool calls the same way, as replies to a response schema
(``providers/tools_via_schema.py``), so what a provider must do for reflect is follow a
schema and fit the context. This runs the real agent loop on whichever provider the
environment configures — the CI matrix runs it once per provider — with the searches
stubbed, so a case is two to four LLM calls and no database:

* reflect answers from what its search returned and cites it;
* a page refresh (``answer_as_document``) closes with a markdown page split into sections;
* with a decision model that answers "not enough", the forced follow-up search runs.
  A model that ignored a forced call answered it with the stop signal instead.
"""

from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.reflect.agent import run_reflect_agent

pytestmark = pytest.mark.hs_llm_mat

_FACTS = [
    {"id": "mem-oncall", "content": "Rainer Moser is the on-call engineer for the payments service this week."},
    {"id": "mem-pager", "content": "Payments incidents are paged through the #pay-oncall channel."},
]
_QUESTION = "Who is on call for the payments service this week, and where are incidents paged?"


def _functions() -> dict:
    return {
        "search_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
        "read_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
        "search_observations_fn": AsyncMock(return_value={"observations": []}),
        "recall_fn": AsyncMock(return_value={"memories": _FACTS}),
        "expand_fn": AsyncMock(return_value={"memories": []}),
    }


async def _reflect(llm_config, functions: dict, **kwargs):
    return await run_reflect_agent(
        llm_config=llm_config,
        bank_id="smoke",
        query=_QUESTION,
        bank_profile={"name": "Smoke", "mission": ""},
        has_mental_models=False,
        include_observations=False,
        include_recall=True,
        budget="low",
        max_iterations=5,
        **functions,
        **kwargs,
    )


async def test_reflect_answers_from_its_search(llm_config):
    functions = _functions()
    result = await _reflect(llm_config, functions)

    functions["recall_fn"].assert_awaited()
    assert "Rainer" in result.text, result.text
    assert set(result.used_memory_ids) & {"mem-oncall", "mem-pager"}, result.used_memory_ids


async def test_page_refresh_writes_a_markdown_page(llm_config):
    functions = _functions()
    result = await _reflect(llm_config, functions, answer_as_document=True)

    assert result.document is not None and result.document.sections, result.text
    assert "Rainer" in result.text, result.text


async def test_forced_follow_up_search_runs(llm_config):
    """The decision model's "not enough" forces one more search, and the model makes it."""
    functions = _functions()
    result = await _reflect(
        llm_config,
        functions,
        evidence_is_sufficient_fn=AsyncMock(return_value=False),
        prune_evidence_fn=AsyncMock(side_effect=lambda query, texts: [True] * len(texts)),
    )

    assert functions["recall_fn"].await_count >= 2, "the forced follow-up search was skipped"
    assert result.text
