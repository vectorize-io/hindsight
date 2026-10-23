"""Decision-orchestrated reflect: retrieval runs in parallel rounds, and a decision model decides when to stop.

Agent mode spends one LLM turn per retrieval layer only to write a search string.
With a decision model, reflect rewrites the request into one query, runs those layers in parallel with it,
then asks a decision model whether the evidence already answers it; if not, the LLM's
next query runs as another such round. These tests pin the
mechanics with a scripted provider; whether the answers hold up is the system
evals' job.
"""

from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.reflect.agent import _FRESH_PAGES_GUIDANCE, run_reflect_agent
from hindsight_api.engine.response_models import LLMCallResult, LLMToolCall, LLMToolCallResult, TokenUsage
from tests.reflect_script import closing_echo


class _ScriptedProvider:
    """Answers each ``call_with_tools`` from a script and records the tool choice.

    The tool-less call that rewrites the request into a search query answers ``search_query``,
    or echoes the request when that is None.
    """

    def __init__(self, scripted: list[LLMToolCallResult], search_query: str | None = None):
        self._scripted = scripted
        self._reply = closing_echo(scripted)
        self.tool_choices: list[str] = []
        self.search_query = search_query
        self.distill_prompts: list[str] = []

    async def call_with_tools(self, *, tool_choice, messages, tools, **_):
        self.tool_choices.append(tool_choice.function_name or tool_choice.mode.value)
        self.last_messages = list(messages)
        self.tool_names = [t["function"]["name"] for t in tools]
        return self._reply(tool_choice=tool_choice)

    async def call(self, *, messages, **_):
        self.distill_prompts.append(messages[-1]["content"])
        content = self.search_query if self.search_query is not None else messages[-1]["content"]
        return LLMCallResult(content=content, usage=TokenUsage(input_tokens=1, output_tokens=1))


def _done() -> LLMToolCallResult:
    return LLMToolCallResult(
        tool_calls=[LLMToolCall(id="d", name="done", arguments={"answer": "A", "memory_ids": ["mem-1"]})],
        finish_reason="tool_calls",
    )


def _functions() -> dict[str, AsyncMock]:
    return {
        "search_mental_models_fn": AsyncMock(return_value={"mental_models": [{"id": "mm-1", "content": "page"}]}),
        "read_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
        "search_observations_fn": AsyncMock(return_value={"observations": [{"id": "obs-1", "text": "o"}]}),
        "recall_fn": AsyncMock(return_value={"memories": [{"id": "mem-1", "text": "m"}]}),
        "expand_fn": AsyncMock(return_value={"memories": []}),
    }


async def _reflect(provider: _ScriptedProvider, functions: dict[str, AsyncMock], sufficient: AsyncMock):
    return await run_reflect_agent(
        llm_config=provider,
        bank_id="b",
        query="where does Alice live?",
        bank_profile={"name": "T", "mission": "M"},
        has_mental_models=True,
        budget="mid",
        max_iterations=6,
        evidence_is_sufficient_fn=sufficient,
        **functions,
    )


@pytest.mark.asyncio
async def test_sufficient_evidence_costs_one_llm_call():
    provider = _ScriptedProvider([_done()])
    functions = _functions()
    sufficient = AsyncMock(return_value=True)

    result = await _reflect(provider, functions, sufficient)

    assert result.text == "A"
    # Every layer ran once, with the one rewritten query, and no agent turn wrote a query.
    for name in ("search_mental_models_fn", "search_observations_fn", "recall_fn"):
        assert functions[name].await_count == 1
        assert functions[name].await_args.args[0] == "where does Alice live?"
    assert provider.tool_choices == ["done"], "the answer is the only LLM call"
    # The decision model saw the evidence all three layers returned.
    question, evidence = sufficient.await_args.args
    assert question == "where does Alice live?"
    assert all(marker in evidence for marker in ("page", '"o"', '"m"'))
    assert [c.scope for c in result.llm_trace] == ["round_query", "decision_sufficiency", "closing_done"]


@pytest.mark.asyncio
async def test_insufficient_evidence_runs_another_parallel_round_with_the_llms_query():
    """The LLM writes the next search; it runs on every layer, pruned, and is judged again."""
    follow_up = LLMToolCallResult(
        tool_calls=[LLMToolCall(id="r", name="recall", arguments={"query": "Alice's new address"})],
        finish_reason="tool_calls",
    )
    provider = _ScriptedProvider([follow_up, _done()])
    functions = _functions()
    sufficient = AsyncMock(side_effect=[False, True])
    prune = AsyncMock(side_effect=lambda q, texts: [True] * len(texts))

    result = await run_reflect_agent(
        llm_config=provider,
        bank_id="b",
        query="where does Alice live?",
        bank_profile={"name": "T", "mission": "M"},
        has_mental_models=True,
        budget="mid",
        max_iterations=6,
        evidence_is_sufficient_fn=sufficient,
        prune_evidence_fn=prune,
        **functions,
    )

    assert result.text == "A"
    for name in ("search_mental_models_fn", "search_observations_fn", "recall_fn"):
        assert [c.args[0] for c in functions[name].await_args_list] == ["where does Alice live?", "Alice's new address"]
    assert prune.await_count == 2, "both rounds are pruned"
    assert sufficient.await_count == 2, "the decision model, not the LLM, ends the second round"


@pytest.mark.asyncio
async def test_after_not_enough_the_next_turn_searches_and_its_done_answers():
    """The turn after a "not enough" verdict must search; the done after that round answers.

    Not re-written by the closing step: that lost the recorded decision on the coding-agent eval.
    """
    follow_up = LLMToolCallResult(
        tool_calls=[LLMToolCall(id="r", name="recall", arguments={"query": "Alice's new address"})],
        finish_reason="tool_calls",
    )
    provider = _ScriptedProvider([follow_up, _done()])
    functions = _functions()

    result = await _reflect(provider, functions, AsyncMock(return_value=False))

    assert result.text == "A"
    assert functions["recall_fn"].await_args.args[0] == "Alice's new address", "the follow-up round ran"
    assert provider.tool_choices == ["recall", "required"], "one forced search, then the free turn's done answers"


@pytest.mark.asyncio
async def test_a_done_that_does_not_parse_is_handed_back_and_re_emitted():
    """With no closing step to rewrite it, a malformed ``done`` goes back to the model with the error."""
    follow_up = LLMToolCallResult(
        tool_calls=[LLMToolCall(id="r", name="recall", arguments={"query": "Alice's new address"})],
        finish_reason="tool_calls",
    )
    # A cut-off payload: not repaired, since repair would invent the missing end.
    broken = LLMToolCallResult(
        tool_calls=[LLMToolCall(id="d0", name="done", arguments={"_raw": '{"answer": "A", "memory_ids": ["mem'})],
        finish_reason="length",
    )
    provider = _ScriptedProvider([follow_up, broken, _done()])

    result = await _reflect(provider, _functions(), AsyncMock(return_value=False))

    assert result.text == "A"
    assert provider.tool_choices == ["recall", "required", "required"], (
        "the broken done was refused once, then re-emitted"
    )


@pytest.mark.asyncio
async def test_a_follow_up_turns_other_calls_run_with_its_round():
    """The search becomes a parallel round; an expand in the same turn still runs."""
    follow_up = LLMToolCallResult(
        tool_calls=[
            LLMToolCall(id="r", name="recall", arguments={"query": "Alice's new address"}),
            LLMToolCall(id="e", name="expand", arguments={"memory_ids": ["mem-1"], "depth": "chunk"}),
        ],
        finish_reason="tool_calls",
    )
    provider = _ScriptedProvider([follow_up, _done()])
    functions = _functions()

    await _reflect(provider, functions, AsyncMock(return_value=False))

    assert functions["recall_fn"].await_args.args[0] == "Alice's new address", "the round ran"
    functions["expand_fn"].assert_awaited_once()


@pytest.mark.asyncio
async def test_a_documents_done_is_written_by_the_closing_step():
    """A page update merges only the facts the answer cites; the closing step asks for all of them."""
    follow_up = LLMToolCallResult(
        tool_calls=[LLMToolCall(id="r", name="recall", arguments={"query": "Alice's new address"})],
        finish_reason="tool_calls",
    )
    provider = _ScriptedProvider([follow_up, _done()])

    await run_reflect_agent(
        llm_config=provider,
        bank_id="b",
        query="where does Alice live?",
        bank_profile={"name": "T", "mission": "M"},
        has_mental_models=True,
        budget="mid",
        max_iterations=6,
        evidence_is_sufficient_fn=AsyncMock(return_value=False),
        answer_as_document=True,
        **_functions(),
    )

    assert provider.tool_choices[-1] == "done", "the closing step wrote the document"


@pytest.mark.asyncio
async def test_a_failing_decision_model_falls_back_to_the_agent():
    provider = _ScriptedProvider([_done()])

    result = await _reflect(provider, _functions(), AsyncMock(side_effect=RuntimeError("HTTP 403")))

    assert result.text == "A"
    assert provider.tool_choices == ["required", "done"]


@pytest.mark.asyncio
async def test_a_refusing_decision_model_hands_the_rounds_back_to_the_llm():
    """A refused check reads as "not enough"; repeating parallel rounds on it would run to the limit."""
    follow_up = LLMToolCallResult(
        tool_calls=[LLMToolCall(id="r", name="recall", arguments={"query": "Alice's new address"})],
        finish_reason="tool_calls",
    )
    provider = _ScriptedProvider([follow_up, _done()])
    functions = _functions()
    refusing = AsyncMock(side_effect=RuntimeError("HTTP 400: unanswerable"))

    result = await _reflect(provider, functions, refusing)

    assert result.text == "A"
    assert refusing.await_count == 1, "asked once, then the LLM decides"
    # The follow-up ran as the one search the LLM asked for, not as another parallel round.
    assert functions["search_observations_fn"].await_count == 1
    assert functions["recall_fn"].await_args_list[-1].args[0] == "Alice's new address"


@pytest.mark.asyncio
async def test_pruned_evidence_never_reaches_the_answer():
    provider = _ScriptedProvider([_done()])
    functions = _functions()
    functions["search_observations_fn"] = AsyncMock(
        return_value={
            "observations": [
                {"id": "obs-keep", "text": "Alice lives in Berlin", "source_fact_ids": ["sf-keep"]},
                {"id": "obs-drop", "text": "Bob likes tea", "source_fact_ids": ["sf-drop"]},
            ],
            "source_facts": {
                "sf-keep": {"id": "sf-keep", "text": "moved"},
                "sf-drop": {"id": "sf-drop", "text": "tea"},
            },
        }
    )
    functions["recall_fn"] = AsyncMock(
        return_value={
            "memories": [{"id": "mem-1", "text": "Alice moved to Berlin"}, {"id": "mem-2", "text": "Bob's tea"}],
            "chunks": {"ch-1": {"chunk_text": "Alice, Berlin"}, "ch-2": {"chunk_text": "tea notes"}},
        }
    )

    async def prune(question: str, texts: list[str]) -> list[bool]:
        assert question == "where does Alice live?"
        return ["Alice" in text for text in texts]

    sufficient = AsyncMock(return_value=True)
    result = await run_reflect_agent(
        llm_config=provider,
        bank_id="b",
        query="where does Alice live?",
        bank_profile={"name": "T", "mission": "M"},
        has_mental_models=True,
        budget="mid",
        max_iterations=6,
        evidence_is_sufficient_fn=sufficient,
        prune_evidence_fn=prune,
        **{k: v for k, v in functions.items()},
    )

    evidence = sufficient.await_args.args[1]
    assert "Alice lives in Berlin" in evidence and "Alice moved to Berlin" in evidence and "Alice, Berlin" in evidence
    assert not any(gone in evidence for gone in ("Bob likes tea", "Bob's tea", "tea notes", '"tea"'))
    assert '"moved"' in evidence, "a kept observation keeps its source facts"
    assert result.text == "A"
    assert [c.scope for c in result.llm_trace] == [
        "round_query",
        "decision_prune",
        "decision_sufficiency",
        "closing_done",
    ]


@pytest.mark.asyncio
async def test_a_failing_pruner_keeps_all_evidence():
    provider = _ScriptedProvider([_done()])
    sufficient = AsyncMock(return_value=True)

    await run_reflect_agent(
        llm_config=provider,
        bank_id="b",
        query="q",
        bank_profile={"name": "T", "mission": "M"},
        has_mental_models=True,
        max_iterations=6,
        evidence_is_sufficient_fn=sufficient,
        prune_evidence_fn=AsyncMock(side_effect=RuntimeError("HTTP 500")),
        **_functions(),
    )

    evidence = sufficient.await_args.args[1]
    assert '"o"' in evidence and '"m"' in evidence


def _fresh_page_functions() -> dict[str, AsyncMock]:
    functions = _functions()
    functions["search_mental_models_fn"] = AsyncMock(
        return_value={"mental_models": [{"id": "mm-1", "content": "the page", "is_stale": False}]}
    )
    return functions


@pytest.mark.asyncio
async def test_fresh_pages_that_answer_hide_the_lower_layers():
    """Pages the decision model says answer alone are read without the facts beside them."""
    provider = _ScriptedProvider([_done()])
    sufficient = AsyncMock(return_value=True)

    result = await _reflect(provider, _fresh_page_functions(), sufficient)

    pages_only = sufficient.await_args_list[0].args[1]
    assert "the page" in pages_only and '"o"' not in pages_only and '"m"' not in pages_only
    assert sufficient.await_count == 1, "the pages' verdict is the only check needed"
    assert [c.scope for c in result.llm_trace] == ["round_query", "decision_pages_sufficiency", "closing_done"]


@pytest.mark.asyncio
async def test_the_pages_verdict_reads_only_the_pages_that_came_back_whole():
    """A snippet is the page opening: judging it as an answer would end retrieval before
    the page that holds it was ever read (#5787)."""
    functions = _functions()
    functions["search_mental_models_fn"] = AsyncMock(
        return_value={
            "mental_models": [
                {"id": "mm-1", "content": "the page", "is_stale": False},
                {"id": "mm-2", "snippet": "the opening", "content_chars": 90000, "is_stale": False},
            ]
        }
    )
    sufficient = AsyncMock(return_value=False)

    await _reflect(_ScriptedProvider([_done()]), functions, sufficient)

    pages_only = sufficient.await_args_list[0].args[1]
    assert "the page" in pages_only and "the opening" not in pages_only


@pytest.mark.asyncio
async def test_fresh_pages_that_do_not_answer_keep_the_facts():
    """Fresh is not on topic: pages about the code must not hide a decision in the facts."""
    provider = _ScriptedProvider([_done()])
    # No on the pages alone, yes once the facts are in.
    sufficient = AsyncMock(side_effect=[False, True])

    await _reflect(provider, _fresh_page_functions(), sufficient)

    evidence = sufficient.await_args_list[1].args[1]
    assert "the page" in evidence and '"o"' in evidence and '"m"' in evidence
    # The pages did not answer, so the LLM is not told to answer from them alone.
    assert not any(_FRESH_PAGES_GUIDANCE in str(m.get("content")) for m in provider.last_messages)


@pytest.mark.asyncio
async def test_every_request_is_searched_with_a_rewritten_query():
    """A plugin's 2k-character prompt found nothing and took seconds per search when searched as is."""
    provider = _ScriptedProvider([_done()], search_query="Alice home city")
    functions = _functions()

    result = await _reflect(provider, functions, AsyncMock(return_value=True))

    assert provider.distill_prompts == ["where does Alice live?"]
    for name in ("search_mental_models_fn", "search_observations_fn", "recall_fn"):
        assert functions[name].await_args.args[0] == "Alice home city"
    assert [c.scope for c in result.llm_trace][0] == "round_query"


@pytest.mark.asyncio
async def test_the_query_rewrite_never_answers_empty():
    """The rewrite used to end reflect on NONE, and a local model said it to real questions."""
    provider = _ScriptedProvider([_done()], search_query="NONE")
    functions = _functions()

    result = await _reflect(provider, functions, AsyncMock(return_value=True))

    assert result.text == "A"
    assert functions["recall_fn"].await_count == 1


@pytest.mark.asyncio
async def test_the_search_query_is_written_with_the_callers_context():
    """ "Should I go ahead with it?" names nothing; the context it came with does."""
    provider = _ScriptedProvider([_done()], search_query="billing migration plan")

    await run_reflect_agent(
        llm_config=provider,
        bank_id="b",
        query="should I go ahead with it?",
        context="the billing migration plan",
        bank_profile={"name": "T", "mission": "M"},
        has_mental_models=True,
        budget="mid",
        max_iterations=6,
        evidence_is_sufficient_fn=AsyncMock(return_value=True),
        **_functions(),
    )

    assert "the billing migration plan" in provider.distill_prompts[0]


@pytest.mark.asyncio
async def test_a_failed_search_query_rewrite_searches_the_request_as_written():
    provider = _ScriptedProvider([_done()])
    provider.call = AsyncMock(side_effect=RuntimeError("provider down"))  # type: ignore[method-assign]
    functions = _functions()

    result = await _reflect(provider, functions, AsyncMock(return_value=True))

    assert result.text == "A"
    assert functions["recall_fn"].await_args.args[0] == "where does Alice live?"


@pytest.mark.asyncio
async def test_round_recall_keeps_a_larger_bank_recall_size():
    from dataclasses import replace

    from hindsight_api.engine.reflect.agent import _SHIPPED_DEFAULT_TOOL_LIMITS

    # Above the 16k cap on a size passed as an argument: the bank's default is not capped.
    bank_size = 20_000
    functions = _functions()

    await run_reflect_agent(
        llm_config=_ScriptedProvider([_done()]),
        bank_id="b",
        query="where does Alice live?",
        bank_profile={"name": "T", "mission": "M"},
        has_mental_models=True,
        budget="mid",
        max_iterations=6,
        evidence_is_sufficient_fn=AsyncMock(return_value=True),
        prune_evidence_fn=AsyncMock(side_effect=lambda q, texts: [True] * len(texts)),
        tool_token_limits=replace(_SHIPPED_DEFAULT_TOOL_LIMITS, recall_max_tokens=bank_size),
        **functions,
    )

    assert bank_size in functions["recall_fn"].await_args.args


@pytest.mark.asyncio
@pytest.mark.parametrize("prunes", [True, False])
async def test_round_recall_is_widened_only_for_the_decision_models_cut(prunes):
    """The cut needs a wider slice to choose from; without one, recall keeps the bank's size."""
    from hindsight_api.engine.reflect.agent import _ROUND_RECALL_MAX_TOKENS, _SHIPPED_DEFAULT_TOOL_LIMITS

    functions = _functions()

    await run_reflect_agent(
        llm_config=_ScriptedProvider([_done()]),
        bank_id="b",
        query="where does Alice live?",
        bank_profile={"name": "T", "mission": "M"},
        has_mental_models=True,
        budget="mid",
        max_iterations=6,
        evidence_is_sufficient_fn=AsyncMock(return_value=True),
        prune_evidence_fn=AsyncMock(side_effect=lambda q, texts: [True] * len(texts)) if prunes else None,
        **functions,
    )

    expected = _ROUND_RECALL_MAX_TOKENS if prunes else _SHIPPED_DEFAULT_TOOL_LIMITS.recall_max_tokens
    assert expected in functions["recall_fn"].await_args.args


@pytest.mark.asyncio
async def test_the_free_turns_done_answers_so_no_answer_ready_is_offered():
    """With a decision model the free turn's done is the answer (#5382): nothing to be ready for (#5832)."""
    provider = _ScriptedProvider([_done()])

    await _reflect(provider, _functions(), AsyncMock(return_value=True))

    assert "answer_ready" not in provider.tool_names
    assert "answer_ready" not in provider.last_messages[0]["content"]


@pytest.mark.asyncio
async def test_a_refresh_still_offers_answer_ready():
    """A document goes through the closing step even with a decision model, so its free turn only says it is ready."""
    provider = _ScriptedProvider([_done()])

    await run_reflect_agent(
        llm_config=provider,
        bank_id="b",
        query="where does Alice live?",
        bank_profile={"name": "T", "mission": "M"},
        has_mental_models=True,
        budget="mid",
        max_iterations=6,
        evidence_is_sufficient_fn=AsyncMock(return_value=True),
        answer_as_document=True,
        **_functions(),
    )

    assert "answer_ready" in provider.tool_names
