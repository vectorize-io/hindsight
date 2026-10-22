"""Reflect that may answer nothing (``abstain``), and searches the caller names (``search_queries``).

A memory watcher beside a coding agent asks reflect, at every step the agent takes, whether
memory has something the agent must hear. Most steps it has not, and the answer must then be
empty and cheap: no LLM call when the decision model already sees nothing that answers. These
tests pin the mechanics with a scripted provider; whether the decisions hold up is the evals' job.
"""

import json
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.cross_encoder import EvidencePick
from hindsight_api.engine.reflect.agent import ReflectNoAnswerError, run_reflect_agent
from hindsight_api.engine.reflect.models import SettledAbstainAnswer
from hindsight_api.engine.response_models import LLMCallResult, LLMToolCall, LLMToolCallResult, TokenUsage
from tests.reflect_script import closing_echo


class _ScriptedProvider:
    """Answers ``call_with_tools`` from a script; records every call and what the closing call read."""

    def __init__(self, scripted: list[LLMToolCallResult], answer: str = "NONE"):
        self._reply = closing_echo(scripted)
        self.tool_choices: list[str] = []
        self.rewrites = 0
        self.last_messages: list[dict] = []
        self.answer = answer
        self.answer_prompts: list[str] = []

    async def call_with_tools(self, *, tool_choice, messages, **_):
        self.tool_choices.append(tool_choice.function_name or tool_choice.mode.value)
        self.last_messages = list(messages)
        return self._reply(tool_choice=tool_choice)

    async def call(self, *, messages, **_):
        prompt = messages[-1]["content"]
        if "## Memories" in prompt:  # the abstain answer, written from what was picked (structured)
            self.answer_prompts.append(prompt)
            reply = (
                {"applies": False, "answer": ""} if self.answer == "NONE" else {"applies": True, "answer": self.answer}
            )
            return LLMCallResult(content=json.dumps(reply), usage=TokenUsage(input_tokens=1, output_tokens=1))
        self.rewrites += 1
        return LLMCallResult(content=prompt, usage=TokenUsage(input_tokens=1, output_tokens=1))


def _done(answer: str, found: bool = True) -> LLMToolCallResult:
    return LLMToolCallResult(
        tool_calls=[
            LLMToolCall(id="d", name="done", arguments={"answer": answer, "answer_found": found, "observation_ids": []})
        ],
        finish_reason="tool_calls",
    )


def _observations() -> AsyncMock:
    return AsyncMock(
        return_value={
            "observations": [
                {"id": "obs-rule", "text": "memory suffixes are binary: K=1024"},
                {"id": "obs-ci", "text": "CI runs on GitHub Actions"},
            ]
        }
    )


async def _reflect(
    provider, *, pick=None, prune=None, sufficient=None, search_observations=None, expand=None, **kwargs
):
    return await run_reflect_agent(
        llm_config=provider,
        bank_id="b",
        query="Does my change go against a recorded decision?",
        bank_profile={"name": "T", "mission": ""},
        search_mental_models_fn=AsyncMock(return_value={"mental_models": []}),
        read_mental_models_fn=AsyncMock(return_value={"mental_models": []}),
        search_observations_fn=search_observations or _observations(),
        recall_fn=AsyncMock(return_value={"memories": []}),
        expand_fn=expand or AsyncMock(return_value={"memories": []}),
        has_mental_models=False,
        include_observations=True,
        include_recall=False,
        budget="low",
        max_iterations=4,
        evidence_is_sufficient_fn=sufficient or AsyncMock(return_value=True),
        prune_evidence_fn=prune or AsyncMock(side_effect=lambda q, texts: [1.0 - i / 10 for i in range(len(texts))]),
        pick_evidence_fn=pick,
        abstain=True,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_nothing_answers_returns_empty_with_no_llm_call():
    provider = _ScriptedProvider([])
    pick = AsyncMock(return_value=EvidencePick(none=0.97, candidates=[0.02, 0.01]))

    result = await _reflect(provider, pick=pick)

    assert result.abstained and result.text == ""
    assert provider.tool_choices == [] and provider.rewrites == 1, "only the search rewrite ran"
    assert [c.detail for c in result.llm_trace if c.scope == "decision_pick"] == [
        "none=0.97 of 2 offered: silent, no LLM"
    ]


@pytest.mark.asyncio
async def test_the_answer_reads_only_what_the_decision_model_picked():
    provider = _ScriptedProvider([], answer="Memory suffixes are binary: K=1024.")
    pick = AsyncMock(return_value=EvidencePick(none=0.2, candidates=[0.78, 0.02]))

    result = await _reflect(provider, pick=pick)

    assert not result.abstained and result.text == "Memory suffixes are binary: K=1024."
    assert result.used_observation_ids == ["obs-rule"], "the note cites the memory it came from"
    (prompt,) = provider.answer_prompts
    assert "K=1024" in prompt and "GitHub Actions" not in prompt, "an unpicked memory never reaches the answer"
    assert provider.tool_choices == [], "one short call, not the agent's closing turn"
    assert [c.scope for c in result.llm_trace] == ["round_query", "decision_pick", "abstain_answer"]
    (pick_call,) = [c for c in result.llm_trace if c.scope == "decision_pick"]
    assert pick_call.detail == "none=0.20, kept 1 of 2 offered: unsure: the reflect LLM judges"


@pytest.mark.asyncio
async def test_the_pick_is_asked_the_question_about_the_context_over_each_searchs_best():
    provider = _ScriptedProvider([])
    # Every search returns the same memories, each in its own order (a bank's whole observation set).
    junk, rule, other = ({"id": i, "text": t} for i, t in (("j", "release notes"), ("r", "units rule"), ("o", "CI")))
    by_query = {
        "boilerplate": {"observations": [junk, other, rule]},
        "units": {"observations": [rule, junk, other]},
    }
    search = AsyncMock(side_effect=lambda q, *a, **k: by_query[q])
    pick = AsyncMock(return_value=EvidencePick(none=0.95, candidates=[0.02, 0.02, 0.01]))

    await _reflect(
        provider,
        pick=pick,
        search_observations=search,
        search_queries=["boilerplate", "units"],
        context="I edited units.py",
    )

    question, context, offered = pick.await_args.args
    assert question == "Does my change go against a recorded decision?" and context == "I edited units.py"
    assert offered == ["release notes", "units rule", "CI"], "each search's best first, in turn, each memory once"


@pytest.mark.asyncio
async def test_an_answer_that_does_not_apply_is_silence_not_text():
    """Below the bar the LLM decides; `applies: false` is an empty answer, not prose."""
    provider = _ScriptedProvider([], answer="NONE")
    pick = AsyncMock(return_value=EvidencePick(none=0.6, candidates=[0.35, 0.05]))

    result = await _reflect(provider, pick=pick)

    assert result.abstained and result.text == ""


@pytest.mark.asyncio
async def test_a_done_reporting_nothing_found_returns_empty_instead_of_text():
    """Without a pick the agent's closing turn answers; its "nothing found" is silence too."""
    provider = _ScriptedProvider([_done("Nothing conflicts: your change follows the decision.", found=False)])

    result = await _reflect(provider, sufficient=AsyncMock(return_value=True))

    assert result.abstained and result.text == ""


@pytest.mark.asyncio
async def test_without_a_pick_not_enough_returns_empty_instead_of_searching_on():
    provider = _ScriptedProvider([])

    result = await _reflect(provider, sufficient=AsyncMock(return_value=False))

    assert result.abstained and result.text == ""
    assert provider.tool_choices == [], "no follow-up LLM turn"


@pytest.mark.asyncio
async def test_an_empty_done_without_abstain_still_fails():
    """The pre-existing contract (#2959): an answer that is not there is an error unless asked for."""
    provider = _ScriptedProvider([_done("")])
    with pytest.raises(ReflectNoAnswerError):
        await run_reflect_agent(
            llm_config=provider,
            bank_id="b",
            query="q",
            bank_profile={"name": "T", "mission": ""},
            search_mental_models_fn=AsyncMock(return_value={"mental_models": []}),
            read_mental_models_fn=AsyncMock(return_value={"mental_models": []}),
            search_observations_fn=_observations(),
            recall_fn=AsyncMock(return_value={"memories": []}),
            expand_fn=AsyncMock(return_value={"memories": []}),
            has_mental_models=False,
            include_recall=False,
            budget="low",
            max_iterations=3,
            evidence_is_sufficient_fn=AsyncMock(return_value=True),
        )


@pytest.mark.asyncio
async def test_search_queries_replace_the_rewrite_and_each_runs():
    provider = _ScriptedProvider([])
    search = _observations()
    pick = AsyncMock(return_value=EvidencePick(none=0.99, candidates=[0.0, 0.0]))

    await _reflect(provider, pick=pick, search_observations=search, search_queries=["parse_quantity units", "disk GB"])

    assert provider.rewrites == 0, "no LLM call before the search"
    assert [c.args[0] for c in search.await_args_list] == ["parse_quantity units", "disk GB"]
    # Both queries returned the same two observations: the pick sees each once.
    _, _, offered = pick.await_args.args
    assert len(offered) == 2


@pytest.mark.asyncio
async def test_a_failed_pick_falls_back_to_the_sufficiency_check():
    provider = _ScriptedProvider([])
    pick = AsyncMock(side_effect=RuntimeError("HTTP 500"))

    result = await _reflect(provider, pick=pick, sufficient=AsyncMock(return_value=False))

    assert result.abstained, "the sufficiency check still decides, and abstains on not enough"


@pytest.mark.asyncio
async def test_the_answer_reads_the_facts_under_the_picked_memory():
    """An observation is a summary: the answer needs the facts it was built from (amended details)."""
    provider = _ScriptedProvider([], answer="Round the prorated quota UP.")
    search = AsyncMock(
        return_value={
            "observations": [{"id": "obs-q", "text": "overage_charge prorates the quota", "source_fact_ids": ["f1"]}],
            "source_facts": {"f1": {"id": "f1", "text": "the prorated quota is rounded up with ceil"}},
        }
    )
    pick = AsyncMock(return_value=EvidencePick(none=0.05, candidates=[0.95]))

    await _reflect(provider, pick=pick, search_observations=search)

    (prompt,) = provider.answer_prompts
    assert "source: the prorated quota is rounded up with ceil" in prompt


@pytest.mark.asyncio
async def test_the_answer_reads_the_source_text_of_the_picked_memory():
    """Source facts are off by default: the picked memory's facts are expanded to their chunks."""
    search = AsyncMock(
        return_value={
            "observations": [{"id": "obs-q", "text": "overage_charge prorates", "source_fact_ids": ["f1", "f2"]}]
        }
    )
    chunk = {"id": "c1", "text": "user: the allowance is rounded UP, never truncated"}
    expand = AsyncMock(
        return_value={"results": [{"memory_id": "f1", "chunk": chunk}, {"memory_id": "f2", "chunk": chunk}]}
    )
    # A pick still to be judged reads the summary first; a settled one, which only words the answer,
    # reads the source first.
    for none, source_first in ((0.5, False), (0.05, True)):
        provider = _ScriptedProvider([], answer="Round the prorated quota UP.")
        pick = AsyncMock(return_value=EvidencePick(none=none, candidates=[0.95]))

        await _reflect(provider, pick=pick, search_observations=search, expand=expand)

        assert expand.await_args.args == (["f1", "f2"], "chunk")
        (prompt,) = provider.answer_prompts
        assert prompt.count("the allowance is rounded UP, never truncated") == 1, "one chunk, once"
        assert (prompt.index("rounded UP") < prompt.index("overage_charge prorates")) is source_first, none


@pytest.mark.asyncio
async def test_an_unreadable_answer_is_silence():
    provider = _ScriptedProvider([])
    provider.call = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda **k: LLMCallResult(
            content="Nothing conflicts."
            if "## Memories" in k["messages"][-1]["content"]
            else k["messages"][-1]["content"],
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )
    )
    pick = AsyncMock(return_value=EvidencePick(none=0.5, candidates=[0.4, 0.1]))

    result = await _reflect(provider, pick=pick)

    assert result.abstained and result.text == ""


@pytest.mark.asyncio
async def test_a_settled_pick_writes_the_answer_then_whether_it_applies():
    """When the decision model has settled it, the LLM writes the answer (no check first) and then labels
    it: an answer it labels as not applying ("Nothing conflicts.") is silence, not a note."""
    for reply, kept in (
        ({"answer": "Memory suffixes are binary: K=1024.", "applies": True}, "Memory suffixes are binary: K=1024."),
        ({"answer": "Nothing conflicts.", "applies": False}, None),
    ):
        provider = _ScriptedProvider([])
        formats: list[object] = []
        systems: list[str] = []

        def answer(reply=reply, provider=provider, formats=formats, systems=systems, **k):
            prompt = k["messages"][-1]["content"]
            if "## Memories" not in prompt:
                return LLMCallResult(content=prompt, usage=TokenUsage(input_tokens=1, output_tokens=1))
            provider.answer_prompts.append(prompt)
            formats.append(k.get("response_format"))
            systems.append(k["messages"][0]["content"])
            return LLMCallResult(content=json.dumps(reply), usage=TokenUsage(input_tokens=1, output_tokens=1))

        provider.call = AsyncMock(side_effect=answer)  # type: ignore[method-assign]
        pick = AsyncMock(return_value=EvidencePick(none=0.02, candidates=[0.97, 0.01]))

        result = await _reflect(provider, pick=pick)

        assert "already judged to answer the question" in provider.answer_prompts[0]
        assert formats == [SettledAbstainAnswer]
        assert "`check`" not in systems[0] and "Then `applies`" in systems[0]
        if kept:
            assert not result.abstained and result.text == kept
        else:
            assert result.abstained and result.text == ""


@pytest.mark.asyncio
async def test_the_writer_model_writes_a_settled_answer_and_only_that():
    """With a writer model set, a settled pick's answer is written by it; a pick the LLM still has to judge
    is answered by the reflect model."""

    def scripted(reply: dict) -> _ScriptedProvider:
        p = _ScriptedProvider([])

        def call(**k):
            prompt = k["messages"][-1]["content"]
            if "## Memories" not in prompt:
                return LLMCallResult(content=prompt, usage=TokenUsage(input_tokens=1, output_tokens=1))
            p.answer_prompts.append(prompt)
            return LLMCallResult(content=json.dumps(reply), usage=TokenUsage(input_tokens=1, output_tokens=1))

        p.call = AsyncMock(side_effect=call)  # type: ignore[method-assign]
        return p

    for none, writes in ((0.02, "writer"), (0.5, "reflect")):
        reflect_llm = scripted({"check": "K differs.", "applies": True, "answer": "From reflect."})
        writer = scripted({"answer": "From writer."})
        pick = AsyncMock(return_value=EvidencePick(none=none, candidates=[0.97, 0.01]))
        result = await _reflect(reflect_llm, pick=pick, settled_llm_config=writer)
        assert result.text == ("From writer." if writes == "writer" else "From reflect."), none
        assert len((writer if writes == "writer" else reflect_llm).answer_prompts) == 1
        assert not (reflect_llm if writes == "writer" else writer).answer_prompts


@pytest.mark.asyncio
async def test_the_http_trace_carries_the_decision_pick_detail(memory, monkeypatch):
    """The pick's detail survives the engine's result and the HTTP response, where a watcher reads it."""
    import httpx

    from hindsight_api.api import create_app
    from hindsight_api.engine.response_models import LLMCallTrace, ReflectResult

    async def reflect_async(**_):
        return ReflectResult(
            text="",
            based_on={},
            abstained=True,
            llm_trace=[LLMCallTrace(scope="decision_pick", duration_ms=3, detail="none=0.97 of 2 offered: silent")],
        )

    monkeypatch.setattr(memory, "reflect_async", reflect_async)
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/default/banks/trace-detail/reflect",
            json={"query": "anything?", "abstain": True, "include": {"tool_calls": {}}},
        )
    assert response.status_code == 200, response.text
    assert response.json()["trace"]["llm_calls"] == [
        {"scope": "decision_pick", "duration_ms": 3, "detail": "none=0.97 of 2 offered: silent"}
    ]
