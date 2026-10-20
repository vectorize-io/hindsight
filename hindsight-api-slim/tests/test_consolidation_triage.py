"""Which facts consolidation triage settles, and which it leaves to the LLM.

The judgments themselves are TypeSafe's; what is pinned here is the code around them —
the thresholds, the guards that keep a verdict from applying where it would be wrong,
and that anything uncertain or failed falls through to the LLM. The model's behaviour
is graded end to end by ``hindsight-system-evals`` ``test_09``.
"""

from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web

from hindsight_api.config import DecisionModelConfig
from hindsight_api.engine import decision_model
from hindsight_api.engine.consolidation import consolidator, triage
from hindsight_api.engine.decision_model import DecisionModel, DecisionResponse
from hindsight_api.tracing import LLMSpanRecorder
from tests.aiohttp_stub import stub_server


def _setup(base_url: str = "https://example.invalid") -> DecisionModelConfig:
    return DecisionModelConfig(
        provider="typesafe",
        api_key="key",
        model="jev-latest",
        base_url=base_url,
        timeout=5,
        max_concurrent=2,
        max_request_tokens=26_000,
    )


@dataclass
class _Obs:
    id: str
    text: str
    mentioned_at: str | None = "2026-01-01T00:00:00+00:00"


def _fact(text: str, mentioned_at: str = "2026-02-01T00:00:00+00:00") -> dict[str, Any]:
    return {"id": uuid.uuid4(), "text": text, "mentioned_at": mentioned_at}


def _yes(flag: bool) -> dict[str, Any]:
    return {"probabilities": {"yes": 0.6, "no": 0.4} if flag else {"yes": 0.4, "no": 0.6}}


def _answer(
    relation: str,
    *,
    match: str | None = None,
    covers: bool = False,
    affected: tuple[str, ...] = (),
    keep_as: str = "not_wanted",
):
    # Probabilities near 0.5 on purpose: only the picked answer may count, never its confidence.
    answers: dict[str, Any] = {
        "relation": {"probabilities": {relation: 0.4, "other": 0.3}},
        "keep_as": {"probabilities": {keep_as: 0.4, "other": 0.3}},
    }
    if match is not None:
        answers["match"] = {"probabilities": {match: 0.55, "o9": 0.45}}
        answers[f"covers_{match}"] = _yes(covers)
    for key in ("o0", "o1"):
        answers[f"touches_{key}"] = _yes(key in affected)
    return DecisionResponse.model_validate({"answers": answers, "usage": {"input_tokens": 10}})


async def _run(monkeypatch, facts, candidates, answers, *, entities=None, mission="", may_create=True):
    replies = iter(answers)

    async def fake_ask(model, fact, cands, mission):
        reply = next(replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr(triage, "_ask", fake_ask)
    return await triage.triage_facts(
        facts,
        candidates,
        DecisionModel.from_config(_setup()),
        entities_by_fact=entities or {},
        mission=mission,
        may_create=may_create,
    )


def test_triage_runs_whenever_a_decision_model_is_set_up_unless_the_bank_opts_out(monkeypatch):
    setups = {"consolidation": _setup()}
    monkeypatch.setattr(triage, "get_config", lambda: SimpleNamespace(decision_model_config=lambda op: setups.get(op)))
    bank = SimpleNamespace(consolidation_use_decision_model=True)
    assert triage.triage_model(bank) is not None
    assert triage.triage_model(SimpleNamespace(consolidation_use_decision_model=False)) is None
    setups.clear()
    assert triage.triage_model(bank) is None


@pytest.mark.asyncio
async def test_a_restatement_is_attached_only_when_the_match_is_confirmed(monkeypatch):
    obs = _Obs("obs-1", "The invoice-api service listens on port 8936.")
    confirmed, doubted = _fact("invoice-api accepts traffic on 8936"), _fact("invoice-api is on 8936")
    result = await _run(
        monkeypatch,
        [confirmed, doubted],
        [[obs], [obs]],
        [_answer("restates", match="o0", covers=True), _answer("restates", match="o0", covers=False)],
    )
    assert result.restates == {str(confirmed["id"]): "obs-1"}


@pytest.mark.asyncio
async def test_a_new_fact_is_recorded_as_its_own_sentence(monkeypatch):
    fact = _fact("The billing-api service is written in Kotlin. | Involving: billing-api")
    result = await _run(monkeypatch, [fact], [[]], [_answer("new")])
    assert result.new == {str(fact["id"]): "The billing-api service is written in Kotlin."}


@pytest.mark.asyncio
async def test_a_verdict_that_contradicts_another_answer_goes_to_the_llm(monkeypatch):
    """'new', 'restates' or 'chatter' cannot stand when the same fact dates an observation."""
    obs = _Obs("obs-1", "The ledger-sync service listens on port 8443.")
    facts = [_fact("ledger-sync is on 9443"), _fact("ledger-sync is on 8443"), _fact("ok thanks")]
    result = await _run(
        monkeypatch,
        facts,
        [[obs], [obs], [obs]],
        [
            _answer("new", affected=("o0",)),
            _answer("restates", match="o0", covers=True, affected=("o0",)),
            _answer("chatter", affected=("o0",)),
        ],
    )
    assert not any(result.settled(str(f["id"])) for f in facts)


MISSION = "Record only generalized, industry-level trends. Name no specific company or person."


@pytest.mark.asyncio
async def test_under_a_mission_a_new_fact_is_stored_only_when_it_can_be_kept_as_stated(monkeypatch):
    """#4619: 'Northwind cut inference spend' names a company the mission forbids naming."""
    named, plain, off_topic = (
        _fact("Northwind cut inference spend"),
        _fact("Inference spend is falling across the industry"),
        _fact("Dana likes hiking"),
    )
    result = await _run(
        monkeypatch,
        [named, plain, off_topic],
        [[], [], []],
        [
            _answer("new", keep_as="reworded"),
            _answer("new", keep_as="as_stated"),
            _answer("new", keep_as="not_wanted"),
        ],
        mission=MISSION,
    )
    assert result.new == {str(plain["id"]): "Inference spend is falling across the industry"}


@pytest.mark.asyncio
async def test_under_a_mission_nothing_is_dropped(monkeypatch):
    """#525: a mission that tracks shell commands wants what the built-in rule calls chatter.
    Whether the mission keeps a fact is the LLM's call."""
    fact = _fact("At 14:02 the agent ran pytest, exit 1")
    result = await _run(monkeypatch, [fact], [[]], [_answer("chatter")], mission=MISSION)
    assert not result.settled(str(fact["id"]))


@pytest.mark.asyncio
async def test_no_new_observation_past_a_scope_cap(monkeypatch):
    result = await _run(
        monkeypatch, [_fact("Marta is allergic to shellfish")], [[]], [_answer("new")], may_create=False
    )
    assert result.new == {}


@pytest.mark.asyncio
async def test_two_new_facts_about_one_entity_go_to_the_llm(monkeypatch):
    """Told twice in one batch, they would become sibling observations; the LLM merges them."""
    a, b, c = _fact("Donald told Athena she is sovereign"), _fact("Donald reaffirmed it"), _fact("Bob joined Acme")
    result = await _run(
        monkeypatch,
        [a, b, c],
        [[], [], []],
        [_answer("new"), _answer("new"), _answer("new")],
        entities={str(a["id"]): ["e-donald"], str(b["id"]): ["e-donald"], str(c["id"]): ["e-bob"]},
    )
    assert set(result.new) == {str(c["id"])}


@pytest.mark.asyncio
async def test_chatter_is_skipped(monkeypatch):
    fact = _fact("thanks!")
    result = await _run(monkeypatch, [fact], [[]], [_answer("chatter")])
    assert result.chatter == {str(fact["id"])}


@pytest.mark.asyncio
async def test_a_change_to_an_older_observation_goes_to_the_llm(monkeypatch):
    """A fact that leaves an observation out of date is the LLM's to write, not triage's."""
    older = _Obs("obs-old", "Alice works at Google", mentioned_at="2026-01-01T00:00:00+00:00")
    newer = _Obs("obs-new", "Alice's favourite framework is Svelte", mentioned_at="2026-03-01T00:00:00+00:00")
    fact = _fact("Alice left Google", mentioned_at="2026-02-01T00:00:00+00:00")
    result = await _run(monkeypatch, [fact], [[older, newer]], [_answer("changes", affected=("o0", "o1"))])
    assert not result.settled(str(fact["id"]))


@pytest.mark.asyncio
async def test_a_failed_request_leaves_the_fact_to_the_llm(monkeypatch):
    fact = _fact("anything")
    result = await _run(monkeypatch, [fact], [[]], [RuntimeError("TypeSafe is down")])
    assert not result.settled(str(fact["id"]))


@pytest.mark.asyncio
async def test_an_answer_without_a_relation_leaves_the_fact_to_the_llm(monkeypatch):
    fact = _fact("anything")
    empty = DecisionResponse.model_validate({"answers": {}})
    result = await _run(monkeypatch, [fact], [[]], [empty])
    assert not result.settled(str(fact["id"]))


def test_evidence_never_revives_an_observation_the_llm_deleted():
    """An update wins over a delete of the same observation, so a restatement of a retired
    one must not become an update of it; a merged one's evidence goes to the survivor."""
    retired, merged, survivor = (SimpleNamespace(id=f"obs-{n}", text=f"text {n}") for n in ("r", "m", "s"))
    llm_result = consolidator._BatchLLMResult(
        deletes=[
            consolidator._DeleteAction(observation_id="obs-r"),
            consolidator._DeleteAction(observation_id="obs-m", merged_into="obs-s"),
        ]
    )
    found = triage.TriageResult(restates={"f1": "obs-r", "f2": "obs-m"})
    consolidator._apply_triage(llm_result, found, [retired, merged, survivor])
    assert [(u.observation_id, u.source_fact_ids) for u in llm_result.updates] == [("obs-s", ["f2"])]


@pytest.mark.asyncio
async def test_a_request_is_traced_in_the_shape_every_recorder_takes(monkeypatch):
    calls: list[inspect.BoundArguments] = []
    signature = inspect.signature(LLMSpanRecorder.record_llm_call)
    monkeypatch.setattr(
        decision_model,
        "get_span_recorder",
        lambda: SimpleNamespace(record_llm_call=lambda **kw: calls.append(signature.bind(None, **kw))),
    )

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {"answers": {"relation": {"probabilities": {"new": 0.9}}}, "usage": {"input_tokens": 7}}
        )

    async with stub_server(handler) as base_url:
        await triage._ask(DecisionModel.from_config(_setup(base_url)), _fact("x"), [], "")

    [call] = calls
    assert call.arguments["scope"] == "decision_consolidation_triage"
    assert call.arguments["input_tokens"] == 7
    # The OTel recorder takes it as a forward-compatible extra; the DB recorder stores it as the input.
    assert "questions" in call.arguments["_extra"]["structured_input"]
