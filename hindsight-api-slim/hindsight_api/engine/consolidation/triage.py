"""Decide the easy consolidation cases with a decision model before the LLM sees them.

Most facts reaching a bank fall in one of three easy cases: they restate what an
observation already says, they add something no observation holds yet (a new entity,
a new attribute of a known one), or they are chatter. None needs writing: attach the
fact as evidence, record the fact as its own observation, or record nothing. A typed
judgment tells them apart in under a second, so those facts never reach the LLM. Only
a fact that affects what an existing observation says — a change, a correction, a
count, a reference it resolves, an older state — needs the LLM's rewrite, and every
fact whose answers disagree goes there too.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from ...config import DecisionModelConfig, get_config
from ...tracing import get_span_recorder
from ..decision_model import DecisionModel, DecisionQuestion, DecisionResponse

if TYPE_CHECKING:
    from ..response_models import MemoryFact

logger = logging.getLogger(__name__)

#: Candidates offered per fact: the head of the fact's own recall, which is ranked.
MAX_CANDIDATES = 10
#: Every judgment is a choice and only the picked answer counts. Previously Nouls and per-answer
#: confidence bars (0.8-0.95) decided: tuned to jev-latest, they let a hedging model such as
#: hindsight-decision (answers 0.3-0.7) settle nothing, and needed retuning per model and per
#: question. Instead, a verdict only stands when the answers agree with each other (see
#: ``triage_facts``), and everything else goes to the LLM.
#: The fact and each observation carry their dates, so the judgment can tell a newer statement
#: from an older one and an event's time from when it was told.
_RELATION = {
    "restates": (
        "Some existing observation already says what `new_fact` says, about the same entity, even if in other "
        "words: a paraphrase, a synonym, a shorter or vaguer version of it. Storing `new_fact` would add nothing."
    ),
    "new": (
        "`new_fact` says something no existing observation says: a new entity, or a new attribute of a known "
        "one. It does not touch anything an observation says: it does not change it, and does not make a vague "
        "part of it specific."
    ),
    "changes": (
        "`new_fact` touches what an existing observation says: it changes, corrects, reverses or retires it; "
        "updates a count, member, status or value in it; makes a vague part of it specific (it names who, "
        "where or what the observation only hints at, like 'her home country'); or tells an earlier state of it."
    ),
    "chatter": (
        "`new_fact` is passing chatter (a greeting, thanks, a momentary activity, what is on a screen right now) "
        "with no lasting knowledge in it."
    ),
}
_MATCH = "Which existing observation is about the same entity and attribute as `new_fact`?"
_COVERS = "Does `existing_observations.{key}` already say what `new_fact` says, even in other words?"
_COVERS_YES = (
    "Everything `new_fact` says is already in `existing_observations.{key}`, for the same entity, possibly paraphrased."
)
_COVERS_NO = "`new_fact` says something `existing_observations.{key}` does not, or is about a different entity."
#: Asked as "should it be reworded" rather than "is it out of date": a fact that names what an
#: observation only hints at ("grew up in Sweden" for "her home country") makes nothing wrong, and
#: the narrower question let it through as new (eval `resolve-reference`).
_TOUCHES = "Knowing `new_fact`, should `existing_observations.{key}` now be worded differently or more precisely?"
_TOUCHES_YES = (
    "`new_fact` makes something in `existing_observations.{key}` out of date or wrong, or tells what a vague "
    "part of it refers to (who 'he' is, where 'her home country' is, what 'the project' is), so the "
    "observation should now say it."
)
_TOUCHES_NO = (
    "`existing_observations.{key}` stays right and complete as it is; `new_fact` is about something else, or "
    "adds only a separate detail."
)
#: Asked only under a bank's own mission. Previously two yes/no questions ("is it wanted", "does it
#: break a wording rule as written"): the second assumed the fact is stored word for word, which
#: nothing told the model. One choice of how the mission would keep it answers both.
_KEEP_AS = (
    "`mission` says what this memory keeps and how it words what it keeps. How should the knowledge in "
    "`new_fact` be kept?"
)
_KEEP_AS_OPTIONS = {
    "as_stated": (
        "`mission` wants this knowledge, and `new_fact` as written already follows every rule in `mission`: it "
        "can be kept exactly as it is."
    ),
    "reworded": (
        "`mission` wants something from `new_fact`, but `new_fact` as written breaks a rule in `mission` (it "
        "names a company or person `mission` says not to name, is more specific than `mission` allows, or is "
        "not in the form `mission` requires), so it must be rewritten first."
    ),
    "not_wanted": "`mission` does not ask to keep anything in `new_fact`.",
}


@dataclass
class TriageResult:
    #: fact id -> observation id the fact restates; attach it as evidence.
    restates: dict[str, str] = field(default_factory=dict)
    #: fact id -> the fact's own sentence; record it as a new observation.
    new: dict[str, str] = field(default_factory=dict)
    #: fact ids that are chatter; record nothing.
    chatter: set[str] = field(default_factory=set)
    input_tokens: int = 0
    seconds: float = 0.0

    def settled(self, fact_id: str) -> bool:
        return fact_id in self.restates or fact_id in self.new or fact_id in self.chatter


@lru_cache(maxsize=4)
def _client(setup: DecisionModelConfig) -> DecisionModel:
    # One client per setup for the process, so its connection pool and concurrency cap are shared.
    return DecisionModel.from_config(setup)


def triage_model(config: Any) -> DecisionModel | None:
    """The decision model that triages this bank's facts, or None to send every fact to the LLM.

    On whenever the server has a decision model for consolidation (HINDSIGHT_API_DECISION_MODEL_*,
    with HINDSIGHT_API_CONSOLIDATION_DECISION_MODEL_* overrides); a bank opts out with
    ``consolidation_use_decision_model``.
    """
    # getattr: callers may pass a partial config (the batch tests pass a bare object).
    if not getattr(config, "consolidation_use_decision_model", False):
        return None
    setup = get_config().decision_model_config("consolidation")
    return _client(setup) if setup is not None else None


def _sentence(fact: dict[str, Any]) -> str:
    # Facts render as "<sentence> | When: … | Involving: …"; only the sentence is prose.
    return str(fact.get("text") or "").split(" | ")[0].strip()


def _yes_no(instructions: str, yes: str, no: str) -> DecisionQuestion:
    """One choice with two options; only the picked one counts."""
    return DecisionQuestion(type="choice", instructions=instructions, criteria={"yes": yes, "no": no})


def _day(value: Any) -> str | None:
    return str(value)[:10] if value else None


def _dated(text: str, item: Any, *, stated: str) -> dict[str, Any]:
    """``text`` with the dates it carries: when it happened (``occurred_*``) and when it was told."""
    get = item.get if isinstance(item, dict) else lambda key: getattr(item, key, None)
    dates = {
        "occurred_start": _day(get("occurred_start")),
        "occurred_end": _day(get("occurred_end")),
        stated: _day(get("mentioned_at")),
    }
    return {"text": text, **{k: v for k, v in dates.items() if v}}


async def _ask(
    model: DecisionModel, fact: dict[str, Any], candidates: list[MemoryFact], mission: str
) -> DecisionResponse:
    keys = [f"o{i}" for i in range(len(candidates))]
    questions: dict[str, DecisionQuestion] = {
        "relation": DecisionQuestion(
            type="choice",
            instructions="How does `new_fact` relate to `existing_observations`?",
            criteria=dict(_RELATION),
        )
    }
    if len(keys) > 1:
        # A choice needs two options; with one candidate its own covers question decides alone.
        questions["match"] = DecisionQuestion(
            type="choice", instructions=_MATCH, criteria={k: f"`existing_observations.{k}`" for k in keys}
        )
    for k in keys:
        questions[f"covers_{k}"] = _yes_no(_COVERS.format(key=k), _COVERS_YES.format(key=k), _COVERS_NO.format(key=k))
        questions[f"touches_{k}"] = _yes_no(
            _TOUCHES.format(key=k), _TOUCHES_YES.format(key=k), _TOUCHES_NO.format(key=k)
        )
    state: dict[str, Any] = {
        "new_fact": _dated(_sentence(fact), fact, stated="stated_on"),
        "existing_observations": {k: _dated(c.text, c, stated="last_stated_on") for k, c in zip(keys, candidates)},
    }
    if mission:
        state["mission"] = mission
        questions["keep_as"] = DecisionQuestion(type="choice", instructions=_KEEP_AS, criteria=dict(_KEEP_AS_OPTIONS))
    started = time.monotonic()
    answer: DecisionResponse | None = None
    error: BaseException | None = None
    try:
        answer = await model.ask(state, questions)
        return answer
    except BaseException as e:
        error = e
        raise
    finally:
        # Traced like any LLM call, so its cost shows next to the consolidation calls it saves.
        get_span_recorder().record_llm_call(
            provider=model.provider,
            model=model.model,
            scope="consolidation_triage",
            # One JSON message, the shape every recorder takes; the request is not a chat.
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        {"state": state, "questions": {k: q.system_one() for k, q in questions.items()}}
                    ),
                }
            ],
            response_content=answer.model_dump_json() if answer is not None else None,
            input_tokens=answer.usage.input_tokens if answer is not None else 0,
            output_tokens=0,
            duration=time.monotonic() - started,
            error=error,
        )


def _picked(answer: DecisionResponse, question_id: str) -> str | None:
    picked = answer.answers.get(question_id)
    if picked is None:
        return None
    if picked.probabilities:
        return max(picked.probabilities, key=lambda option: picked.probabilities[option])
    return picked.choice


def _yes(answer: DecisionResponse, question_id: str) -> bool:
    return _picked(answer, question_id) == "yes"


def stated_after(observation_at: Any, fact_at: Any) -> bool:
    def parse(value: Any) -> datetime | None:
        if value is None:
            return None
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    obs, fact = parse(observation_at), parse(fact_at)
    return obs is not None and fact is not None and obs > fact


async def triage_facts(
    memories: list[dict[str, Any]],
    candidates_per_fact: list[list[MemoryFact]],
    model: DecisionModel,
    *,
    entities_by_fact: dict[str, list[str]],
    mission: str,
    may_create: bool,
) -> TriageResult:
    """One System One request per fact, in parallel. A failed request leaves its fact to the LLM.

    ``mission``: the bank's own observations mission, empty for the built-in one. A bank's
    mission may want chatter kept (#525) or a fact reworded (no names, trends only, #4619), so
    under it the built-in "chatter" and "new" verdicts are not enough: nothing is dropped, and a
    fact is stored as stated only when its own wording already follows the mission. Anything else
    is the LLM's job. ``may_create``: false when the scope has an observation cap to respect.
    """
    result = TriageResult()
    started = time.monotonic()
    pending = [(m, c[:MAX_CANDIDATES]) for m, c in zip(memories, candidates_per_fact)]
    # return_exceptions: one failed request must not cancel the others; its fact goes to the LLM.
    answers = await asyncio.gather(*(_ask(model, m, c, mission) for m, c in pending), return_exceptions=True)

    new_candidates: list[dict[str, Any]] = []
    for (memory, candidates), answer in zip(pending, answers):
        if isinstance(answer, BaseException):
            logger.warning(f"[CONSOLIDATION][triage] request failed, fact goes to the LLM: {answer}")
            continue
        result.input_tokens += answer.usage.input_tokens
        fact_id = str(memory["id"])
        relation = _picked(answer, "relation")
        # The answers must agree: a fact that leaves an observation out of date is not "new", a
        # restatement or chatter, whatever the relation says. Cross-checking replaces the
        # confidence bars as the guard against a wrong verdict. An observation stated after the
        # fact does not count: older news never rewrites a newer belief (prompt rule 10).
        if any(
            _yes(answer, f"touches_o{i}") and not stated_after(c.mentioned_at, memory.get("mentioned_at"))
            for i, c in enumerate(candidates)
        ):
            continue
        # Under a bank's own mission nothing is dropped here: what the mission keeps is its call
        # (#525), and with a yes/no answer "drop" cannot be held to a higher bar than "keep" —
        # probed, both jev-latest and hindsight-decision dropped facts the mission wanted.
        if mission and relation == "chatter":
            continue
        if relation == "chatter":
            result.chatter.add(fact_id)
        elif relation == "restates" and candidates:
            target = _picked(answer, "match") or "o0"
            if _yes(answer, f"covers_{target}"):
                result.restates[fact_id] = str(candidates[int(target[1:])].id)
        elif (
            relation == "new"
            and may_create
            and _sentence(memory)
            and (not mission or _picked(answer, "keep_as") == "as_stated")
        ):
            new_candidates.append(memory)

    # Two new facts about the same entity in one batch may restate each other (the same event
    # told twice); recording both verbatim would make siblings. Those go to the LLM to merge.
    entity_count: dict[str, int] = {}
    for memory in new_candidates:
        for entity in set(entities_by_fact.get(str(memory["id"]), [])):
            entity_count[entity] = entity_count.get(entity, 0) + 1
    for memory in new_candidates:
        if all(entity_count.get(e, 0) == 1 for e in set(entities_by_fact.get(str(memory["id"]), []))):
            result.new[str(memory["id"])] = _sentence(memory)

    result.seconds = time.monotonic() - started
    logger.info(
        f"[CONSOLIDATION][triage] facts={len(memories)} restates={len(result.restates)} new={len(result.new)} "
        f"chatter={len(result.chatter)} decision_input_tokens={result.input_tokens} seconds={result.seconds:.2f}"
    )
    return result
