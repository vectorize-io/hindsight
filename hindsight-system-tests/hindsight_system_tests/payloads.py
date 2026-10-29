"""The payloads the stub returns, as models rather than hand-built dicts.

Two reasons these are typed. The extraction envelope has six required fields, most
of which are "N/A" in any given test, so writing them inline would bury the one or
two a test actually cares about. And a mistyped key in a raw dict does not fail
here — it fails several layers away as "all N facts returned by the LLM were
unusable", which is a long walk back to a typo.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field, model_serializer

from .rulebook import ChatRequest

FactType = Literal["world", "assistant"]


class Fact(BaseModel):
    """One extracted fact, mirroring the server's ``ExtractedFact``."""

    what: str
    when: str = "N/A"
    where: str = "N/A"
    who: str = "N/A"
    why: str = "N/A"
    fact_kind: Literal["event", "conversation"] = "conversation"
    """Whether this is a datable occurrence. Load-bearing: the extraction
    pipeline only keeps ``occurred_start``/``occurred_end`` for ``event`` facts
    and discards them for a ``conversation``, so a date without this set is
    silently dropped."""

    fact_type: FactType = "world"
    """The server's own distinction, not the docs' user-facing one: ``world`` covers
    objective facts (including the user's preferences and corrections), ``assistant``
    covers what the agent itself did."""

    entities: list[str] = Field(default_factory=list)
    occurred_start: str | None = None
    occurred_end: str | None = None
    from_attachments: list[int] | None = None
    """1-based numbers of the chunk's attachments this fact was read off — the edge
    that makes a fact (and what it is consolidated into) carry its attachments."""
    labels: dict[str, str | None] | None = None
    """The bank's entity-label values for this fact (``{"kind": "rule"}``), keyed by label
    group. Only meaningful when the bank configures ``entity_labels``; a group with
    ``tag: true`` turns its value into a ``key:value`` tag on the fact."""


class ExtractedFacts(BaseModel):
    """The ``FactExtractionResponse`` envelope."""

    facts: list[Fact] = Field(default_factory=list)


class Observation(BaseModel):
    """One entry for ``Consolidation.creates``."""

    text: str
    source_fact_ids: list[str]
    reason: str = "system test"


class ObservationUpdate(BaseModel):
    text: str
    observation_id: str
    source_fact_ids: list[str]
    reason: str = "system test"


class ObservationDelete(BaseModel):
    observation_id: str
    #: The observation that absorbed this one's content; its evidence moves there.
    merged_into: str | None = None
    reason: str = "system test"


class Consolidation(BaseModel):
    """The writes a consolidation answer must produce.

    All three default to empty, which is the honest answer for a story that is not
    about consolidation: the facts arrived and none warranted an observation. A test
    still has to say so — it is a declaration, not a silent default.

    The model answers one entry per new fact (:class:`FactConsolidation`); a story
    knows the writes it wants instead, so this serializes as the answer that
    produces them: each write on its first source fact, the other source facts
    pointing at it (``same_as_fact`` for a create, ``repeats`` for an update).
    """

    creates: list[Observation] = Field(default_factory=list)
    updates: list[ObservationUpdate] = Field(default_factory=list)
    deletes: list[ObservationDelete] = Field(default_factory=list)

    @model_serializer
    def _as_answer(self) -> dict[str, object]:
        entries: dict[str, FactEntry] = {}

        def entry(fact_id: str) -> FactEntry:
            return entries.setdefault(fact_id, FactEntry(fact_id=fact_id))

        for c in self.creates:
            if not c.source_fact_ids:
                continue
            first, *rest = c.source_fact_ids
            entry(first).new.append(c.text)
            entry(first).adds = "yes"
            for other in rest:
                if other != first:
                    entry(other).same_as_fact = first
        for u in self.updates:
            if not u.source_fact_ids:
                continue
            first, *rest = u.source_fact_ids
            entry(first).existing.append(
                ObservationJudgement(observation_id=u.observation_id, effect="changes", text=u.text)
            )
            for other in rest:
                if other != first:
                    entry(other).existing.append(
                        ObservationJudgement(observation_id=u.observation_id, effect="repeats")
                    )
        return FactConsolidation(facts=list(entries.values()), deletes=self.deletes).model_dump()


class ObservationJudgement(BaseModel):
    """One existing observation a consolidation answer entry judges."""

    observation_id: str
    observation_text: str = ""
    observation_about: str = ""
    effect: Literal["unrelated", "repeats", "changes", "ends"]
    text: str = ""


class FactEntry(BaseModel):
    """What a consolidation answer does with one new fact."""

    fact_id: str
    about: str = ""
    kind: Literal["lasting", "momentary"] = "lasting"
    existing: list[ObservationJudgement] = Field(default_factory=list)
    adds: Literal["yes", "no"] = "no"
    new: list[str] = Field(default_factory=list)
    same_as_fact: str | None = None


class FactConsolidation(BaseModel):
    """The consolidation answer: one entry per new fact."""

    facts: list[FactEntry] = Field(default_factory=list)
    deletes: list[ObservationDelete] = Field(default_factory=list)


def fact(what: str, **fields: object) -> Fact:
    """``Fact(what=...)`` with the required-but-usually-irrelevant fields defaulted."""
    return Fact(what=what, **fields)  # type: ignore[arg-type]


def extracted(*facts: Fact) -> ExtractedFacts:
    return ExtractedFacts(facts=list(facts))


def consolidation(
    *,
    creates: list[Observation] | None = None,
    updates: list[ObservationUpdate] | None = None,
    deletes: list[ObservationDelete] | None = None,
) -> Consolidation:
    return Consolidation(creates=creates or [], updates=updates or [], deletes=deletes or [])


def fact_ids_in(prompt: str) -> list[str]:
    """The fact ids the consolidation prompt is asking about.

    Consolidation must cite the facts it drew on by id, and the prompt names them
    per call (``[F1] fact text``) — a test cannot write them down in advance, so a
    rule reads them back out of the request it is answering. Only the batch's
    facts: the system prompt's worked example uses the same ids.
    """
    batch = prompt[prompt.rfind("### New facts") : prompt.rfind("### Existing observations")]
    return re.findall(r"^\[(F\d+)\]", batch, re.MULTILINE)


def batch_fact_ids(request: "ChatRequest") -> list[str]:
    """The ids of the facts this consolidation call is about."""
    return fact_ids_in(request.user_text)


def batch_fact_id(request: "ChatRequest", needle: str) -> str | None:
    """The id of the batch fact whose text contains ``needle``, if it is in this batch."""
    text = request.user_text
    facts = text[text.rfind("### New facts") : text.rfind("### Existing observations")]
    return next(
        (line[1 : line.index("]")] for line in facts.splitlines() if line.startswith("[") and needle in line), None
    )


def shown_observations(request: "ChatRequest") -> dict[str, str]:
    """``{text: id}`` of the existing observations this call was shown."""
    text = request.user_text
    shown = text[text.rfind("### Existing observations") + len("### Existing observations") :].strip()
    if not shown.startswith("["):
        return {}
    return {o["text"]: o["id"] for o in json.loads(shown[: shown.rindex("]") + 1])}


def observes(text: str) -> Callable[["ChatRequest"], Consolidation]:
    """A consolidation reply drawing one observation from every fact in the prompt.

    Pair with ``llm.on_step("consolidate").answers_with(...)``. The ids have to be
    read out of the request because the server minted them during the retain that
    triggered this call.
    """

    def build(request: ChatRequest) -> Consolidation:
        return Consolidation(
            creates=[Observation(text=text, source_fact_ids=fact_ids_in(request.all_text), reason="system test")]
        )

    return build


class AddSection(BaseModel):
    """One operation of a delta refresh's edit: a new section at the end of the page."""

    op: Literal["add_section"] = "add_section"
    heading: str
    blocks: list[str]


class DeltaOperations(BaseModel):
    """What a delta knowledge-page refresh's edit call returns."""

    operations: list[AddSection]


def add_section(heading: str, *blocks: str) -> DeltaOperations:
    return DeltaOperations(operations=[AddSection(heading=heading, blocks=list(blocks))])
