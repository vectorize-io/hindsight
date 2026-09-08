"""The payloads the stub returns, as models rather than hand-built dicts.

Two reasons these are typed. The extraction envelope has six required fields, most
of which are "N/A" in any given test, so writing them inline would bury the one or
two a test actually cares about. And a mistyped key in a raw dict does not fail
here — it fails several layers away as "all N facts returned by the LLM were
unusable", which is a long walk back to a typo.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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
    reason: str = "system test"


class Consolidation(BaseModel):
    """The three arrays consolidation must always return.

    All three default to empty, which is the honest answer for a story that is not
    about consolidation: the facts arrived and none warranted an observation. A test
    still has to say so — it is a declaration, not a silent default.
    """

    creates: list[Observation] = Field(default_factory=list)
    updates: list[ObservationUpdate] = Field(default_factory=list)
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
