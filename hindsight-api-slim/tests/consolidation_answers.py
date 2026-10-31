"""Hand-written consolidation answers for tests that stand in for the LLM.

The model answers one entry per new fact (``_FactBatchResponse``); a test usually knows the
writes it wants instead. :func:`fact_answer` takes those writes and gives the answer that
produces them, and :func:`batch_fact_ids` reads the batch's fact ids out of the prompt.
"""

import re

from hindsight_api.engine.consolidation.consolidator import (
    _CreateAction,
    _DeleteAction,
    _FactBatchResponse,
    _FactDecision,
    _ObservationChange,
    _UpdateAction,
)


def batch_fact_ids(prompt: str) -> list[str]:
    """The ids of the new facts the prompt lists (``[F1] text``), in order."""
    return re.findall(r"^\[(F\d+)\]", prompt, re.MULTILINE)


def batch_fact_texts(prompt: str) -> dict[str, str]:
    """Each new fact's text by its id. The ids are F1, F2, … in every batch, so an observation
    written from the id alone repeats across batches."""
    return dict(re.findall(r"^\[(F\d+)\] (.*)$", prompt, re.MULTILINE))


def fact_answer(
    creates: list[_CreateAction] | None = None,
    updates: list[_UpdateAction] | None = None,
    deletes: list[_DeleteAction] | None = None,
) -> _FactBatchResponse:
    """The answer that turns into these writes: each is put on its first source fact, and the
    other source facts point at it (``same_as_fact`` for a create, ``repeats`` for an update)."""
    entries: dict[str, _FactDecision] = {}

    def entry(fact_id: str) -> _FactDecision:
        return entries.setdefault(fact_id, _FactDecision(fact_id=fact_id, kind="lasting", adds="no"))

    for c in creates or []:
        first, *rest = c.source_fact_ids
        entry(first).new.append(c.text)
        entry(first).adds = "yes"
        for other in rest:
            entry(other).same_as_fact = first
    for u in updates or []:
        first, *rest = u.source_fact_ids
        entry(first).existing.append(_ObservationChange(observation_id=u.observation_id, effect="changes", text=u.text))
        for other in rest:
            entry(other).existing.append(_ObservationChange(observation_id=u.observation_id, effect="repeats"))
    return _FactBatchResponse(facts=list(entries.values()), deletes=list(deletes or []))
