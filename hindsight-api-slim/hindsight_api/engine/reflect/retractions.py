"""Retracted grounding: the facts a document still cites that no longer exist.

A mental model records what it was built from under ``reflect_response.based_on``,
keyed by fact type. When one of those facts is invalidated (moved to
``invalidated_memory_units``), deleted, or swept away as a stale observation, the
row simply stops existing — retrieval can never return it again, but the document
keeps stating it and keeps citing it.

Nothing in the pipeline notices, because every signal it has is phrased over rows
that are *present*: staleness asks whether any memory was written since the last
refresh, and the delta prompt is handed only facts that are new. A retraction is
the absence of a row, so it raises no watermark and reaches no prompt.

This module turns that absence into a value. It is deliberately pure — the caller
supplies the stored ``based_on`` and the set of ids still live (see
``Memories.live_memory_ids``), and everything here is a dict transform, so the
partition and the pruning are testable without a database or an LLM.

Why the cause is not distinguished
----------------------------------
Invalidated, deleted, and re-ingested all present identically: an id in
``based_on`` with no live row. They are also, from the document's point of view,
the same event — the sentence it is telling the reader is no longer supported by
anything. So this asks one question ("is it live?") rather than trying to recover
a cause it cannot see. The one case that genuinely differs is a re-ingest, where
an equivalent fact returns under a fresh id; that is handled by *when* the caller
acts (never while consolidation is pending) and by the prompt, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The ``based_on`` keys whose ids address rows in ``memory_units``.
#
# ``based_on`` also carries ``mental-models`` and ``directives``, whose ids address
# entirely different tables. Feeding those to a live-memory lookup would report every
# one of them as missing and retract content grounded on a perfectly healthy sibling
# document — so membership here is an allowlist, never an exclusion list. ``opinion``
# is left out on the same principle: reflect still initialises the key, but no such
# ``fact_type`` is written to ``memory_units`` today, and treating an id we cannot
# resolve as "retracted" is the one error this module must never make.
MEMORY_BACKED_FACT_TYPES = frozenset({"world", "experience", "observation"})


@dataclass(frozen=True)
class RetractedGrounding:
    """The memory-backed facts a document cites that no longer exist.

    ``facts`` keeps the stored ``based_on`` entries verbatim — including their
    ``text``. That matters: an observation swept away when its source was
    invalidated is hard-deleted along with its history, so the copy the document
    recorded is the only surviving record of what it actually said, and therefore
    the only thing that can be quoted back to an LLM being asked to unsay it.
    """

    facts: list[dict[str, Any]] = field(default_factory=list)
    cited_count: int = 0
    """How many memory-backed facts the document cited in total, live ones included."""

    @property
    def ids(self) -> set[str]:
        return {str(fact.get("id")) for fact in self.facts if fact.get("id")}

    @property
    def unresolvable(self) -> bool:
        """True when *none* of the cited grounding resolves — a broken link, not a retraction.

        A retraction removes some of a document's evidence. Losing all of it at once
        is the signature of provenance that never pointed at this bank in the first
        place: a whole-bank import carries ``mental_models`` rows verbatim while the
        memories are re-created under fresh ids, so every id a restored page cites is
        absent on arrival. Treating that as a retraction would ask the unsay pass to
        delete everything the page says — irreversibly, on a page that is perfectly
        correct.

        So this case prunes the dangling citations and leaves the prose alone. The
        cost is a false negative for a page whose entire (small) grounding really was
        retracted; it keeps a stale sentence until later refreshes re-ground it. That
        is the right way round: the failure mode here is unrecoverable and the other
        one is not.
        """
        return self.cited_count > 0 and len(self.facts) == self.cited_count

    def __bool__(self) -> bool:
        return bool(self.facts)


def based_on_fact_ids(based_on: dict[str, Any] | None) -> list[str]:
    """Every memory-backed fact id a stored ``based_on`` cites, in stable order.

    Non-memory fact types are skipped (see :data:`MEMORY_BACKED_FACT_TYPES`), as are
    malformed entries — a ``based_on`` written by an older version, or hand-edited,
    must not break a refresh.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for fact_type, facts in (based_on or {}).items():
        if fact_type not in MEMORY_BACKED_FACT_TYPES or not isinstance(facts, list):
            continue
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            fact_id = fact.get("id")
            if fact_id and str(fact_id) not in seen:
                seen.add(str(fact_id))
                ids.append(str(fact_id))
    return ids


def partition_retracted(based_on: dict[str, Any] | None, live_ids: set[str]) -> RetractedGrounding:
    """Split a stored ``based_on`` into the facts still live and those retracted.

    ``live_ids`` is the answer from ``Memories.live_memory_ids`` for
    :func:`based_on_fact_ids`. Anything memory-backed and absent from it is
    retracted; everything else — live facts, mental models, directives — is not
    reported, because nothing about it has changed.
    """
    retracted: list[dict[str, Any]] = []
    cited = 0
    for fact_type, facts in (based_on or {}).items():
        if fact_type not in MEMORY_BACKED_FACT_TYPES or not isinstance(facts, list):
            continue
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            fact_id = fact.get("id")
            if not fact_id:
                continue
            cited += 1
            if str(fact_id) not in live_ids:
                retracted.append(fact)
    return RetractedGrounding(facts=retracted, cited_count=cited)


def prune_based_on(based_on: dict[str, Any] | None, drop_ids: set[str]) -> dict[str, list[Any]]:
    """A copy of ``based_on`` without the named ids.

    Called only for retractions the refresh actually acted on. A deferred one must
    keep its entry: the id is the sole remaining evidence that the document is
    citing something that no longer exists, so dropping it without editing the prose
    would leave the stale sentence in place and unreachable — the next refresh would
    have nothing left to notice.
    """
    pruned: dict[str, list[Any]] = {}
    for fact_type, facts in (based_on or {}).items():
        if not isinstance(facts, list):
            continue
        kept = [
            fact
            for fact in facts
            if not (isinstance(fact, dict) and fact.get("id") and str(fact.get("id")) in drop_ids)
        ]
        pruned[fact_type] = kept
    return pruned
