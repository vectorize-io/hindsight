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

import re
from dataclasses import dataclass, field
from typing import Any

from .structured_doc import Block, StructuredDocument

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
        """True when *none* of the cited grounding resolves: the page's whole source is gone.

        That is a deleted source, or one re-ingested under fresh ids. Either way an
        unsay pass would be asked to remove everything the page says, so the refresh
        rebuilds the page from live facts instead (``mode_fallback_reason=
        "grounding_gone"``). Bank transfer does not land here: import rewrites
        ``based_on`` to the new ids (#3833).

        The rebuild is a full refresh, so it keeps nothing the page holds beyond what
        the bank can still ground — including text a user wrote by hand before turning
        on delta mode. That is the intended trade: a page none of whose evidence exists
        is stating things nothing supports, and the old behaviour (keep the prose)
        kept a deleted document's claims, and their citations, forever (#6764).
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


#: Stands in for a removed id while the punctuation around it is tidied, so only
#: brackets and commas that held a dropped id are ever touched — never a ``foo()``
#: or an empty ``[]`` the text already had.
_GAP = "\x00"
#: A ``[...]`` or ``(...)`` group holding nothing but removed ids and separators.
_EMPTIED_GROUP_RE = re.compile(rf"([ \t]*)[\[(](?:[ \t,;]|{_GAP})*{_GAP}(?:[ \t,;]|{_GAP})*[\])](?=([\[(])?)")


def _drop_ids_from_text(text: str, drop_ids: set[str]) -> str:
    alternatives = "|".join(re.escape(i) for i in sorted(drop_ids))
    an_id = rf"(?<![\w-])(?:{alternatives})(?![\w-])"
    # A link pointing at a dropped id keeps only its text: ``[A](/m/A)`` -> ``A``.
    text = re.sub(rf"\[([^\[\]\n]*)\]\([^()\n]*{an_id}[^()\n]*\)", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(an_id, _GAP, text, flags=re.IGNORECASE)
    if _GAP not in text:
        return text
    # One separator goes with a removed id — the one before it, or after it when it
    # opens a group: ``[B, A, C]`` -> ``[B, C]``, ``[A, B]`` -> ``[B]``.
    text = re.sub(rf"[ \t]*[,;][ \t]*{_GAP}", _GAP, text)
    text = re.sub(rf"{_GAP}+", _GAP, text)
    text = re.sub(rf"(?:(?<=[\[(])|^){_GAP}[ \t]*[,;][ \t]*", _GAP, text, flags=re.MULTILINE)
    text = re.sub(rf"`{_GAP}`", _GAP, text)
    # A group left holding nothing goes too, from the inside out (``[[A]]``). The space
    # before it stays only when another group follows: ``x. [A][B]`` -> ``x. [B]``.
    while True:
        tidied = _EMPTIED_GROUP_RE.sub(
            lambda m: _GAP + m.group(1) if m.group(2) else m.group(1) + _GAP,
            text,
        )
        if tidied == text:
            break
        text = tidied
    # A line (or list item, or quote) that held nothing but the id goes; one that
    # starts with it keeps its text.
    marker = r"(?:[-*+>]|\d+[.)])?"
    text = re.sub(rf"\n[ \t]*{marker}[ \t]*{_GAP}[ \t]*(?=\n|$)", "", text)
    text = re.sub(rf"^[ \t]*{marker}[ \t]*{_GAP}[ \t]*(?:\n|$)", "", text)
    text = re.sub(rf"^([ \t]*(?:{marker}[ \t]+)?){_GAP}[ \t]*", r"\1", text, flags=re.MULTILINE)
    return re.sub(rf"[ \t]*{_GAP}", "", text)


def drop_inline_citations(doc: StructuredDocument, drop_ids: set[str]) -> StructuredDocument:
    """A copy of ``doc`` with ``drop_ids`` removed from its text.

    The text twin of :func:`prune_based_on`. The model cites handles that are resolved
    to bare UUIDs, so an id may sit anywhere: ``[id]``, ``(id)``, ``[a, id]``, a link,
    or bare in a sentence. Only an exact dropped id is removed, with the separators and
    the brackets that held it; any other UUID, bracket or parenthesis is left alone.
    """
    lowered = {i.lower() for i in drop_ids}
    if not lowered:
        return doc
    return doc.model_copy(
        update={
            "sections": [
                section.model_copy(
                    update={
                        # A block that was nothing but the id goes with it.
                        "blocks": [
                            Block(id=block.id, text=text)
                            for block in section.blocks
                            if (text := _drop_ids_from_text(block.text, lowered)).strip()
                        ]
                    }
                )
                for section in doc.sections
            ]
        }
    )
