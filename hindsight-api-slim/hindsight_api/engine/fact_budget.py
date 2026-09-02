"""Token-budget selection for the facts recall returns.

Recall ranks its candidates and then hands back as many as ``max_tokens`` worth
of fact text. Which facts make it into that answer is what this module decides —
the same job :mod:`hindsight_api.engine.source_facts` does for the provenance
map, and it answers to the same two properties (issues #3221, #3688):

* the budget is spent in **rank order**, so truncation hits the tail of the
  result list and the best-ranked facts are the ones that survive;
* one oversized fact skips itself only — it does not evict every shorter fact
  behind it.

Recall also keeps a floor: as long as the caller asked for a positive budget, a
run that matched something never answers with nothing. When no fact fits at all,
the top-ranked one is returned whole and over budget, because the alternatives
are worse — an empty answer reads as "this bank knows nothing about that", and a
fact clipped mid-sentence would be a claim the memory never made.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class FactSelection:
    """The facts that fit the budget, plus what the budget cost."""

    ids: list[str]
    """Selected fact IDs, in rank order."""

    total_tokens: int
    """Tokens the selected facts spend. Exceeds the budget when the floor applied."""

    truncated: bool
    """True when at least one ranked fact was dropped to stay inside the budget.

    Diagnostics only — it feeds the recall log line and trace. It is not worth
    reporting to callers: a broad candidate set usually carries more text than any
    sane budget, so on a bank of any size this is true on nearly every recall.
    """


def select_facts_within_budget(
    *,
    fact_ids_ordered: list[str],
    text_by_id: dict[str, str],
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> FactSelection:
    """Pick the ranked facts that fit ``max_tokens``.

    ``fact_ids_ordered`` must be in rank order — the budget is spent front to
    back, so their order decides which facts survive when it runs out. IDs absent
    from ``text_by_id`` are skipped without counting as truncation: those rows did
    not resolve at all, which is a different condition from running out of budget.

    A non-positive ``max_tokens`` selects nothing and the floor does not apply:
    ``max_tokens=0`` is the documented way to ask for chunks without facts
    (issue #364).
    """
    selected: list[str] = []
    total_tokens = 0
    skipped = False

    for fact_id in fact_ids_ordered:
        text = text_by_id.get(fact_id)
        if text is None:
            continue
        fact_tokens = count_tokens(text)
        if total_tokens + fact_tokens > max_tokens:
            skipped = True
            continue
        total_tokens += fact_tokens
        selected.append(fact_id)

    if not selected and skipped and max_tokens > 0:
        # Every candidate is individually over budget. Answering with nothing would be
        # indistinguishable from "no such memory", so spend the floor on the top-ranked
        # fact — the returned `total_tokens` shows the overshoot.
        for fact_id in fact_ids_ordered:
            text = text_by_id.get(fact_id)
            if text is not None:
                selected.append(fact_id)
                total_tokens = count_tokens(text)
                break

    return FactSelection(ids=selected, total_tokens=total_tokens, truncated=skipped)
