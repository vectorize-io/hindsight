"""Token-budget selection for the recall ``source_facts`` map.

Recall returns each observation's ``source_fact_ids`` in full and resolves them
through a ``source_facts`` map that is filled up to a token budget. Which facts
make it into that map is what this module decides.

Two properties matter to callers resolving provenance (issue #3221):

* the budget is spent in **observation-rank order**, so truncation hits the tail
  of the result list and the top-ranked results keep their provenance;
* one oversized fact skips itself only — it does not evict every shorter fact
  behind it — and any skip is reported so the caller can tell a budget-truncated
  map from a dangling reference.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFactSelection:
    """The source facts that fit the budget, plus whether anything was dropped."""

    ids: list[str]
    """Selected source fact IDs, in the order they were considered."""

    truncated: bool
    """True when at least one resolvable source fact was skipped for budget."""


def select_source_facts_within_budget(
    *,
    source_ids_ordered: list[str],
    source_fact_ids_by_obs: dict[str, list[str]],
    text_by_id: dict[str, str],
    max_total_tokens: int,
    max_tokens_per_observation: int,
    count_tokens: Callable[[str], int],
) -> SourceFactSelection:
    """Pick the source facts that fit the requested token budget.

    ``source_ids_ordered`` and ``source_fact_ids_by_obs`` must both be in
    observation-rank order — the budget is spent front to back, so their order
    decides which results keep their provenance when the budget runs out.

    A non-negative ``max_tokens_per_observation`` gives every observation its own
    budget (the global one is then unused); otherwise a single ``max_total_tokens``
    budget is shared, with a negative value meaning unlimited. IDs absent from
    ``text_by_id`` are skipped without counting as truncation: those rows did not
    resolve at all, which is a different condition from running out of budget.
    """
    selected: list[str] = []
    seen: set[str] = set()
    # Skipped for budget, not necessarily lost: under a per-observation cap the same
    # source can be dropped by one observation's budget and still fit another's. Only
    # a skip that leaves the fact out of the final map counts as truncation, so the
    # flag means exactly what it says — some advertised ID has no entry.
    skipped: set[str] = set()

    def _take(sid: str) -> None:
        if sid not in seen:
            seen.add(sid)
            selected.append(sid)

    if max_tokens_per_observation >= 0:
        # Per-observation capping: each observation independently selects source
        # facts up to its own budget.
        for sids in source_fact_ids_by_obs.values():
            obs_tokens = 0
            for sid in sids:
                text = text_by_id.get(sid)
                if text is None:
                    continue
                fact_tokens = count_tokens(text)
                if obs_tokens + fact_tokens > max_tokens_per_observation:
                    skipped.add(sid)
                    continue
                obs_tokens += fact_tokens
                _take(sid)
        return SourceFactSelection(ids=selected, truncated=bool(skipped - seen))

    total_tokens = 0
    for sid in source_ids_ordered:
        text = text_by_id.get(sid)
        if text is None:
            continue
        fact_tokens = count_tokens(text)
        if max_total_tokens >= 0 and total_tokens + fact_tokens > max_total_tokens:
            skipped.add(sid)
            continue
        total_tokens += fact_tokens
        _take(sid)

    return SourceFactSelection(ids=selected, truncated=bool(skipped - seen))
