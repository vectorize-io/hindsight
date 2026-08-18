"""Pure helpers for multi-bank recall merge + token budget cut.

The orchestrator (``MemoryEngine.recall_multi_async``) fans out one ``recall_async``
per bank (``reranking="rrf"`` when union CE will run) and then uses these helpers
to select a union, apply one CrossEncoder pass, order, and truncate.

Pipeline order (orchestrator must compose in this order):

1. Per-bank RRF retrieval (no per-bank CE when union CE is eligible).
2. Per-bank pre-cap + union select (``select_union_candidates``) — floor then
   interleave fill up to ``union_cap`` (default 200). CE cost is O(union_cap),
   not O(N_banks).
3. One ``CrossEncoderReranker.rerank`` over the union (skipped if CE is
   disabled / null / passthrough — today's rrf/interleave path).
4. Per-bank floor after the CE cut (``apply_per_bank_floor``).
5. Merge (``score_merge`` or ``interleave_merge`` fallback).
6. Exact / normalized-text dedup (``dedup_exact_normalized``) - BEFORE token cut.
7. Token budget cut (``cut_to_token_budget``).

Multi-bank fan-out defaults (orchestrator only; single-bank defaults unchanged):

- ``prefer_observations=True`` via :data:`MULTI_BANK_PREFER_OBSERVATIONS` (narrow
  semantics - see constant docstring).
- Full caller ``max_tokens`` per bank (no shared pool; no ``max_tokens / n_banks``).

Limitations still open:

- Union cap 200: a 5-bank fan-out keeps per-bank RRF ranks 0-39 in the CE
  pool (``union_cap // n_banks``). A caller ``max_tokens`` of ~2000 applied
  *before* CE in RRF order drops those deeper ranks; the union path therefore
  widens the per-bank subcall budget and applies the caller budget after CE.
- No embedding / near-duplicate collapse (would silently merge distinct facts such
  as "invalidate" vs "supersede").
- No cross-bank supersession: a stale fact in bank A can outrank its correction in
  bank B. Dedup / caps / prefer_observations do not fix that.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal, TypeVar

from .response_models import MemoryFact, RecallScores
from .search.types import MergedCandidate, RetrievalResult
from .token_encoding import get_token_encoding

MultiBankMerge = Literal["score", "interleave"]

# Hard cap on parallel bank fan-out (engine + HTTP request model).
MAX_MULTI_BANK_RECALL_BANKS = 10

# ---------------------------------------------------------------------------
# Per-bank contribution cap (anti-flood into the merge pool)
# ---------------------------------------------------------------------------
#
# Default rationale:
# - Stock per-bank ``recall_async`` already ranks/CE-reranks; the multi-bank pool
#   only needs each bank's *best* head, not its entire returned list.
# - Typical CE top windows and single-bank result heads sit around ~50 items; keeping
#   that head per bank preserves ranking quality without letting a large bank
#   (thousands of facts) dump a long CE-ranked tail into the union.
# - Worst-case pool before merge/dedup/cut is ``MAX_MULTI_BANK_RECALL_BANKS * M``
#   (10 * 50 = 500) - large enough not to starve score-merge, small enough to be
#   an anti-flood measure rather than "pass everything through".
# - Bank-size asymmetry already OVER-represents small banks (they contribute all
#   their results while the large bank is ranking/budget-limited). This cap trims
#   the large bank's flood; it must never starve a bank that has results to zero
#   (``max_per_bank`` is clamped to >= 1).
# - Rejected alternatives: ``interleave`` as the anti-flood fix (forces parity and
#   makes small-bank over-representation worse); ``max_tokens / n_banks`` (starves
#   both banks' cross-encoder windows).
DEFAULT_PER_BANK_MERGE_CAP = 50

# Union CE pool. 200 keeps per-bank RRF ranks 0-39 at N=5 (measured golds
# lived at ranks 13-37). Cost stays O(cap). Floor default 1 = never drop a
# bank that returned facts when union_cap >= n_banks (true at defaults:
# 200 >= 10).
DEFAULT_MULTI_UNION_CAP = 200
DEFAULT_PER_BANK_FLOOR = 1
# When union CE will run, each recall_async subcall uses at least this many
# tokens so a caller budget of ~2000 cannot drop deep RRF ranks before CE.
DEFAULT_UNION_SUBCALL_MAX_TOKENS = 100_000

# Multi-bank fan-out only. Single-bank ``recall_async`` default is unchanged.
#
# Real semantics are narrower than the name: when True, the engine drops raw facts
# whose ids appear in a returned observation's ``source_memory_ids``. It does NOT
# catch unlinked "same sentence, different fact_type" twins - that is what
# :func:`dedup_exact_normalized` is for.
MULTI_BANK_PREFER_OBSERVATIONS = True

# Metadata keys written into ``RecallResult.metadata`` by the orchestrator.
META_MULTI_BANK = "multi_bank"
META_MERGE_REQUESTED = "merge_requested"
META_MERGE_APPLIED = "merge_applied"
META_MERGE_FALLBACK_REASON = "merge_fallback_reason"
META_BANKS = "banks"
META_DEDUP = "dedup"
META_DEDUP_DROPPED = "dedup_dropped"
META_PER_BANK_CAP = "per_bank_cap"
META_UNION_CE = "union_ce"
META_UNION_CAP = "union_cap"
META_PER_BANK_FLOOR = "per_bank_floor"
# Was hard-coded ``"none"`` in v1 (no cross-bank dedup). Now exact/normalized text.
META_DEDUP_V1 = "exact_normalized"
META_DEDUP_MODE = META_DEDUP_V1  # alias; value is the active mode string

# Distinct post-gather fallback reason when returned facts lack usable CE scores
# (e.g. RRF passthrough cross-encoder sets scores.reranker=None for every result).
FALLBACK_NO_USABLE_RERANKER_SCORES = (
    "no usable cross-encoder scores in returned results (scores.reranker is None); "
    "score-merge would not order by relevance"
)

_T = TypeVar("_T")

# Banks with no facts in the merged list still may contribute side dicts
# (e.g. chunks fetched independently of max_tokens); rank them after all others.
_SIDE_DICT_UNRANKED = 10**9


@dataclass(frozen=True)
class UnionCandidatePool:
    """Facts selected for the single union CrossEncoder pass, plus bank order."""

    facts: list[MemoryFact]
    bank_order: list[str]


@dataclass(frozen=True)
class TokenedCandidate:
    """One MemoryFact paired with a unique CE token and a MergedCandidate."""

    token: str
    fact: MemoryFact
    candidate: MergedCandidate


@dataclass(frozen=True)
class DedupedFacts:
    """Facts kept after exact/normalized dedup, plus how many copies were dropped.

    Named type instead of a (facts, dropped) tuple: the project bar forbids
    multi-item tuple returns even for internal helpers.
    """

    facts: list[MemoryFact]
    dropped: int


def stamp_bank_id(fact: MemoryFact, bank_id: str) -> MemoryFact:
    """Return a copy of ``fact`` with ``bank_id`` set (orchestrator attribution)."""
    return fact.model_copy(update={"bank_id": bank_id})


def bank_rank_from_merged(facts: Sequence[MemoryFact]) -> dict[str, int]:
    """Map each bank_id to its best (earliest) position in the merged result list.

    Lower rank wins on side-dict key collisions. Banks absent from ``facts`` are
    omitted (callers treat them as unranked).
    """
    ranks: dict[str, int] = {}
    for index, fact in enumerate(facts):
        bid = fact.bank_id
        if bid is not None and bid not in ranks:
            ranks[bid] = index
    return ranks


def union_merge_dicts(
    bank_dicts: Sequence[tuple[str, Mapping[str, _T] | None]],
    *,
    bank_rank: Mapping[str, int],
) -> dict[str, _T] | None:
    """Union-merge optional per-bank dicts; on key collision keep the higher-ranked bank.

    ``bank_rank`` maps bank_id → rank (lower is better), typically from
    :func:`bank_rank_from_merged`. Banks not present in ``bank_rank`` sort last.
    Returns ``None`` when no bank contributed any entries (mirrors single-bank
    ``include_*`` omitted behaviour).
    """
    winners: dict[str, tuple[int, _T]] = {}
    for bank_id, mapping in bank_dicts:
        if not mapping:
            continue
        rank = bank_rank.get(bank_id, _SIDE_DICT_UNRANKED)
        for key, value in mapping.items():
            previous = winners.get(key)
            if previous is None or rank < previous[0]:
                winners[key] = (rank, value)
    if not winners:
        return None
    return {key: pair[1] for key, pair in winners.items()}


def _reranker_score(fact: MemoryFact) -> float:
    """Normalized cross-encoder score used for score-merge; missing → -inf (sort last)."""
    if fact.scores is None or fact.scores.reranker is None:
        return float("-inf")
    return float(fact.scores.reranker)


def cap_per_bank_results(
    bank_results: Sequence[tuple[str, Sequence[MemoryFact]]],
    *,
    max_per_bank: int = DEFAULT_PER_BANK_MERGE_CAP,
) -> list[tuple[str, list[MemoryFact]]]:
    """Take ``outcome.results[:M]`` per bank before merging (anti-flood).

    ``max_per_bank`` is clamped to at least 1 so a bank that returned results is
    never starved to zero by the cap alone (an empty bank still contributes nothing).

    Does not re-rank; preserves each bank's existing within-bank order. Apply this
    *before* :func:`score_merge` / :func:`interleave_merge`.
    """
    m = max(1, int(max_per_bank))
    return [(bank_id, list(facts[:m])) for bank_id, facts in bank_results]


def _parse_iso_datetime(value: Any) -> datetime | None:
    """Accept ISO strings *and* datetime/date (to_dict hands datetimes).

    ``datetime.replace("Z", ...)`` is ``datetime.replace`` (year/month/...)
    and raises TypeError, which would abort the whole union CE. Strings from
    MemoryFact.isoformat() are unchanged.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def union_per_bank_max_tokens(caller_max_tokens: int, *, union_ce: bool) -> int:
    """Token budget for each recall_async subcall.

    When union CE will run, do not apply the caller's (often 2000) cut in
    RRF order — that cut drops the deeper ranks that the union pass is
    meant to rescore. The caller budget is still applied after union CE
    via merge_cap_dedup_cut.
    """
    caller = int(caller_max_tokens)
    if not union_ce:
        return caller
    return max(caller, DEFAULT_UNION_SUBCALL_MAX_TOKENS)


def select_union_candidates(
    bank_results: Sequence[tuple[str, Sequence[MemoryFact]]],
    *,
    per_bank_pre_cap: int = DEFAULT_PER_BANK_MERGE_CAP,
    union_cap: int = DEFAULT_MULTI_UNION_CAP,
    k_floor: int = DEFAULT_PER_BANK_FLOOR,
) -> UnionCandidatePool:
    """Pre-cap each bank, reserve a floor, then interleave-fill up to ``union_cap``.

    Round-robin floor reservation so a CE-weak bank is not dropped before the
    union CE runs, as long as ``union_cap >= n_banks_with_facts``. When the cap
    is smaller than the bank count, later banks can be omitted — that is the
    cap, not a silent starve-to-zero of a bank that fit.
    """
    union_cap = max(1, int(union_cap))
    k_floor = max(1, int(k_floor))
    pre_capped = cap_per_bank_results(bank_results, max_per_bank=per_bank_pre_cap)
    stamped: list[tuple[str, list[MemoryFact]]] = [
        (bank_id, [stamp_bank_id(fact, bank_id) for fact in facts]) for bank_id, facts in pre_capped
    ]
    bank_order = [bank_id for bank_id, _facts in stamped]
    n_with = sum(1 for _bank_id, facts in stamped if facts)
    fair = min(k_floor, max(1, union_cap // max(1, n_with)))

    taken: dict[str, int] = {bank_id: 0 for bank_id, _facts in stamped}
    reserved: list[MemoryFact] = []
    for _round in range(fair):
        for bank_id, facts in stamped:
            if len(reserved) >= union_cap:
                break
            idx = taken[bank_id]
            if idx < len(facts) and idx < fair:
                reserved.append(facts[idx])
                taken[bank_id] = idx + 1
        if len(reserved) >= union_cap:
            break

    leftovers = [(bank_id, facts[taken[bank_id] :]) for bank_id, facts in stamped]
    if len(reserved) < union_cap:
        reserved.extend(interleave_merge(leftovers)[: union_cap - len(reserved)])
    return UnionCandidatePool(facts=reserved[:union_cap], bank_order=bank_order)


def apply_per_bank_floor(
    facts: Sequence[MemoryFact],
    *,
    k_floor: int,
    keep: int,
    bank_order: Sequence[str],
) -> list[MemoryFact]:
    """After CE scoring, keep each bank's top ``min(k_floor, count)`` then fill by CE.

    ``keep`` is the post-CE cut. A CE-weak bank still contributes its floor
    provided ``keep >= n_banks_with_facts``.
    """
    keep = max(0, int(keep))
    if keep == 0 or not facts:
        return []
    k_floor = max(1, int(k_floor))

    by_bank: dict[str, list[MemoryFact]] = {}
    for fact in facts:
        bid = fact.bank_id or ""
        by_bank.setdefault(bid, []).append(fact)
    for lst in by_bank.values():
        lst.sort(key=_reranker_score, reverse=True)

    ordered = [bid for bid in bank_order if by_bank.get(bid)]
    for bid in by_bank:
        if bid not in ordered:
            ordered.append(bid)
    if not ordered:
        return []

    fair = min(k_floor, max(1, keep // max(1, len(ordered))))
    protected: list[MemoryFact] = []
    protected_keys: set[tuple[str, str]] = set()
    taken: dict[str, int] = {bid: 0 for bid in ordered}
    for _round in range(fair):
        for bid in ordered:
            if len(protected) >= keep:
                break
            lst = by_bank[bid]
            idx = taken[bid]
            if idx < len(lst) and idx < fair:
                fact = lst[idx]
                protected.append(fact)
                protected_keys.add((bid, fact.id))
                taken[bid] = idx + 1
        if len(protected) >= keep:
            break

    rest = [
        fact
        for fact in sorted(facts, key=_reranker_score, reverse=True)
        if (fact.bank_id or "", fact.id) not in protected_keys
    ]
    return (protected + rest)[:keep]


def memory_facts_to_merged_candidates(facts: Sequence[MemoryFact]) -> list[TokenedCandidate]:
    """Build CE inputs. Tokens stay unique when two banks share a fact id."""
    tokened: list[TokenedCandidate] = []
    for index, fact in enumerate(facts):
        token = f"{index}:{fact.bank_id or ''}:{fact.id}"
        retrieval = RetrievalResult(
            id=token,
            text=fact.text,
            fact_type=fact.fact_type,
            context=fact.context,
            occurred_start=_parse_iso_datetime(fact.occurred_start),
        )
        tokened.append(
            TokenedCandidate(
                token=token,
                fact=fact,
                candidate=MergedCandidate(retrieval=retrieval, rrf_score=0.0),
            )
        )
    return tokened


def stamp_union_ce_scores(
    facts: Sequence[MemoryFact],
    tokens: Sequence[str],
    score_by_token: Mapping[str, float],
) -> list[MemoryFact]:
    """Copy CE-normalized scores onto each fact (comparable across banks)."""
    stamped: list[MemoryFact] = []
    for fact, token in zip(facts, tokens, strict=True):
        score = score_by_token.get(token)
        if score is None:
            stamped.append(fact)
            continue
        previous = fact.scores
        stamped.append(
            fact.model_copy(
                update={
                    "scores": RecallScores(
                        final=float(score),
                        reranker=float(score),
                        semantic=previous.semantic if previous is not None else None,
                        keyword=previous.keyword if previous is not None else None,
                    )
                }
            )
        )
    return stamped


def score_merge(bank_results: Sequence[tuple[str, Sequence[MemoryFact]]]) -> list[MemoryFact]:
    """Sort the union of per-bank results by normalized cross-encoder score descending.

    Tie-break (stable, deterministic): bank list order, then within-bank rank (input
    order). Each result is stamped with its source ``bank_id``.

    Score-merge is only honest when every contributing bank actually ran
    cross_encoder reranking (see ``cross_encoder_eligible`` / orchestrator auto-fallback).
    """
    tagged: list[tuple[float, int, int, MemoryFact]] = []
    for bank_idx, (bank_id, facts) in enumerate(bank_results):
        for rank, fact in enumerate(facts):
            stamped = stamp_bank_id(fact, bank_id)
            # Negate score so ascending sort yields highest first; bank_idx/rank break ties.
            tagged.append((-_reranker_score(stamped), bank_idx, rank, stamped))
    tagged.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in tagged]


def interleave_merge(bank_results: Sequence[tuple[str, Sequence[MemoryFact]]]) -> list[MemoryFact]:
    """Round-robin merge by per-bank rank: bankA#1, bankB#1, bankA#2, bankB#2, ...

    Banks appear in the order of ``bank_results``. Empty banks contribute no slots.
    Each result is stamped with its source ``bank_id``. Guarantees per-bank
    representation when banks have results (unlike pure score sort).

    Note: interleave is the score-merge *fallback* when CE scores are unusable. It
    is NOT the anti-flood measure for bank-size asymmetry (that is
    :func:`cap_per_bank_results`); using interleave as parity-forcing anti-flood
    was considered and rejected.
    """
    stamped_lists: list[list[MemoryFact]] = [
        [stamp_bank_id(fact, bank_id) for fact in facts] for bank_id, facts in bank_results
    ]
    merged: list[MemoryFact] = []
    max_len = max((len(lst) for lst in stamped_lists), default=0)
    for rank in range(max_len):
        for lst in stamped_lists:
            if rank < len(lst):
                merged.append(lst[rank])
    return merged


def normalize_dedup_key(text: str | None) -> str:
    """Normalize fact text for exact cross-bank dedup: casefold + whitespace collapse.

    Whitespace is any Unicode whitespace (``str.split`` with no args). Case uses
    ``str.casefold`` (stronger than lower for some locales). Punctuation and
    near-paraphrase differences are intentionally preserved so genuinely distinct
    facts are not collapsed.
    """
    return " ".join((text or "").casefold().split())


def dedup_exact_normalized(facts: Sequence[MemoryFact]) -> DedupedFacts:
    """Drop later duplicates of the same normalized text; keep the higher-ranked copy.

    Higher-ranked = earlier in ``facts`` (call after score/interleave merge so the
    list is already ordered). Must run *before* :func:`cut_to_token_budget` so
    dropped duplicates do not consume token budget.

    Exact / normalized only - no embedding or near-duplicate collapse.
    """
    seen: set[str] = set()
    kept: list[MemoryFact] = []
    dropped = 0
    for fact in facts:
        key = normalize_dedup_key(fact.text)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(fact)
    return DedupedFacts(facts=kept, dropped=dropped)


def cut_to_token_budget(facts: Sequence[MemoryFact], max_tokens: int) -> list[MemoryFact]:
    """Keep results until ``max_tokens`` on the ``text`` field (same semantics as single-bank).

    Stops before including a fact that would exceed the budget. Counts tokens with
    the shared cl100k_base encoding. ``max_tokens <= 0`` yields an empty list.
    """
    if max_tokens <= 0:
        return []
    encoding = get_token_encoding()
    filtered: list[MemoryFact] = []
    total = 0
    for fact in facts:
        text_tokens = len(encoding.encode(fact.text or ""))
        if total + text_tokens <= max_tokens:
            filtered.append(fact)
            total += text_tokens
        else:
            break
    return filtered


def merge_cap_dedup_cut(
    bank_results: Sequence[tuple[str, Sequence[MemoryFact]]],
    *,
    merge: MultiBankMerge,
    max_tokens: int,
    max_per_bank: int = DEFAULT_PER_BANK_MERGE_CAP,
) -> DedupedFacts:
    """Compose the multi-bank post-gather pipeline: cap -> merge -> dedup -> token cut.

    ``DedupedFacts.facts`` is the token-cut list. ``DedupedFacts.dropped`` is how
    many facts ``dedup_exact_normalized`` removed (counted before the cut).
    Orchestrators may call the steps individually; this helper encodes the order.
    """
    capped = cap_per_bank_results(bank_results, max_per_bank=max_per_bank)
    if merge == "score":
        merged = score_merge(capped)
    else:
        merged = interleave_merge(capped)
    deduped = dedup_exact_normalized(merged)
    return DedupedFacts(
        facts=cut_to_token_budget(deduped.facts, max_tokens),
        dropped=deduped.dropped,
    )


def cross_encoder_eligible(
    *,
    requested_reranking: str,
    bank_enable_reranking: Sequence[bool],
) -> tuple[bool, str | None]:
    """Whether score-merge is *predicted* valid from config (pre-flight).

    Returns ``(True, None)`` when every bank would resolve to cross_encoder, else
    ``(False, reason)`` describing why score-merge must fall back to interleave.

    Mirrors ``_resolve_reranking``: only ``cross_encoder`` is downgraded when
    ``enable_reranking`` is false (to ``rrf``). Caller-requested ``rrf`` / ``interleave``
    never produce comparable ``scores.reranker`` values.

    This is necessary but not sufficient: the orchestrator also checks
    :func:`has_usable_reranker_scores` on the actual returned facts (post-gather),
    because an RRF passthrough cross-encoder still "resolves" to cross_encoder yet
    writes ``scores.reranker=None`` on every result.
    """
    if requested_reranking != "cross_encoder":
        return (
            False,
            f"requested reranking={requested_reranking!r} (score-merge requires cross_encoder)",
        )
    if not all(bank_enable_reranking):
        return (
            False,
            "one or more banks have enable_reranking=false (resolved reranking would be rrf)",
        )
    return True, None


def has_usable_reranker_scores(bank_results: Sequence[tuple[str, Sequence[MemoryFact]]]) -> bool:
    """True when returned facts include at least one usable ``scores.reranker`` value.

    Empty result sets are treated as usable (nothing to mis-order). If any facts are
    present, at least one must carry a non-None ``scores.reranker``; otherwise
    score-merge degenerates to bank-concatenation under a stable sort of all -inf.
    """
    saw_fact = False
    for _bank_id, facts in bank_results:
        for fact in facts:
            saw_fact = True
            if fact.scores is not None and fact.scores.reranker is not None:
                return True
    return not saw_fact


def build_multi_bank_metadata(
    *,
    merge_requested: MultiBankMerge,
    merge_applied: MultiBankMerge,
    merge_fallback_reason: str | None,
    bank_statuses: dict[str, dict],
    dedup: str = META_DEDUP_MODE,
    dedup_dropped: int = 0,
    per_bank_cap: int = DEFAULT_PER_BANK_MERGE_CAP,
    union_ce: bool = False,
    union_cap: int = DEFAULT_MULTI_UNION_CAP,
    per_bank_floor: int = DEFAULT_PER_BANK_FLOOR,
) -> dict:
    """Assemble the multi-bank block stored on ``RecallResult.metadata``.

    ``dedup`` is the mode string (default :data:`META_DEDUP_MODE` /
    ``exact_normalized``). ``dedup_dropped`` is how many facts were removed by
    :func:`dedup_exact_normalized`. ``per_bank_cap`` records the M used for
    :func:`cap_per_bank_results`. ``union_ce`` is whether one CrossEncoder pass
    ran over the union (HTTP request schema unchanged; this is response metadata).
    """
    return {
        META_MULTI_BANK: {
            META_MERGE_REQUESTED: merge_requested,
            META_MERGE_APPLIED: merge_applied,
            META_MERGE_FALLBACK_REASON: merge_fallback_reason,
            META_BANKS: bank_statuses,
            META_DEDUP: dedup,
            META_DEDUP_DROPPED: int(dedup_dropped),
            META_PER_BANK_CAP: int(per_bank_cap),
            META_UNION_CE: bool(union_ce),
            META_UNION_CAP: int(union_cap),
            META_PER_BANK_FLOOR: int(per_bank_floor),
        }
    }
