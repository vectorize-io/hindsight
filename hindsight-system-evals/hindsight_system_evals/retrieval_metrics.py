"""Ranked-retrieval metrics, in the one form this repo can score.

Why they are computed here rather than taken from a library: `recall` has no `k`.
It budgets by `max_tokens` and returns memory *units*, while a BEIR qrel labels a
*document*. So the ranked list a metric needs does not exist until two things
happen — collapse the units onto their source document, keeping each document at
the rank of its best-placed unit, then truncate at k. Both are decisions about
Hindsight's output shape, not about the metric, which is why `pytrec_eval` cannot
be handed the raw response.

The definitions follow `pytrec_eval`, which is what BEIR itself scores with, so a
number here is comparable to a published one whenever the corpus is (see the
module docstring of `beir.py` for when it is not):

* **nDCG@k** — `grade / log2(rank + 1)`, normalised by the ideal ordering. The gain
  is LINEAR, which is what `trec_eval`'s `ndcg_cut` and therefore BEIR use. The
  exponential `2**grade - 1` form is the other common convention and this module
  shipped it first: identical on binary qrels, so SciFact never noticed, and wrong
  by ~0.04 on the graded 0/1/2 qrels of NFCorpus and TREC-COVID — the datasets the
  graded support exists for. Caught by the randomised `pytrec_eval` cross-check in
  `tests/test_retrieval_metrics.py`, which is why that test is not optional.
* **MAP@k** — mean of P@i at each relevant hit in the top k, divided by the TOTAL
  number of relevant documents, not by how many fit in k. That is `map_cut`'s
  convention and it means MAP@1 cannot reach 1.0 for a query with 2 gold
  documents. Dividing by `min(|rel|, k)` instead would read ~0.1 higher on every
  multi-gold query and would not match anything published.
* **Recall@k**, **P@k** — the obvious ones.
* **MRR@k** — `1/rank` of the first hit within k. Reported per cutoff because that
  is how BEIR reports it (MRR@10), plus an untruncated `mrr` for diagnosis: the
  gap between `mrr@10` and `mrr` is exactly "the gold document came back, below
  the cutoff".
* **Hit@k** — 1.0 if anything relevant is in the top k. BEIR calls it Accuracy@k.
  It is the metric a *user* feels, and the one that saturates first, which is why
  it is reported next to nDCG rather than instead of it.
* **R-precision** — P@R where R is that query's number of relevant documents.
  Cutoff-free, so it is the one number that does not flatter a slice with one gold
  document per query.

Everything here is a pure function of a ranked list of document ids and a qrel
map, so it is unit-testable without a server — see `tests/test_retrieval_metrics.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: The cutoffs reported — BEIR's own set. 10 is the headline; 1 and 3 are where a
#: reranker earns its keep, and 100 is where recall stops being about ranking at
#: all. A change that moves @100 but not @1 is a retrieval change and a change
#: that moves @1 but not @100 is a ranking change; the two need opposite fixes,
#: which is the whole reason for reporting a spread rather than one number.
CUTOFFS = (1, 3, 5, 10, 20, 100)

#: Metrics reported per cutoff, in table order.
PER_CUTOFF = ("nDCG", "MAP", "Recall", "P", "MRR", "Hit")


def dedupe_preserving_order(document_ids: list[str]) -> list[str]:
    """Each document once, at the rank of its best-placed unit.

    One document yields many facts, so a raw result list can spend its whole top
    10 on one abstract. Scoring that against a document-level qrel would report a
    retrieval failure where there was none.
    """
    seen: set[str] = set()
    ranked: list[str] = []
    for document_id in document_ids:
        if document_id not in seen:
            seen.add(document_id)
            ranked.append(document_id)
    return ranked


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Fraction of the relevant documents that appear in the top k."""
    if not relevant:
        return 0.0
    hits = sum(1 for document_id in ranked[:k] if relevant.get(document_id, 0) > 0)
    return hits / len(relevant)


def precision_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Fraction of the top k that is relevant.

    Divided by k, not by `len(ranked[:k])`: a query that returned 3 documents has
    P@10 = 0.3 at best, which is the honest reading — the other seven slots were
    offered and left empty. Normalising by what came back would score a system
    that returns almost nothing as perfectly precise.
    """
    if k <= 0:
        return 0.0
    return sum(1 for document_id in ranked[:k] if relevant.get(document_id, 0) > 0) / k


def average_precision_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """P@i averaged over the relevant hits in the top k, over ALL relevant documents.

    See the module docstring: the denominator is `len(relevant)`, matching
    `pytrec_eval`'s `map_cut`, so a gold document that never arrives costs score
    rather than being quietly excluded from the average.
    """
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, document_id in enumerate(ranked[:k], start=1):
        if relevant.get(document_id, 0) > 0:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """nDCG@k with `trec_eval`'s LINEAR gain and `log2(rank + 1)` discount.

    Linear, not `2**grade - 1`: see the module docstring. The two agree exactly on
    binary qrels and diverge on graded ones, so only a graded dataset shows the
    difference — and only the cross-check against `pytrec_eval` shows it reliably.

    Graded at all (rather than collapsing every label to 1) so the 0/1/2 qrels of
    NFCorpus and TREC-COVID need no second implementation.
    """
    gain = sum(relevant.get(document_id, 0) / math.log2(rank + 2) for rank, document_id in enumerate(ranked[:k]))
    ideal = sum(
        grade / math.log2(rank + 2) for rank, grade in enumerate(sorted(relevant.values(), reverse=True)[:k])
    )
    return (gain / ideal) if ideal else 0.0


def reciprocal_rank_at_k(ranked: list[str], relevant: dict[str, int], k: int | None = None) -> float:
    """1/rank of the first relevant document within k, 0 when none is there.

    `k=None` is the untruncated form, which is how a buried gold document gets
    distinguished from one that never came back at all.
    """
    for rank, document_id in enumerate(ranked if k is None else ranked[:k], start=1):
        if relevant.get(document_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def hit_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """1.0 if anything relevant is in the top k. BEIR's Accuracy@k."""
    return 1.0 if any(relevant.get(document_id, 0) > 0 for document_id in ranked[:k]) else 0.0


def r_precision(ranked: list[str], relevant: dict[str, int]) -> float:
    """P@R, where R is this query's number of relevant documents.

    The only cutoff-free precision here, and therefore the one number a slice with
    one gold document per query cannot flatter: R is 1 there, so it is exactly
    "was the right document first".
    """
    return precision_at_k(ranked, relevant, len(relevant)) if relevant else 0.0


@dataclass(frozen=True)
class QueryScore:
    """One query's scores, kept per query so a failure names the query that caused it."""

    query_id: str
    retrieved: int
    """Distinct documents the response reached, before any cutoff."""
    gold_total: int
    gold_found: int
    """Gold documents anywhere in the response — the ceiling every cutoff is bounded by."""
    first_hit_rank: int | None
    """Rank of the first gold document, or None if none came back. The diagnosis."""
    ndcg: dict[int, float]
    map: dict[int, float]
    recall: dict[int, float]
    precision: dict[int, float]
    mrr_at: dict[int, float]
    hit: dict[int, float]
    mrr: float
    """Untruncated, so `mrr_at[10] == 0 < mrr` reads as "returned, ranked too low"."""
    r_precision: float

    @property
    def blame(self) -> str:
        """Which half a low score came from. Retrieval and ranking need opposite fixes."""
        if self.gold_found < self.gold_total:
            return f"retrieval ({self.gold_found}/{self.gold_total} gold documents reached the response)"
        return f"ranking (every gold document returned; first at rank {self.first_hit_rank})"


def score_query(query_id: str, ranked: list[str], relevant: dict[str, int]) -> QueryScore:
    ranked = dedupe_preserving_order(ranked)
    first_hit = reciprocal_rank_at_k(ranked, relevant)
    return QueryScore(
        query_id=query_id,
        retrieved=len(ranked),
        gold_total=len(relevant),
        gold_found=sum(1 for document_id in ranked if relevant.get(document_id, 0) > 0),
        first_hit_rank=round(1 / first_hit) if first_hit else None,
        ndcg={k: ndcg_at_k(ranked, relevant, k) for k in CUTOFFS},
        map={k: average_precision_at_k(ranked, relevant, k) for k in CUTOFFS},
        recall={k: recall_at_k(ranked, relevant, k) for k in CUTOFFS},
        precision={k: precision_at_k(ranked, relevant, k) for k in CUTOFFS},
        mrr_at={k: reciprocal_rank_at_k(ranked, relevant, k) for k in CUTOFFS},
        hit={k: hit_at_k(ranked, relevant, k) for k in CUTOFFS},
        mrr=first_hit,
        r_precision=r_precision(ranked, relevant),
    )


@dataclass(frozen=True)
class RunScore:
    """The macro-average over queries — the number a floor is set against."""

    queries: int
    ndcg: dict[int, float]
    map: dict[int, float]
    recall: dict[int, float]
    precision: dict[int, float]
    mrr_at: dict[int, float]
    hit: dict[int, float]
    mrr: float
    r_precision: float
    #: Mean distinct documents a response reached. Every @k above it is bounded by
    #: the token budget rather than by ranking, so a flat @20/@100 is a budget
    #: reading and not a quality one — which the table has to show, not hide.
    mean_retrieved: float
    per_query: tuple[QueryScore, ...]

    def _row(self, label: str, values: dict[int, float]) -> str:
        return f"{label:<8}" + "".join(f"{values[k]:>8.3f}" for k in CUTOFFS)

    def table(self) -> str:
        rows = [
            f"{'metric':<8}" + "".join(f"{f'@{k}':>8}" for k in CUTOFFS),
            self._row("nDCG", self.ndcg),
            self._row("MAP", self.map),
            self._row("Recall", self.recall),
            self._row("P", self.precision),
            self._row("MRR", self.mrr_at),
            self._row("Hit", self.hit),
        ]
        rows.append(f"\nMRR (untruncated) {self.mrr:.3f}   R-precision {self.r_precision:.3f}")
        rows.append(f"queries {self.queries}   mean documents returned {self.mean_retrieved:.1f}")
        return "\n".join(rows)


def as_record(run: RunScore, label: str, notes: dict[str, str]) -> dict[str, object]:
    """One run, flat enough to compare several and plot them.

    A dict rather than a model because the consumer is a chart script and a human
    reading JSON, and the per-cutoff keys are integers that would have to be
    stringified for any schema anyway. `notes` carries what configuration produced
    it — a run record without that is a number nobody can reproduce.
    """
    return {
        "label": label,
        "notes": notes,
        "queries": run.queries,
        "mean_retrieved": run.mean_retrieved,
        "mrr_untruncated": run.mrr,
        "r_precision": run.r_precision,
        "cutoffs": list(CUTOFFS),
        "ndcg": {str(k): run.ndcg[k] for k in CUTOFFS},
        "map": {str(k): run.map[k] for k in CUTOFFS},
        "recall": {str(k): run.recall[k] for k in CUTOFFS},
        "precision": {str(k): run.precision[k] for k in CUTOFFS},
        "mrr_at": {str(k): run.mrr_at[k] for k in CUTOFFS},
        "hit": {str(k): run.hit[k] for k in CUTOFFS},
        "per_query": [
            {
                "query_id": q.query_id,
                "ndcg_10": q.ndcg[10],
                "mrr": q.mrr,
                "gold_found": q.gold_found,
                "gold_total": q.gold_total,
                "retrieved": q.retrieved,
                "first_hit_rank": q.first_hit_rank,
                "blame": q.blame,
            }
            for q in run.per_query
        ],
    }


def aggregate(scores: list[QueryScore]) -> RunScore:
    """Macro-average, which is what BEIR reports — every query counts the same."""
    count = len(scores)

    def mean(values: list[float]) -> float:
        return (sum(values) / count) if count else 0.0

    def per_cutoff(pick) -> dict[int, float]:  # noqa: ANN001 - a QueryScore attribute getter
        return {k: mean([pick(s)[k] for s in scores]) for k in CUTOFFS}

    return RunScore(
        queries=count,
        ndcg=per_cutoff(lambda s: s.ndcg),
        map=per_cutoff(lambda s: s.map),
        recall=per_cutoff(lambda s: s.recall),
        precision=per_cutoff(lambda s: s.precision),
        mrr_at=per_cutoff(lambda s: s.mrr_at),
        hit=per_cutoff(lambda s: s.hit),
        mrr=mean([s.mrr for s in scores]),
        r_precision=mean([s.r_precision for s in scores]),
        mean_retrieved=mean([float(s.retrieved) for s in scores]),
        per_query=tuple(scores),
    )
