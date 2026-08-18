"""Synthetic multi-bank union-CE bench + quality overlap.

The wall-clock bench is skipped unless ``HS_BENCH=1``. The quality test always
runs: it is the arithmetic/ranking proof, not a live API measurement.

Corpus (fixed, no LLM)::

    query = "multi bank recall"
    5 banks x 8 facts. Gold facts sit at RRF rank 0 of each bank and share
    the query tokens. Remaining facts are decoys with no query-token overlap.

    Fake CE score = |query tokens ∩ doc tokens| / |query tokens|.
    Union cap 200 admits every fact (5*8=40), so union-CE top-5 vs
    per-bank-CE-then-score-merge top-5 must overlap >= 4/5 (expect 5/5).
"""

from __future__ import annotations

import os
import time

import pytest

from hindsight_api.engine.multi_bank_recall import (
    apply_per_bank_floor,
    score_merge,
    select_union_candidates,
    stamp_union_ce_scores,
)
from hindsight_api.engine.response_models import MemoryFact, RecallScores

QUERY = "multi bank recall"

GOLDS = {
    "bank0": "multi bank recall union pass",
    "bank1": "multi bank recall score merge",
    "bank2": "multi bank recall one cross encoder",
    "bank3": "multi bank recall worker isolation",
    "bank4": "multi bank recall token budget",
}


def _ce_score(query: str, text: str) -> float:
    q = set(query.lower().split())
    t = set(text.lower().split())
    return len(q & t) / max(1, len(q))


def _fact(fact_id: str, text: str, *, bank_id: str) -> MemoryFact:
    return MemoryFact(id=fact_id, text=text, fact_type="world", bank_id=bank_id)


def _corpus() -> list[tuple[str, list[MemoryFact]]]:
    banks: list[tuple[str, list[MemoryFact]]] = []
    for i in range(5):
        bid = f"bank{i}"
        facts = [_fact(f"{bid}-gold", GOLDS[bid], bank_id=bid)]
        facts.extend(_fact(f"{bid}-d{j}", f"decoy filler sentence {j} about weather", bank_id=bid) for j in range(7))
        banks.append((bid, facts))
    return banks


def _score_facts(query: str, facts: list[MemoryFact]) -> list[MemoryFact]:
    scored: list[MemoryFact] = []
    for fact in facts:
        value = _ce_score(query, fact.text)
        scored.append(fact.model_copy(update={"scores": RecallScores(final=value, reranker=value)}))
    return scored


def test_union_ce_top5_overlaps_per_bank_ce() -> None:
    """Quality: union-CE top-5 vs per-bank-CE top-5 overlap >= 4/5 on the corpus above."""
    banks = _corpus()
    per_bank_scored = [(bid, _score_facts(QUERY, facts)) for bid, facts in banks]
    per_bank_top = [f.id for f in score_merge(per_bank_scored)[:5]]

    pool = select_union_candidates(banks, per_bank_pre_cap=50, union_cap=100, k_floor=1)
    union_scored = _score_facts(QUERY, pool.facts)
    union_scored = stamp_union_ce_scores(
        union_scored,
        [f"{i}:{f.bank_id}:{f.id}" for i, f in enumerate(union_scored)],
        {f"{i}:{f.bank_id}:{f.id}": (f.scores.reranker or 0.0) for i, f in enumerate(union_scored)},
    )
    floored = apply_per_bank_floor(union_scored, k_floor=1, keep=100, bank_order=[b for b, _ in banks])
    by_bank: dict[str, list[MemoryFact]] = {}
    for fact in floored:
        by_bank.setdefault(fact.bank_id or "", []).append(fact)
    union_top = [f.id for f in score_merge([(bid, by_bank.get(bid, [])) for bid, _ in banks])[:5]]

    overlap = len(set(per_bank_top) & set(union_top))
    assert overlap >= 4, (per_bank_top, union_top, overlap)


@pytest.mark.hs_bench
@pytest.mark.skipif(os.environ.get("HS_BENCH") != "1", reason="set HS_BENCH=1 to run the union-CE wall bench")
def test_union_ce_wall_under_half_per_bank() -> None:
    """5 banks x 60 candidates; fake CE sleeps per pair. union wall < 0.5 x per-bank wall."""
    sleep_s = 0.002
    n_banks = 5
    n_cands = 60

    def fake_predict(n_pairs: int) -> None:
        time.sleep(sleep_s * n_pairs)

    banks = [
        (f"bank{i}", [_fact(f"b{i}-{j}", f"text {i} {j}", bank_id=f"bank{i}") for j in range(n_cands)])
        for i in range(n_banks)
    ]

    t0 = time.perf_counter()
    for _bid, facts in banks:
        fake_predict(len(facts))
    per_bank_wall = time.perf_counter() - t0

    t1 = time.perf_counter()
    pool = select_union_candidates(banks, per_bank_pre_cap=50, union_cap=100, k_floor=1)
    fake_predict(len(pool.facts))
    union_wall = time.perf_counter() - t1

    assert len(pool.facts) == 100
    print(
        f"HS_BENCH union_ce: union={union_wall:.4f}s per_bank={per_bank_wall:.4f}s "
        f"ratio={union_wall / per_bank_wall:.3f} sleep={sleep_s} "
        f"banks={n_banks} cands={n_cands} union_cap=100 pre_cap=50 k_floor=1"
    )
    assert union_wall < 0.5 * per_bank_wall, (
        f"union={union_wall:.3f}s per_bank={per_bank_wall:.3f}s sleep={sleep_s} banks={n_banks} cands={n_cands} cap=100"
    )
