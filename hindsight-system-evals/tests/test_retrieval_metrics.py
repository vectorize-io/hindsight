"""The metric arithmetic, two ways: worked out by hand, and against `pytrec_eval`.

The eval that uses these needs a real model and minutes per run, so a wrong gain
function would only ever surface as a mysteriously low nDCG. These run in under a
second.

Both halves are needed, which the history shows. The hand-written cases pin the
definitions a reader can check; the randomised cross-check at the bottom caught
nDCG shipping with the exponential gain instead of `trec_eval`'s linear one —
invisible to every hand-written case, because they were all binary, where the two
forms agree exactly.
"""

from __future__ import annotations

import math
import random

import pytest

from hindsight_system_evals.beir import BeirDocument, BeirQuery, BeirSlice, cut_slice, load_beir_slice
from hindsight_system_evals.retrieval_metrics import (
    CUTOFFS,
    aggregate,
    average_precision_at_k,
    dedupe_preserving_order,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
    reciprocal_rank_at_k,
    score_query,
)


def test_dedupe_keeps_each_document_at_its_best_rank():
    assert dedupe_preserving_order(["a", "a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_recall_at_k_counts_only_the_top_k():
    ranked = ["x", "gold1", "y", "z", "gold2"]
    relevant = {"gold1": 1, "gold2": 1}
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 5) == 1.0
    # No labels means nothing to find; 0.0 rather than a ZeroDivisionError.
    assert recall_at_k(ranked, {}, 10) == 0.0


def test_ndcg_is_one_when_the_gold_document_ranks_first():
    assert ndcg_at_k(["gold"], {"gold": 1}, 10) == 1.0


def test_ndcg_applies_the_log2_discount():
    # Gold at rank 2: gain (2**1-1)/log2(3), ideal (2**1-1)/log2(2) = 1.
    assert ndcg_at_k(["miss", "gold"], {"gold": 1}, 10) == 1 / math.log2(3)


def test_ndcg_rewards_the_higher_grade_first():
    """Graded labels: the 2 belongs above the 1, and ordering them wrong costs score."""
    relevant = {"strong": 2, "weak": 1}
    right = ndcg_at_k(["strong", "weak"], relevant, 10)
    wrong = ndcg_at_k(["weak", "strong"], relevant, 10)
    assert right == 1.0
    assert wrong < right


def test_ndcg_is_zero_when_nothing_relevant_is_returned():
    assert ndcg_at_k(["a", "b"], {"gold": 1}, 10) == 0.0


def test_reciprocal_rank_reports_the_first_hit():
    assert reciprocal_rank_at_k(["a", "gold"], {"gold": 1}) == 0.5
    assert reciprocal_rank_at_k(["a", "b"], {"gold": 1}) == 0.0


def test_reciprocal_rank_at_k_ignores_a_hit_past_the_cutoff():
    """The gap between MRR@k and untruncated MRR is the whole ranking diagnosis."""
    ranked = ["a", "b", "c", "gold"]
    assert reciprocal_rank_at_k(ranked, {"gold": 1}, 3) == 0.0
    assert reciprocal_rank_at_k(ranked, {"gold": 1}) == 0.25


def test_precision_divides_by_k_not_by_what_came_back():
    # One gold at rank 1 out of a 2-document response: P@10 is 0.1, not 0.5.
    # Normalising by the response length would call a system that returns almost
    # nothing perfectly precise.
    assert precision_at_k(["gold", "x"], {"gold": 1}, 10) == 0.1
    assert precision_at_k(["gold", "x"], {"gold": 1}, 1) == 1.0


def test_average_precision_divides_by_total_relevant_not_by_k():
    """`map_cut`'s convention: MAP@1 cannot reach 1.0 with two gold documents."""
    relevant = {"g1": 1, "g2": 1}
    # Both gold, ranks 1 and 2: (1/1 + 2/2) / 2 = 1.0
    assert average_precision_at_k(["g1", "g2"], relevant, 10) == 1.0
    # Truncated at 1, only the first hit counts, still divided by 2.
    assert average_precision_at_k(["g1", "g2"], relevant, 1) == 0.5
    # A gap costs: ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert average_precision_at_k(["g1", "x", "g2"], relevant, 10) == (1 + 2 / 3) / 2


def test_hit_at_k_saturates_on_the_first_relevant_document():
    assert hit_at_k(["x", "gold"], {"gold": 1}, 2) == 1.0
    assert hit_at_k(["x", "gold"], {"gold": 1}, 1) == 0.0
    # Two gold documents, one found: Hit is 1.0 where Recall@2 is only 0.5. That
    # difference is why both are reported.
    assert hit_at_k(["g1", "x"], {"g1": 1, "g2": 1}, 2) == 1.0
    assert recall_at_k(["g1", "x"], {"g1": 1, "g2": 1}, 2) == 0.5


def test_r_precision_uses_the_querys_own_relevant_count():
    # R = 2, and one of the top 2 is relevant.
    assert r_precision(["g1", "x", "g2"], {"g1": 1, "g2": 1}) == 0.5
    # R = 1: exactly "was the right document first".
    assert r_precision(["gold", "x"], {"gold": 1}) == 1.0
    assert r_precision(["x", "gold"], {"gold": 1}) == 0.0


def test_score_query_blames_retrieval_when_a_gold_document_never_came_back():
    score = score_query("q1", ["a", "b"], {"gold": 1, "other": 1})
    assert score.gold_found == 0
    assert "retrieval" in score.blame


def test_score_query_blames_ranking_when_every_gold_document_came_back_too_late():
    ranked = [f"filler{i}" for i in range(30)] + ["gold"]
    score = score_query("q1", ranked, {"gold": 1})
    assert score.gold_found == 1
    assert score.recall[20] == 0.0
    assert score.recall[100] == 1.0
    # Returned but below the cutoff — the gap MRR@20 == 0 < mrr says exactly that.
    assert score.mrr_at[20] == 0.0
    assert score.first_hit_rank == 31
    assert "ranking" in score.blame
    assert "rank 31" in score.blame


def test_aggregate_macro_averages_every_query_equally():
    perfect = score_query("a", ["gold"], {"gold": 1})
    missed = score_query("b", ["nope"], {"gold": 1})
    run = aggregate([perfect, missed])
    assert run.queries == 2
    assert run.ndcg[10] == 0.5
    assert run.map[10] == 0.5
    assert run.hit[10] == 0.5
    assert run.mrr == 0.5
    assert run.r_precision == 0.5
    assert run.mean_retrieved == 1.0


def test_every_per_cutoff_metric_is_reported_at_every_cutoff():
    """A new cutoff must reach every metric, not just the one someone remembered."""
    run = aggregate([score_query("a", ["gold"], {"gold": 1})])
    for values in (run.ndcg, run.map, run.recall, run.precision, run.mrr_at, run.hit):
        assert set(values) == set(CUTOFFS)
    assert run.table().count("@") == len(CUTOFFS)


def test_committed_slice_loads_and_every_query_has_a_gold_document_in_the_corpus():
    sliced = load_beir_slice()
    corpus = {doc.id for doc in sliced.documents}
    assert sliced.queries, "the committed slice has no queries"
    for query in sliced.queries:
        assert query.relevant, f"query {query.id} has no relevance labels"
        assert set(query.relevant) <= corpus, f"query {query.id} points outside the slice's corpus"


def test_cut_slice_keeps_every_gold_document_and_pads_with_distractors():
    documents = tuple(BeirDocument(id=str(i), title="t", text="x") for i in range(10))
    full = BeirSlice(
        name="toy",
        documents=documents,
        queries=(BeirQuery(id="1", text="q", relevant={"7": 1}), BeirQuery(id="2", text="q", relevant={"3": 1})),
    )
    sliced = cut_slice(full, queries=1, documents=4)
    ids = {doc.id for doc in sliced.documents}
    assert len(sliced.queries) == 1
    assert "7" in ids  # the kept query's gold document survives
    assert len(ids) == 4


def test_reads_a_beir_directory_and_coerces_integer_ids(tmp_path, monkeypatch):
    """The `HINDSIGHT_EVAL_BEIR_DIR` path, which the committed fixture never exercises.

    The integer `_id` is the case worth pinning: BEIR spells it as a string in
    SciFact and as a bare number elsewhere, and the qrels TSV is always text — so
    without coercion every lookup misses and the run scores zero while looking
    like a retrieval collapse.
    """
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        '{"_id": 10, "title": "T", "text": "body"}\n{"_id": "11", "title": "", "text": "other"}\n',
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text('{"_id": 1, "text": "a claim", "metadata": {}}\n', encoding="utf-8")
    # The 0-grade row and the row pointing outside the corpus must both be dropped.
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n1\t10\t2\n1\t99\t1\n1\t11\t0\n", encoding="utf-8"
    )

    monkeypatch.setenv("HINDSIGHT_EVAL_BEIR_DIR", str(tmp_path))
    sliced = load_beir_slice()

    assert [doc.id for doc in sliced.documents] == ["10", "11"]
    assert sliced.documents[0].content == "T\n\nbody"
    # No title: content must not start with the blank line the separator would leave.
    assert sliced.documents[1].content == "other"
    assert len(sliced.queries) == 1
    assert sliced.queries[0].relevant == {"10": 2}


def test_query_limit_keeps_the_whole_corpus(tmp_path, monkeypatch):
    """A smoke run must face the same distractors, or its number is not reproducible."""
    monkeypatch.setenv("HINDSIGHT_EVAL_BEIR_QUERIES", "2")
    sliced = load_beir_slice()
    assert len(sliced.queries) == 2
    assert len(sliced.documents) == 500


def test_matches_pytrec_eval_on_random_runs():
    """Cross-check every metric against the implementation BEIR itself scores with.

    The hand-written asserts above pin the arithmetic on cases a human chose, which
    is exactly the wrong shape for this class of bug: `nDCG` shipped with the
    exponential `2**grade - 1` gain instead of `trec_eval`'s linear one, and every
    hand-written case was binary, where the two are identical. Only graded labels
    on inputs nobody picked show it. 300 random runs do, in under a second.

    Skipped rather than failed without the extra (`uv sync --extra
    reference-metrics`) — it is a C extension, and a missing wheel on some
    interpreter must not take the suite down with it.
    """
    pytrec_eval = pytest.importorskip("pytrec_eval", reason="uv sync --extra reference-metrics")

    cuts = ",".join(str(k) for k in CUTOFFS)
    measures = {f"ndcg_cut.{cuts}", f"map_cut.{cuts}", f"recall.{cuts}", f"P.{cuts}", "recip_rank", "Rprec"}
    rng = random.Random(7)

    for trial in range(300):
        documents = [f"d{i}" for i in range(rng.randint(1, 40))]
        ranked = rng.sample(documents, k=rng.randint(1, len(documents)))
        # Grades of 1 AND 2, deliberately: a binary-only generator reproduces the
        # blind spot this test exists to cover.
        relevant = {d: rng.choice([1, 1, 2]) for d in rng.sample(documents, k=rng.randint(1, min(4, len(documents))))}

        ours = score_query("q", ranked, relevant)
        # trec_eval ranks by descending score, so rank 1 gets the highest.
        run = {"q": {d: float(len(ranked) - i) for i, d in enumerate(ranked)}}
        theirs = pytrec_eval.RelevanceEvaluator({"q": relevant}, measures).evaluate(run)["q"]

        context = f"trial {trial}: ranked={ranked[:8]} relevant={relevant}"
        for k in CUTOFFS:
            assert ours.ndcg[k] == pytest.approx(theirs[f"ndcg_cut_{k}"]), f"nDCG@{k} — {context}"
            assert ours.map[k] == pytest.approx(theirs[f"map_cut_{k}"]), f"MAP@{k} — {context}"
            assert ours.recall[k] == pytest.approx(theirs[f"recall_{k}"]), f"Recall@{k} — {context}"
            assert ours.precision[k] == pytest.approx(theirs[f"P_{k}"]), f"P@{k} — {context}"
        assert ours.mrr == pytest.approx(theirs["recip_rank"]), f"MRR — {context}"
        assert ours.r_precision == pytest.approx(theirs["Rprec"]), f"R-precision — {context}"


def test_reads_a_beir_directory_with_non_numeric_query_ids(tmp_path, monkeypatch):
    """NFCorpus names its queries (`PLAIN-3`), SciFact numbers them (`3`).

    The loader sorted with `int(query_id)`, which raised on the first dataset that
    did not use numbers. Both orderings have to work, and numeric ids must sort
    numerically — string order would put "10" before "2" and silently change which
    queries `cut_slice` keeps.
    """
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        '{"_id": "MED-1", "title": "A", "text": "a"}\n{"_id": "MED-2", "title": "B", "text": "b"}\n',
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text(
        '{"_id": "PLAIN-10", "text": "q ten"}\n{"_id": "PLAIN-2", "text": "q two"}\n', encoding="utf-8"
    )
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nPLAIN-10\tMED-1\t2\nPLAIN-2\tMED-2\t1\n", encoding="utf-8"
    )

    monkeypatch.setenv("HINDSIGHT_EVAL_BEIR_DIR", str(tmp_path))
    sliced = load_beir_slice()

    assert [q.id for q in sliced.queries] == ["PLAIN-10", "PLAIN-2"]  # non-numeric: plain string order
    assert sliced.queries[0].relevant == {"MED-1": 2}  # the grade-2 label survives


def test_numeric_query_ids_sort_numerically(tmp_path, monkeypatch):
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text('{"_id": "d1", "title": "", "text": "x"}\n', encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text('{"_id": "2", "text": "a"}\n{"_id": "10", "text": "b"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n10\td1\t1\n2\td1\t1\n", encoding="utf-8")

    monkeypatch.setenv("HINDSIGHT_EVAL_BEIR_DIR", str(tmp_path))
    assert [q.id for q in load_beir_slice().queries] == ["2", "10"]
