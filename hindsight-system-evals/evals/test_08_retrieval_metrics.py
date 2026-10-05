"""A documented retrieval baseline: nDCG and friends on BEIR SciFact.

The only suite here graded against labels nobody in this repo wrote, and the only
one that makes **no model call**. Every other eval judges an answer with an LLM;
this one scores `recall` against BEIR SciFact's own relevance judgements —
scientific claims as queries, abstracts as documents — and reports the standard
set at k = 1, 3, 5, 10, 20, 100: nDCG, MAP, Recall, P, MRR and Hit, plus
untruncated MRR and R-precision. Definitions follow `pytrec_eval`, which is what
BEIR itself scores with, and `tests/test_retrieval_metrics.py` cross-checks every
one of them against it.

**This records a baseline; it does not gate CI.** It lives in `system-evals`,
which runs on perf-test's nightly cron rather than on pull requests, and the
floors below exist so a nightly run says something when retrieval moves — not so
a merge is blocked. The numbers and how they were measured are in the block above
the floors, which is the point of the file: somewhere to look up what this
pipeline scored, and on what.

It is also the one suite here that needs no credentials, because the memories
were extracted once by a real model and frozen as
`fixtures/beir-scifact-bank.zip` (500 abstracts -> 1,812 memories). The story
restores that archive instead of retaining, so nothing calls an LLM:

    HINDSIGHT_EVAL_LLM_PROVIDER=none HINDSIGHT_EVAL_LLM_MODEL=none \
      uv run pytest tests evals/test_08_retrieval_metrics.py

It stays in `system-evals` rather than `system-tests` for one reason: that suite
stubs the embedder AND the reranker with lexical stand-ins, and an nDCG measured
over hashed word overlap is a fact about blake2b. The models this grades have to
be the ones that ship, which is what `system-evals` already runs.

Two facts about `recall` shape the scoring. It returns facts, several per
document, while a qrel labels a document — so units are collapsed onto their
source at the rank of the best-placed one. And it takes no `k`, budgeting by
`max_tokens` — so the list is widened with tokens and cut at k in the scorer.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest

from hindsight_system_evals import wait_until_settled
from hindsight_system_evals.beir import load_beir_slice
from hindsight_system_evals.retrieval import (
    RECALL_BUDGET,
    RECALL_MAX_TOKENS,
    prunes_candidates,
    read_manifest,
    restore_bank,
    run_label,
    run_notes,
    score,
    worst_queries,
)
from hindsight_system_evals.retrieval_metrics import as_record

pytestmark = pytest.mark.asyncio

#: Floors, not targets, and set under a MEASURED RANGE rather than one number.
#:
#: Measured over repeated runs of `fixtures/beir-scifact-bank.zip` (500 BEIR
#: SciFact abstracts -> 1,812 memories, `concise` extraction by
#: gemini-3.1-flash-lite) on the default local embedder and reranker:
#:
#:     metric        @1      @3      @5     @10     @20    @100
#:     nDCG       0.800   0.870   0.884   0.887   0.887   0.892     <- upper state
#:     nDCG       0.800   0.857   0.872   0.874   0.874   0.880     <- lower state
#:     MAP@10     0.860 / 0.850    Recall@10  0.956 / 0.936
#:     MRR        0.866 / 0.857    R-precision      0.802 (both)
#:     documents returned          159.8 / 137.2
#:
#: **The run-to-run spread is real and not yet explained.** Runs land in one of
#: two states, never between them. What is ruled out, by measurement rather than
#: by argument: it is not a partial restore (the memory count reaches 1,812 either
#: way), not embedding lag (the numbers do not converge if you wait — a probe
#: query is stable within a session from the first reading), not ANN recall
#: (deferring the vector index so every search is exact does not remove it), and
#: not CPU-vs-MPS float drift (forcing both models onto CPU does not remove it).
#:
#: A trace comparison across fresh sessions shows the per-arm candidate COUNTS are
#: identical every time — semantic 1000, bm25 87, graph 360 — while the documents
#: that survive differ by a few. Identical pool sizes with different membership
#: points at the import rather than at retrieval: entity links are rebuilt on
#: restore and feed the graph arm. That is a hypothesis, not a finding.
#:
#: So the floors sit below the LOWER state, which is the honest thing to assert
#: until the spread is understood: this still catches a real regression (anything
#: that drops ranking by more than ~5%) and will not produce a random red build.
#: Tighten them to the upper state once a run is reproducible — that is worth
#: doing, because the gate is much weaker than it looks like it should be.
#:
#: NOT comparable to a published SciFact nDCG@10 (~0.70 for a strong dense
#: retriever): that is 5,183 abstracts, this is 500, and this ranks extracted
#: claims rather than the abstracts themselves.
MIN_NDCG_AT_1 = 0.74
MIN_NDCG_AT_10 = 0.83
MIN_MAP_AT_10 = 0.80
MIN_RECALL_AT_10 = 0.88
MIN_HIT_AT_10 = 0.88
#: Cutoff-free, so the one floor a mostly-one-gold-per-query slice cannot flatter:
#: it is "was the right document first". Identical in both states.
MIN_R_PRECISION = 0.74


#: Restoring the archive re-embeds all 1,812 memories with a real model, which is
#: minutes of local CPU — the suite's 90s settle is sized for a stub that answers
#: instantly and would time out here long before the import is wrong.
RESTORE_TIMEOUT_SECONDS = 1800.0


@pytest.fixture
async def restored_bank(client, bank_id):
    """The archive restored into a fresh bank, cleaned up afterwards on a remote tenant.

    A fixture rather than a call in the test body because the import creates the
    target bank BEFORE it streams memories into it: a failure part-way leaves a
    half-filled bank behind, and only a teardown that runs on the error path can
    clear it. Found the hard way — an interrupted import against a shared server
    left a bank holding 7,029 of 11,139 memories with nothing to delete it.
    """
    async def settle(bank: str) -> None:
        await wait_until_settled(client, bank, timeout=RESTORE_TIMEOUT_SECONDS)

    try:
        yield await restore_bank(client, bank_id, settle)
    finally:
        # The import creates the target bank BEFORE it streams memories into it, so
        # a failure part-way leaves a half-filled bank behind and only a teardown
        # on the error path clears it. Found the hard way against a shared remote
        # tenant, which kept a bank holding 7,029 of 11,139 memories.
        with contextlib.suppress(Exception):
            await client.adelete_bank(f"{bank_id}-beir")


async def test_recall_ranks_beir_scifact(request, client, restored_bank):
    sliced = load_beir_slice()
    manifest = read_manifest()
    print(
        f"\nslice: {sliced.describe()}"
        f"\nbank:  {manifest.describe() if manifest else 'no manifest beside the archive'}"
        f"\nrecall: budget={RECALL_BUDGET} max_tokens={RECALL_MAX_TOKENS}"
    )

    target = restored_bank
    run = await score(client, target, sliced)

    print(f"\n{run.table()}\n\nworst queries:\n{worst_queries(run)}\n")

    # Written BEFORE the assertions, like the other suites record before grading:
    # an arm that trips a floor is exactly the arm whose numbers are wanted. This
    # is how two rerankers or two embedders get compared at all — each needs its
    # own server process, so each is its own run, and the JSON is what lines them
    # up afterwards. It carries the configuration that produced it, because a run
    # record without that is a number nobody can reproduce.
    if output := request.config.getoption("--retrieval-output"):
        record = as_record(run, label=run_label(), notes=run_notes(manifest))
        Path(output).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {output}")

    # A query that returned nothing at all is not a low score, it is a broken
    # pipeline — and it would drag every average down while looking like a quality
    # problem. Asserted separately so the two cannot be confused.
    empty = [s.query_id for s in run.per_query if s.retrieved == 0]
    assert not empty, f"recall returned no attributable documents for {empty}"

    # The whole scoring scheme rests on `document_id` surviving export and import:
    # if the archive renumbered documents, every id would miss the qrels and the
    # run would read as a total retrieval collapse rather than as a broken
    # fixture. Checked against the gold ceiling, which is the half that matters.
    reachable = sum(1 for s in run.per_query if s.gold_found > 0)
    assert reachable > len(run.per_query) // 2, (
        f"only {reachable}/{len(run.per_query)} queries reached ANY gold document — "
        "the archive's document_ids probably no longer match the slice's"
    )

    # `recall` budgets by tokens, not by k, so a cutoff above the number of
    # documents a response actually reaches measures the budget instead of the
    # ranking — and would read as a flat, flattering line across @20 and @100.
    # Guarded rather than commented, because the budget is a default someone will
    # change.
    #
    # Except when the reranker prunes: dropping the candidates it judges
    # irrelevant is the whole point of that mode, and it returns ~1.5 documents
    # instead of ~160. Asserting the budget there sent the reader to the wrong
    # knob — the list was short by design, not by starvation — so the prune arm
    # gets the assertion that actually matters for it instead: the kept set must
    # still contain the gold document most of the time, which is the only thing
    # that makes aggressive pruning safe.
    if prunes_candidates():
        assert run.recall[10] >= MIN_RECALL_AT_10, (
            f"pruning kept {run.mean_retrieved:.1f} documents per query and Recall@10 fell to "
            f"{run.recall[10]:.3f} — it is dropping gold documents, not just noise"
        )
    else:
        assert run.mean_retrieved >= 100, (
            f"responses reached only {run.mean_retrieved:.1f} documents on average, so the metrics at "
            f"@100 are bounded by max_tokens={RECALL_MAX_TOKENS} rather than by ranking — raise it"
        )

    assert run.ndcg[1] >= MIN_NDCG_AT_1, f"nDCG@1 {run.ndcg[1]:.3f} < {MIN_NDCG_AT_1}"
    assert run.ndcg[10] >= MIN_NDCG_AT_10, f"nDCG@10 {run.ndcg[10]:.3f} < {MIN_NDCG_AT_10}"
    assert run.map[10] >= MIN_MAP_AT_10, f"MAP@10 {run.map[10]:.3f} < {MIN_MAP_AT_10}"
    assert run.recall[10] >= MIN_RECALL_AT_10, f"Recall@10 {run.recall[10]:.3f} < {MIN_RECALL_AT_10}"
    assert run.hit[10] >= MIN_HIT_AT_10, f"Hit@10 {run.hit[10]:.3f} < {MIN_HIT_AT_10}"
    assert run.r_precision >= MIN_R_PRECISION, f"R-precision {run.r_precision:.3f} < {MIN_R_PRECISION}"
