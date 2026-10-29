"""Recall: which memories come back for a query, and how many times.

Each case has a ``category`` and its test id is ``<category>/<case id>``, so
``-k "duplicates/"`` runs one group and ``eval-table`` compares runs per group:

* semantic — meaning over words: a paraphrase, a query sharing no word with its memory;
* keyword — exact identifiers an embedder blurs: error codes, order ids, rare names;
* temporal — "last month", "yesterday", a year, a named week, against ``query_timestamp``;
* currency — the newer of two conflicting memories ranks first;
* multi_hop — the answer is one entity or cause away from what the query names;
* language — non-English queries over memories in that language;
* precision — the right number among near-identical ones; a negation kept;
* duplicates — one fact stated several times comes back several times, unless the
  decision model prunes (``same-sentence-four-documents``, ``reworded-repeats``);
* distinct — memories that only look alike are all kept, by recall and by the
  pruning (``alike-but-distinct``, ``same-subject``).

The capability cases are in ``hindsight_system_evals.recall_capabilities``, each naming
the history behind it. Recall itself calls no LLM by default — dates in a query are read
by a rule-based parser and ranking is the embedder plus a reranker — and each case's
memories are imported from the archive a real extraction recorded once, so a run makes
no model call and grades with none.

The benchmark at the bottom (``test_recall_benchmark_beir_scifact``) is not a case: it
scores recall against BEIR SciFact's own labels and makes no model call.

Duplicates — Recall returns a fact once per document that stated it — unless the decision model prunes.

Recall does not merge facts; observations do. With observations and consolidation
off, the same statement retained as several documents becomes that many facts, and
a recall about it hands back every copy. That is the default, and the first test pins
it: it fails if recall starts returning the fact once, because then something
deduplicates and the reason should be known.

``HINDSIGHT_API_RECALL_DECISION_MODEL_PRUNE`` (#6131) asks the decision model to drop a
candidate that repeats a better-ranked one, or that is no help at all. The second test
stresses that against pools built to tempt it into the wrong drop: a reworded repeat
must go, but a different customer signing up for the same beta, or a different detail
about the same offsite, reads alike and must stay. Run it with the decision model set up:

    HINDSIGHT_EVAL_SET_DECISION_MODEL_PROVIDER=typesafe \\
    HINDSIGHT_EVAL_SET_DECISION_MODEL_API_KEY=... \\
    HINDSIGHT_EVAL_SET_RECALL_DECISION_MODEL_PRUNE=true uv run pytest evals/recall

BEIR SciFact benchmark — A documented retrieval baseline: nDCG and friends on BEIR SciFact.

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
      uv run pytest tests evals/recall/test_08_retrieval_metrics.py

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
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import evaluate, wait_until_settled
from hindsight_system_evals.beir import load_beir_slice
from hindsight_system_evals.pages import SettleFn, prepare_bank
from hindsight_system_evals.recall_capabilities import CASES, Case, run_case
from hindsight_system_evals.report import RECORDED, EvalRecord, case_params
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

log = logging.getLogger(__name__)

PRUNING = os.getenv("HINDSIGHT_EVAL_SET_RECALL_DECISION_MODEL_PRUNE", "").lower() == "true"
DEDUP = os.getenv("HINDSIGHT_EVAL_SET_RECALL_DECISION_MODEL_DEDUP", "").lower() == "true"

RETAIN_REPEATS = 4

#: One atomic fact, so every fact from these documents is a copy of the same statement.
CONTENT = "Marta Rossi's favourite tea is genmaicha, which she drinks every morning."
QUERY = "What tea does Marta Rossi like?"


# ── Capabilities ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", case_params(CASES, {c.id for c in CASES if c.minimum}))
async def test_recall(client: Hindsight, bank_id: str, settled: SettleFn, case: Case) -> None:
    outcome = await run_case(client, bank_id, settled, case)
    RECORDED.append(
        EvalRecord(
            kind="recall",
            question_id=case.id,
            category=case.category,
            correct=not outcome.failed,
            hit_trap=False,
            bank_id=bank_id,
            reason="; ".join(outcome.failed) or "ok",
        )
    )
    log.info("%s [%s] %r:\n%s", case.id, case.category, case.query, outcome.rendered)
    evidence = (
        f"bank {bank_id} ({case.origin})\nquery: {case.query!r} now={case.query_timestamp}\n"
        f"results:\n{outcome.rendered}"
    )
    assert not outcome.failed, f"{case.id}:\n" + "\n".join(outcome.failed) + f"\n{evidence}"


# ── Duplicates: one statement, several documents ─────────────────────────────
@pytest.mark.parametrize(
    "case", [pytest.param("same-sentence-four-documents", id="duplicates/same-sentence-four-documents")]
)
async def test_recall_duplicates(client: Hindsight, bank_id: str, settled: SettleFn, case: str) -> None:
    """Real extraction, so the copies are the model's own rewordings, dates and all."""
    await client.aupdate_bank_config(bank_id, enable_observations=False, enable_auto_consolidation=False)
    await client.aretain_batch(
        bank_id=bank_id,
        items=[{"content": CONTENT, "document_id": f"tea-{i}"} for i in range(RETAIN_REPEATS)],
    )
    await settled(bank_id)

    recalled = await client.arecall(bank_id, QUERY)
    results = "\n".join(f"- [{r.type}] (doc {r.document_id}) {r.text}" for r in recalled.results)
    documents = {r.document_id for r in recalled.results if r.document_id}
    log.info("recall returned %d results from %d documents:\n%s", len(recalled.results), len(documents), results)

    # The judge, not string equality: each document is extracted separately, so the
    # copies are paraphrases of one another rather than identical strings.
    verdict = await evaluate(
        results,
        "At least two of the listed results state the same fact about Marta Rossi's favourite tea.",
        context=f"The same sentence was retained as {RETAIN_REPEATS} separate documents: {CONTENT!r}",
    )
    where = f"{len(documents)} documents among the results — bank {bank_id}: {verdict.reasoning}\n{results}"
    # Only dedup drops repeats; pruning drops what is no help, so it leaves every copy.
    expected = not DEDUP
    RECORDED.append(
        EvalRecord(
            kind="recall",
            question_id=case,
            category="duplicates",
            correct=verdict.meets_criteria == expected and bool(recalled.results),
            hit_trap=False,
            bank_id=bank_id,
            reason=verdict.reasoning,
        )
    )
    if DEDUP:
        assert not verdict.meets_criteria, f"recall deduped, yet returned the fact more than once ({where})"
        assert recalled.results, f"recall dropped every copy ({where})"
    else:
        assert verdict.meets_criteria, f"recall returned no duplicates ({where})"


@dataclass(frozen=True)
class PruneCase:
    id: str
    category: str
    query: str
    #: Each inner tuple is one fact, reworded: one of them must come back, and with dedup on
    #: only one.
    facts: tuple[tuple[str, ...], ...]
    #: Nothing to do with the query: should not come back.
    junk: tuple[str, ...]


_JUNK = (
    "The staging cluster was upgraded to Kubernetes 1.31.",
    "The office coffee machine was repaired on Tuesday.",
    "The design team switched from Figma to Penpot.",
    "Release 0.9.3 was deployed to production on 18 March 2026.",
)

PRUNE_CASES = (
    PruneCase(
        id="reworded-repeats",
        category="duplicates",
        query=QUERY,
        facts=(
            (
                "Marta Rossi's favourite tea is genmaicha.",
                "Marta Rossi's favorite tea is genmaicha.",
                "Genmaicha is the tea Marta Rossi likes best.",
                "Marta Rossi likes genmaicha more than any other tea.",
            ),
            ("Marta Rossi drinks sencha in the afternoon.",),
        ),
        junk=_JUNK,
    ),
    # Alike in wording, distinct in fact: a "which customers" answer needs every one.
    PruneCase(
        id="alike-but-distinct",
        category="distinct",
        query="Which customers signed up for the billing beta?",
        facts=(
            ("Acme Corp signed up for the billing beta.", "Acme Corp joined the billing beta."),
            ("Globex signed up for the billing beta.",),
            ("Initech signed up for the billing beta.",),
            ("Umbrella Health signed up for the billing beta.",),
        ),
        junk=_JUNK[:3] + ("Marta Rossi's favourite tea is genmaicha.",),
    ),
    # One subject, several attributes: none repeats another.
    PruneCase(
        id="same-subject",
        category="distinct",
        query="Tell me about the Lisbon offsite.",
        facts=(
            ("The quarterly offsite is in Lisbon in June 2026.", "The June 2026 offsite will take place in Lisbon."),
            ("Everyone stays at the Memmo Alfama hotel during the Lisbon offsite.",),
            ("The Lisbon offsite budget was approved at 40,000 EUR.",),
        ),
        junk=_JUNK,
    ),
)


# ── Prune: labelled pools, stored as written ─────────────────────────────────
@pytest.mark.parametrize("case", case_params(PRUNE_CASES, {c.id for c in PRUNE_CASES}))
async def test_recall_pools(client: Hindsight, bank_id: str, settled: SettleFn, case: PruneCase) -> None:
    """Every needed fact comes back. With pruning on, also: most junk gone. With dedup on,
    also: one copy of each.

    Without them recall returns junk and repeats by design, so only the first holds.
    """
    # Stored as written, so each label points at the exact text recall returns.
    await prepare_bank(client, bank_id)
    texts = [text for fact in case.facts for text in fact] + list(case.junk)
    await client.aretain_batch(bank_id=bank_id, items=[{"content": text} for text in texts])
    await settled(bank_id)

    recalled = await client.arecall(bank_id, case.query)
    returned = [r.text for r in recalled.results]
    log.info("%s: recall returned %d of %d:\n%s", case.id, len(returned), len(texts), "\n".join(returned))

    def came_back(text: str) -> bool:
        return any(text in r for r in returned)

    lost = [fact[0] for fact in case.facts if not any(came_back(text) for text in fact)]
    repeated = [fact[0] for fact in case.facts if sum(came_back(text) for text in fact) > 1] if DEDUP else []
    junk = [text for text in case.junk if came_back(text)]
    # Most, not all: an unrelated memory is dropped only when the model is sure, since
    # dropping a needed one costs more than showing a stray one.
    too_much_junk = PRUNING and len(junk) > len(case.junk) // 2
    RECORDED.append(
        EvalRecord(
            kind="recall",
            question_id=case.id,
            category=case.category,
            correct=not (lost or repeated or too_much_junk),
            hit_trap=False,
            bank_id=bank_id,
            reason=f"lost {lost}; repeated {repeated}; junk {len(junk)}/{len(case.junk)}",
        )
    )
    where = f"bank {bank_id}, returned:\n" + "\n".join(returned)
    assert not lost, f"{case.id}: every copy of {lost} is missing — {where}"
    assert not repeated, f"{case.id}: more than one copy of {repeated} came back — {where}"
    assert not too_much_junk, f"{case.id}: {len(junk)}/{len(case.junk)} unrelated came back — {where}"


# ── Benchmark: BEIR SciFact ──────────────────────────────────────────────────
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
#: that drops ranking by more than ~5%) without firing on the spread itself.
#: Tighten them to the upper state once a run is reproducible — worth doing,
#: because until then this detects far less than the numbers suggest it could.
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


async def test_recall_benchmark_beir_scifact(request, client, restored_bank):
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
