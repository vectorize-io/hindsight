# Document evolution

Does a living document survive being edited, and is it any good afterwards?

A mental model in delta mode is rewritten by an LLM over and over. Two things can
go wrong and only one of them is visible in a single refresh:

- the **markdown degrades** — a table welded onto one line, a nested list
  flattened, a code fence dropped. No single round looks wrong, and because the
  damage is a fixed point the document never recovers (issue #3361);
- the **content degrades** — facts arrive but do not land, superseded claims are
  still stated, detail is replaced by generalities.

This benchmark measures both across several rounds, and compares two builds so a
change to the pipeline can be shown to be an improvement rather than asserted to
be one.

## How it works

The harness talks HTTP only, so the same code drives a server built from any
revision. That is what makes an A/B possible without putting a feature flag in
the code under test: run it once per build, then compare the two artifacts.

```bash
# a server per build, each with its own database
uv run python -m benchmarks.document_evolution run \
    --api-url http://localhost:8899 --build main   --out results/main.json
uv run python -m benchmarks.document_evolution run \
    --api-url http://localhost:9999 --build branch --out results/branch.json

uv run python -m benchmarks.document_evolution compare results/main.json results/branch.json
```

Every document from every round is stored in the artifact, so the metrics are
recomputed at comparison time. Sharpening a metric therefore costs nothing —
apply it to results you already have instead of spending another few hundred LLM
calls re-measuring.

### Seeding

`--seeding authored` (needs `--db-url`) writes the same starting document into
both builds. The HTTP API has no way to set a document's text — creation only
takes a topic — so this reaches past it deliberately: identical input on both
sides means any divergence afterwards is the pipeline and not the weather.

`--seeding generated` lets each build write its own first version from the same
memories. Less controlled, but it is what a real page does, and it exercises
generation rather than only editing.

## What it measures

**Structural, no LLM.** Damage is counted only in sections that no operation
named. A refresh that rewrites a section it deliberately targeted may
legitimately restructure it; scoring that as corruption would punish the model
for doing its job. A section nobody named is different — nothing there was
supposed to change, so anything missing from it was destroyed by the machinery.

- `collapsed_tables` — a separator cell sharing a line with other cells, the
  detector from #3361
- `table_rows_lost`, `nesting_lost`, `hard_breaks_lost`, `fences_lost`,
  `quotes_lost`, `headings_lost`
- `drifted_sections` — sections whose bytes changed with no operation naming them
- `delta_applied_rounds`, `failed_rounds`, `ops_skipped` — pipeline health
- `median_ms` — wall clock per refresh. Token usage is not reported: the stored
  `reflect_response` does not carry it, and a column that is always zero reads as
  "this is free" rather than "this is not measured here".

**Content, judged.** Claims are checked one at a time against the final document,
so a miss points at a specific fact rather than at a score:

- `recall` — facts that reached the document
- `stale` — superseded claims the document still makes
- `preferred` — blind pairwise comparison of the two builds' final documents.
  Absolute quality scores from an LLM do not calibrate; a forced choice does.
  Each pair is judged twice with the documents swapped, and a pair that changes
  its answer is recorded as a tie rather than counted for whichever side happened
  to go first.

The judge is Gemini by default (`HINDSIGHT_TEST_JUDGE_MODEL`), independent of the
model under test.

## The corpus

`corpus.py` holds three cases, and the third one is the point:

- **api-reference** — tables, a three-level nested list, a fenced block with an
  interior blank line, a hard line break, a blockquote, an ordered list that
  starts at 5.
- **onboarding-playbook** — prose only, no fragile markdown. This is where
  "at least as good as before" is measured: a pipeline change that fixes tables
  but degrades ordinary documents is not an improvement.
- **release-runbook** — a table whose rows do not all carry both outer pipes.
  It renders as a table, people write it and models emit it, and it is the exact
  shape #3361 was reported against. A corpus of only well-formed tables reports
  the pipeline as healthy, because a well-formed table never triggered the bug.

Each round contributes one fact and declares what must be true afterwards
(`asserts`), and what must no longer be stated (`supersedes`) when it replaces an
earlier fact.

Write those claims about the **fact**, not about one phrasing of it. A claim
demanding that a document call an operation "answers questions" failed against
documents describing it as "synthesises stored memories" — the same operation,
in the wording the source fact itself used. A claim a correct document can fail
measures the corpus, not the pipeline.

## Baseline

`baseline_report.json` holds the run behind PR #3622: `main` (400fd3684) against
the branch, three repetitions of the three cases, 45 refresh rounds per build,
identical authored seeds, `gemini-3.1-flash-lite` driving both the pipeline and
the judge.

| | main | branch |
|---|---|---|
| collapsed tables | 3 | 0 |
| table rows lost | 9 | 0 |
| nesting lost | 6 | 0 |
| hard breaks lost | 5 | 0 |
| damaged rounds | 6 | 0 |
| drifted sections | 14 | 0 |
| delta applied | 45/45 | 45/45 |
| failed rounds / skipped ops | 0 / 0 | 0 / 0 |
| median ms per refresh | 12935 | 12940 |

Content came out level: 100% recall and zero stale claims on both sides, and a
pairwise preference of two wins to main, three to the branch, four ties — with
the branch's documents about 10% shorter, which is the direction the judge's
length bias pushes against it.

Read the structural column as the result and the content column as a guard: the
change was made to stop documents being destroyed, and the evidence needed was
that it does that without making them worse.
