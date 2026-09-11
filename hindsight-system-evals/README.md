# Hindsight system evals

Blackbox **quality** evals over a real `hindsight-api` process and a **real
model**, driven only through the published Python client. No engine imports, no
SQL, no internals — the same rule `hindsight-system-tests` follows.

## Why this is a separate package

`hindsight-system-tests` points every LLM call at a stub. That is what makes it
deterministic, secret-free, and runnable on fork PRs.

An eval cannot do that. Stub the model and you score the stub. So this package
inherits the blackbox *shape* of the system tests and none of their determinism
mechanism, and everything downstream follows from that:

| | system-tests | system-evals |
|---|---|---|
| model | stub | real provider, required |
| secrets | none, runs on forks | needed → **not run on PRs** |
| assertion | exact equality | judged, and only meaningful as a rate |
| a failure means | a broken mechanism | a quality regression, or sampling noise |

That last row is the one to keep in mind. The same request can produce a
destructive edit once and a correct one the next time, so a single red run is a
signal to re-run, not proof of a regression.

## What it evaluates

Both suites share one corpus and grade twice per question: **correct** (meets
its criteria — can fail on an incomplete answer) and **trap** (asserts the
specific baited falsehood — the one that matters). The trap is asserted first.

**`test_01` — knowledge-page convergence.** A page is created with a source
query and then *accumulates*: data arrives in waves and each refresh edits what
is already stored. That is where a wrong answer stops being a wrong answer and
becomes a wrong *memory*. Two regressions found this way, neither visible to a
one-shot reflect:

- a page saying *"Release 0.9.3 was deployed to production on 18 March 2026"*
  came back one wave later saying *"No release was deployed"* — the fact was in
  an earlier wave, outside the delta window;
- a page counting 3 customers, handed 4 more, reported **4**.

Both had one cause: the delta step treated the reflect synthesis as
authoritative, when that synthesis is written from the new batch alone.

**`test_02` — reflect answers.** The whole corpus in one bank, one reflect call,
the answer judged. The regression behind it: asked for the 2024 headcount in a
bank covering only 2025-26, reflect extrapolated a number and called it
"reliably deduced". A failure reports whether every gold fact reached the model
(from the tool trace), because a retrieval miss and a reasoning miss need
opposite fixes.

## The corpus

`hindsight_system_evals/corpus.py` generates facts and their gold labels
together, so a label cannot drift from the text it points at. Two properties it
enforces, both learned by getting them wrong:

- **Every subject is internally consistent.** An earlier version had one release
  "deployed to production" on three different dates. That is not a hard question,
  it is a contradiction, and a model superseding it was following its rules.
- **Waves never split a subject.** All facts about one release travel together.
  Split across waves, the later batch reads as a correction of the earlier one.

`_assert_subjects_are_unique` fails the build if two rows in a cluster ever claim
to describe the same thing again.

## Where it runs

**Not on PRs.** It needs provider secrets, and a single red run is as likely to
be sampling noise as a regression. It runs in the `system-evals` job of
`.github/workflows/perf-test.yml` — the daily schedule, alongside LoComo and
obs-dedup — and publishes to the
[continuous performance monitor](https://vectorize-io.github.io/hindsight-continuous-performance-monitor/system-evals.html)
as a quality metric tracked over time: correct rate and trap count, overall and
per suite (`by_kind`). Traps must stay at 0.

A failing run is still published: for a quality metric the red run is the data
point. `scripts/benchmarks/publish-system-evals-results.sh` does the push.

## Two modes

```bash
# minimum acceptance: the cases that have actually regressed
uv run pytest evals

# full: every category — supersession, entity confusion, scoped truth,
# dense absence, numeric precision — plus the page-collapse check
uv run pytest evals --full

# either, writing the JSON the dashboard publishes
uv run pytest evals --full --output system-evals-results.json
```

The workflow runs `full` by default; `system_evals_mode: minimum` on a manual
dispatch runs the small set.

## Why seeding costs no model calls

Measured on the first blackbox run, per page: 789s of LLM time on fact
extraction and 199s on consolidation, against 21s for the reflect and delta
calls actually under test. So each bank is configured — through the public config
endpoint, still blackbox — with `retain_extraction_mode=chunks` (store each item
as written) and consolidation/observations off, and each page refresh is
triggered explicitly. `chunks` also removes a confound: extraction may paraphrase
a fact, while the gold labels point at the exact authored text.

## Running

```bash
# the model under test (api key, or vertexai with a service account)
export HINDSIGHT_EVAL_LLM_PROVIDER=gemini
export HINDSIGHT_EVAL_LLM_MODEL=gemini-3.7-flash
export HINDSIGHT_EVAL_LLM_API_KEY=...

# The judge is configured separately ON PURPOSE — a model grading its own output
# agrees with itself. The suite warns when these resolve to the same model.
export HINDSIGHT_EVAL_JUDGE_MODEL=gemini-2.5-flash
export HINDSIGHT_EVAL_JUDGE_API_KEY=...        # or HINDSIGHT_EVAL_JUDGE_PROVIDER=vertexai

cd hindsight-system-evals && uv run pytest evals
```

VertexAI works for both: set `HINDSIGHT_EVAL_LLM_PROVIDER=vertexai` and
`HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY` / `_PROJECT_ID` — the judge
falls back to the same service account when no judge key is set. That is how the
perf workflow runs it. `HINDSIGHT_EVAL_REFLECT_BUDGET` (default `low`) sets the
reflect budget for `test_02`.

If your shell exports `PYTEST_ADDOPTS` with `-n` (the repo `.env` does), unset
it: xdist is not installed here, and parallel evals against one server would
compete for it anyway.

The server runs on its own pg0 instance (`hindsight-system-evals`), so a run does
not compete for connections with a developer's server or with the system tests.

## Debugging a failure

Each assertion prints the bank id, the page id (or the reflect queries and
whether the gold evidence arrived), and what the page or answer actually reads.
The bank is left in place, so it can be opened in the control plane.

For a knowledge page that goes wrong, two tools, both through the public API:

```bash
# 1. Where does it go wrong? Builds wave 1, then dry-runs the second refresh N
#    times: raw synthesis vs the document after the delta operations. --dump
#    writes each traced prompt verbatim.
uv run python -m hindsight_system_evals.debug.diagnose_page \
    --question hq-count-billing --repeats 3 --dump captures/

# 2. Is it the prompt? Replays one captured delta-ops request N times per
#    system-prompt variant and scores the resulting document; --interrogate
#    asks the model which lines misled it.
uv run python -m hindsight_system_evals.debug.replay_delta_ops \
    --input captures/hq-count-billing-mental_model_delta_ops-<n>.json \
    --must-keep "Denning|Barrow|Fairbank" --variants variants.json
```

A variant that wins the replay is a lead; it still has to pass the eval.
