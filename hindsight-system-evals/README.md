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

The knowledge-page and reflect suites share one corpus and grade twice per question: **correct** (meets
its criteria — can fail on an incomplete answer) and **trap** (asserts the
specific baited falsehood — the one that matters). The trap is asserted first.
The two retain suites bring their own inputs, because they grade ingestion
rather than retrieval.

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

**`test_03` — retain language.** Real fact extraction (observations and
consolidation off), each input retained `HINDSIGHT_EVAL_RETAIN_REPEATS` times
(default 6) as separate documents, every document judged. A fact in a language
other than the input's is the trap. The regression behind it (#4283): English
coding-agent sessions stored as Spanish, French or Russian facts — about one run
in six on gpt-5.6-luna. Italian and Japanese inputs guard the other direction, a
fix that just forces English. It does not use the corpus.

**`test_04` — retain fidelity.** The other retain suite asks whether a fact is
written in the right language; this one asks whether it is true. Each document
states a date, an owner or a place for one subject, then makes a claim about a
different subject that has none of its own — and the extracted fact must not
carry the borrowed value across. A fabrication here is the worst kind, because
reflect can be asked again while an invented date is written once and read back
as fact forever. Behind it (#4457): `when`/`where`/`who`/`why` are required
non-null strings, so under strict structured output a model has no legal way to
say "not stated" and supplies the nearest plausible value instead.
`HINDSIGHT_API_RETAIN_OPTIONAL_FACT_DIMENSIONS` makes them nullable; this suite
sets nothing and measures the server as configured, so CI watches the default.

It verifies behaviour rather than guarding the regression, and the difference
was measured: Qwen3.6-35B under strict schema passes all three cases with the
nullable fields *and* with the old required-non-null ones. The pressure to
invent a value only bites a model weak enough to feel it — the reported case was
a 9B — so a green run means "extraction is sound on this model", not "the
regression cannot return". The test that fails on that is
`hindsight-api-slim/tests/test_fact_extraction_nullable_dimensions.py`.

**`test_05` — refresh cost.** Not a quality grade: a baseline of what a
knowledge-page refresh spends. Every LLM call the refresh makes is read back from
`/llm-requests` — input, cached and output tokens, and the prompt — for a full
rebuild and for a delta after a small wave of new facts. Behind it: on real
project banks a refresh sent 30–90k-token synthesis prompts, because a tool's
`max_tokens` counts fact text only and the JSON around each fact is 3–4× larger
(#4566). That missed the 30s call deadline (#4568), ended searches before
`recall` ran (#4563), and billed metered keys all night (#4532).

It runs against a **frozen bank**, `fixtures/refresh-cost-bank.zip`: 900 facts
generated from a seed as one fictional product's history, consolidated by a real
model, with the six pages a coding agent seeds. Seeding that live would cost a
consolidation pass per run and start each run from a different bank. Rebuild it
only on purpose — every earlier baseline then measured a different bank:

```bash
uv run python -m hindsight_system_evals.refresh_cost build
# measure a candidate archive before replacing the committed one
HINDSIGHT_EVAL_REFRESH_COST_FIXTURE=/tmp/new.zip uv run pytest evals/test_05_refresh_cost.py
```

`--cost-output DIR` writes `refresh-cost.json` (every call, every prompt), a
summary table, and each refresh's synthesis prompt on its own. The table ends in
USD at the model's list price (`PRICES` in `refresh_cost.py`), including what
explicit Gemini caching costs to create and store — those are not LLM calls and
never reach the trace, so they are estimated from the cached tokens. Each refresh
also records its page size, distinct specifics and citations, a floor under any
cost cut. The model is not frozen, so compare medians over a few runs, not one.

To A/B a server setting, `HINDSIGHT_EVAL_SET_<X>=v` reaches the server as
`HINDSIGHT_API_<X>=v` (the server's environment is otherwise wiped):

```bash
HINDSIGHT_EVAL_SET_REFLECT_PROMPT_CACHE_ENABLED=true uv run pytest evals/test_05_refresh_cost.py
```

**`test_06` — reflect reads the page it was given.** The only suite where the
page layer is live: the others create their pages with `exclude_mental_models`, so
`has_mental_models` is false and reflect is never offered the page tools at all.
This one builds the page, then asks the page's own question and checks reflect
went to the page layer (from the tool trace) and that the answer says what the
page says (judged against the page, not the corpus gold — whether the page itself
converged is `test_01`'s grade).

It exists because that layer changed: `search_mental_models` used to return five
pages whole, and now returns the best hit whole plus a snippet of the others, with
`read_mental_models` for the rest. It is what showed that snippets *alone* let the
model answer from a snippet without ever reading the page, which is why the best
hit still arrives in full.

**`test_07` — source priority.** One bank holding a handbook AND the
conversations about it, which is what a bank someone has actually used looks
like. After extraction both are flat assertions — "a pull request needs two
approvals" and "one approval is enough for small ones" — equally on-topic and
equally retrievable, and the chatter is the more recent of the two, so the
temporal rule actively picks the wrong one. Six questions in the same bank, built
so no blunt rule passes: two where the handbook is right and the chatter is
confidently wrong, one only the handbook answers, one only the chatter answers,
one where a dated decision in a meeting supersedes the handbook, and one where an
approved exception applies alongside the rule. Provenance is recorded in each
document's metadata, which is what the model has to read — it was stripped from
reflect's tool results until this suite landed, so no amount of prompting could
rank the sources. The ways an operator can state the precedence with no new configuration
(the bank's reflect mission, a directive, the question itself) are selectable with
`HINDSIGHT_EVAL_SOURCE_STRATEGY`. It uses `hindsight_system_evals/sources.py`,
not the shared corpus.

**`test_09` — a coding agent's first-prompt reflect, in both reflect modes.** The only suite on
banks shaped like a coding agent's: six sde-bench tasks as the coding-agents plugin ingests
them — each task repo's history, the chat or commit where the decision was made, 140 decoy
developer conversations, and the pages the plugin wrote about the code. Decisions recorded in
chats, in commits (with a later commit that broke them), and amended after the fact. It sends
the plugin's own first-prompt request (`buildReflectQuery`: the developer's goal wrapped in ~2k
characters of rendering rules) in `agent` and `fast` reflect mode on each bank, interleaved,
`HINDSIGHT_EVAL_CODING_REFLECT_REPEATS` times each (default 2). Graded as rates over every run:
fast mode must report the task's recorded decision as often as agent mode, less one; answer
"the bank holds no decision" no more often than agent mode; and be faster (median) with fewer
LLM calls.

Behind it: fast reflect passed every suite above and still injected "the bank holds no
decision" on sde-bench, in 14s against agent mode's 7-10s. The wrapped prompt was the search
query, fresh pages about the code hid the facts, and the decision model's "partly" was a
rounding of a "fully". Then, once fast: the distilled search query asked for "implementation
logic" and surfaced the commit that broke each history task's decision. None of that shows on
short questions over an on-topic corpus. The speed check needs a decision model
(`HINDSIGHT_EVAL_SET_RERANKER_TYPESAFE_API_KEY`); without one only the answers are graded.

The banks are `fixtures/coding-agent-banks/<task>.zip`, with each task's bug report and
recorded policy in `manifest.json` there. Rebuild them on purpose, against a server you keep
up, with sde-bench's own setup (`SDE_HSCODING_PLUGIN_DIR` at a built plugin,
`SDEBENCH_BOLTONS_HOST` at a boltons clone, `OMB_ANSWER_LLM`/`OMB_JUDGE_LLM` set to a provider
you have a key for):

```bash
uv run run-amb --api-url http://localhost:8888 --dataset sdebench --split boltons \
  --memory hindsight-coding -- --mode coding --query-id <task> --skip-answer
# then export bank sde-coding-<task> with export_bank into fixtures/coding-agent-banks/
```

**`test_08` — ranked retrieval on BEIR.** The only suite graded against labels
nobody here wrote, and the only one that makes **no model call**. It records a
*baseline*: what this pipeline scores on a public IR benchmark, and on what.

It takes **BEIR SciFact** — scientific claims as queries, abstracts as documents,
relevance judgements from the dataset — and reports the standard set at
k = **1, 3, 5, 10, 20, 100**:

| | |
|---|---|
| **nDCG@k** | `grade / log2(rank + 1)`, LINEAR gain, which is what `trec_eval` and therefore BEIR use |
| **MAP@k** | divided by the TOTAL relevant (`pytrec_eval`'s `map_cut`), so MAP@1 cannot reach 1.0 on a 2-gold query |
| **Recall@k**, **P@k** | P divides by k, not by what came back |
| **MRR@k** | plus an untruncated MRR, because `MRR@10 == 0 < MRR` is exactly "returned, ranked too low" |
| **Hit@k** | BEIR's Accuracy@k — what a user feels, and the first thing to saturate |
| **R-precision** | P@R for that query's own R; cutoff-free, so a mostly-one-gold slice cannot flatter it |

A unit test cross-checks all of them against `pytrec_eval` on 300 random graded
runs — which is how we found nDCG had shipped with the exponential
`2**grade - 1` gain instead of the linear one. Identical on binary qrels like
SciFact, wrong by ~0.04 on the graded qrels of NFCorpus and TREC-COVID. Install
it with `--extra reference-metrics`; without it that test skips.

### It is a baseline, not a CI gate

This package runs on perf-test's nightly cron, not on pull requests. The floors in
the module exist so a nightly run says something when retrieval moves — not to
block a merge. The measured numbers and the conditions behind them are recorded
above those floors, and that record is the deliverable.

It stays here rather than in `hindsight-system-tests` even though it makes no LLM
call, which is normally that suite's whole criterion. That suite stubs the
embedder AND the reranker with lexical stand-ins, and an nDCG measured over hashed
word overlap is a fact about blake2b — it could never catch a real reranker or
embedder regression. The models this grades have to be the ones that ship, which
is what this package already runs.

### Build once, evaluate many times

```
build (on purpose, ~500 model calls)            eval (every run, ZERO model calls)
------------------------------------            ----------------------------------
retain 500 abstracts, real extraction           import the archive
export_bank -> fixtures/beir-scifact-bank.zip   recall each query, score vs. qrels
```

Re-ingesting per run would pay for extraction to measure something extraction is
not under test for — and worse, would move the corpus underneath every
comparison: a model that extracted different claims this afternoon changes the
score with nothing in retrieval having changed. The archive makes the memories a
constant, which is the only way an A/B on a retrieval setting means anything.

| baked into the zip at build | free to A/B at eval time |
|---|---|
| extraction model and mode | embedder, reranker |
| which claims each abstract became | `budget`, `max_tokens` |
| observations / consolidation off | fusion and scoring settings |

The embedder is on the right because **`export_bank` deliberately leaves
embeddings out** — the importer regenerates them with its own model. That is the
feature that makes the embedder itself a variable here.

`fixtures/beir-scifact-bank.json` sits beside the archive and records which model
wrote the memories, with the document and fact counts. Unknowable from the zip,
and the single most important thing about it.

Because nothing calls a model, this is the one suite that needs no credentials:

```bash
HINDSIGHT_EVAL_LLM_PROVIDER=none HINDSIGHT_EVAL_LLM_MODEL=none \
  uv run pytest tests evals/test_08_retrieval_metrics.py        # ~2.5 minutes

# rebuild the archive, against a server configured with the extraction model you want
uv run python -m hindsight_system_evals.retrieval build --api-url http://localhost:8888
```

### Two things it asserts beyond the floors

- **`mean documents returned` must exceed 100.** `recall` has no `k` — it budgets
  by `max_tokens` — so every cutoff above the number of documents a response
  reaches measures the *budget*, not the ranking, and would read as a flat,
  flattering line across @20 and @100. The exception is a pruning reranker, which
  returns a handful on purpose; that arm gets a Recall@10 floor instead, since the
  only thing making aggressive pruning safe is keeping the gold document.
- **Most queries must reach a gold document.** The scoring rests on `document_id`
  surviving export and import; if the archive renumbered documents, every id would
  miss the qrels and the run would read as a total retrieval collapse rather than
  a broken fixture.

### The committed slice, and what its number is for

50 queries over 500 documents (51 gold), so a gold abstract beats ~475
distractors. It was 200 first and that was a mistake worth recording: nDCG@10 read
0.933, above SciFact's published ~0.70 over the full 5,183, because a gold hit
only had ~190 distractors to beat. A benchmark that flatters is a floor that never
trips. Documents cost one extraction call each and queries cost nothing but a
`recall`, which is why the query count is generous and the document count is where
the budget goes.

A number here is **not** comparable to a published SciFact nDCG@10: 500 is not
5,183, and this ranks extracted claims rather than the abstracts themselves.

Re-cutting the slice invalidates the bank archive, which has to be rebuilt from it:

```bash
uv run python -m hindsight_system_evals.beir build          # re-downloads SciFact
uv run python -m hindsight_system_evals.retrieval build ... # then rebuild the bank
```

`HINDSIGHT_EVAL_BEIR_QUERIES=5` trims the query set for a smoke run without
touching the corpus, so the smoke run faces the same distractors.
`HINDSIGHT_EVAL_BEIR_DIR` points at any unpacked BEIR directory (`corpus.jsonl`,
`queries.jsonl`, `qrels/test.tsv`); the graded nDCG already handles NFCorpus's and
TREC-COVID's 0/1/2 qrels.

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

## Where it points

By default every eval starts its own `hindsight-api` on its own pg0 — that is
what CI measures, and what makes a run mean the same thing on every machine.
Pass `--api-url` (or set `HINDSIGHT_EVAL_API_URL`, with `HINDSIGHT_EVAL_API_KEY`)
to run against a server that is already up instead: cloud dev, a colleague's box,
a docker compose. The two are independent, so the retain evals can run against a
deployment while a benchmark runs locally, or the reverse:

```bash
uv run pytest evals --api-url https://api.dev.example    # evals against cloud dev
uv run run-amb --dataset locomo --split locomo10         # LoComo against a local server
```

Two things change on a remote target, and neither is worked around:

- **the model is not asserted, it is read back.** `HINDSIGHT_EVAL_LLM_*` configures
  a server *we* start; a remote server's model is its own, so the report records
  what the server says it runs rather than what we hoped;
- **banks are deleted at the end.** On pg0 they are free and left behind on
  purpose, for the control plane. In a shared tenant they would accumulate every
  run. `--keep-banks` opts out, for inspecting a failure.

## Benchmarks (AMB)

LoComo, LongMemEval, BEAM, PersonaMem and the coding-agent suite (sde-bench) live
in [AMB](https://github.com/vectorize-io/agent-memory-benchmark), which owns their
datasets, prompts, judge and scoring, and publishes to
[agentmemorybenchmark.ai](https://agentmemorybenchmark.ai). None of that is
duplicated here — a second copy of LoComo is how two copies drift until neither
number means anything. This package only points AMB's `hindsight-http` provider
at a target:

```bash
# A split is the dataset slice; one conversation is a `--unit` within it.
uv run run-amb --dataset locomo --split locomo10 -- --unit conv-26 --query-limit 20
uv run run-amb --dataset longmemeval --split s -- --category single-session-user --query-limit 20
uv run run-amb --dataset beam --split 100k --api-url https://api.dev.example

cd "$(uv run python -c 'import os;print(os.path.expanduser("~/.cache/hindsight/amb"))')" && uv run amb splits --dataset locomo
```

AMB tracks the ref in `AMB_REF` — `main`, so a fix merged in AMB reaches the next
run without a PR here (override with `--amb-ref`, or `AMB_REF=`, to replay an old
number). Every run prints the AMB commit it resolved to: benchmark drift and engine
drift look identical in the numbers, so that commit is what makes a movement
attributable.
It needs `GEMINI_API_KEY` — AMB judges and answers with Gemini through the API
key, not through our VertexAI service account. AMB pins its own interpreter
(`.python-version`, 3.12) and uv honours it; `--python` / `AMB_PYTHON` override
that for a one-off. A pin matters there because AMB's `requires-python` is only
`>=3.11`: unpinned, uv takes the newest interpreter on the machine, and on 3.14
the install dies before the benchmark starts — `onnxruntime`, via cognee,
publishes no wheel for it. The sde-bench coding suite needs
more still (Docker, a boltons host clone, an agent CLI with its own key); it runs
through the same command with `--dataset sdebench`, but it is a campaign, not a
scheduled job.

AMB is now the only copy. The in-repo LoComo and LongMemEval runners are gone —
`hindsight-dev/benchmarks/locomo/`, `longmemeval/`, `run-locomo.sh`,
`run-longmemeval.sh`, and the benchmark visualizer, which served nothing else.
`publish-locomo-results.sh` stays — it now reads AMB's report. The `locomo` job in `perf-test.yml` became the `amb`
job, a matrix over both datasets, and it still publishes the LoComo run to the
[continuous performance monitor](https://vectorize-io.github.io/hindsight-continuous-performance-monitor/)
— `publish-locomo-results.sh` normalises AMB's report onto the existing series
(a percentage, not AMB's 0-1 fraction) so the 86-run chart keeps its shape. `perf/`, `micro/`, `obs/`,
`document_evolution/` and `multimodal_retain/` measure things AMB does not, and
stay where they are.

## Running

```bash
# the model under test (api key, or vertexai with a service account)
export HINDSIGHT_EVAL_LLM_PROVIDER=gemini
export HINDSIGHT_EVAL_LLM_MODEL=gemini-3.8-flash
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

The server runs on a **fresh** pg0 instance per run (`hindsight-system-evals-<id>`),
dropped when the session ends. Its own, so a run does not compete for connections
with a developer's server or with the system tests — and a new one, because a
reused database still holds the previous run's banks *and their queued refresh
and consolidation tasks*. The worker claims those within seconds of starting and
bills them to your provider key: one run spent 5.9M tokens on four banks it never
created, against 0.8M of its own.

Set `HINDSIGHT_EVAL_PG0_INSTANCE` to pick the instance yourself (the coding-agents
runner does, one per run); one you named is yours, and is left in place.

A crash can leave an instance behind. `pg0 list` shows them, `pg0 drop <name>`
removes one.

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
