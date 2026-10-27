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

One test file per operation, `evals/<operation>/test_<operation>.py`: `retain`,
`recall`, `consolidation`, `reflect`, `refresh-pages` (and `misc/` for the
coding-agents metric). Run one operation with `uv run pytest evals/recall`.

**Every case has a category**, and its test id is `<category>/<case id>`
(`test_retain[temporal/no-event-date]`). That is the syntax:

```bash
uv run pytest evals/retain --full -k "temporal/"           # one category
uv run pytest evals --full --output gemma.json             # every case, one model
uv run eval-table gemma.json sol.json luna.json            # operation x category x model
```

`eval-table` prints each run's correct/total per operation and category, so a model
that is strong on one group and weak on another shows it; the run JSON carries the
same counts as `by_category`. Each category holds at least two cases. A test named
`*_benchmark_*` is not a case: it measures a rate or a cost over a frozen bank and
stays out of the category counts.

| operation | category | cases | what it checks |
|---|---|---|---|
| retain | selectivity | 3 | chatter stores nothing; plans survive among small talk; a fact said three times is stored once |
| retain | completeness | 2 | exact figures, never rounded; sensory and emotional detail kept |
| retain | attribution | 4 | "my roommate" becomes Tamsin; the user's rules are `world`, the agent's actions `experience`; a speaker named in context or metadata is credited |
| retain | temporal | 4 | relative dates resolved; a year spans the year; open-ended and undated facts; no event date, no invented date |
| retain | language | 7 | German, Chinese, Spanish keep their language and script; English, Italian, Japanese inputs retained 6× never switch |
| retain | fidelity | 5 | hedges stay hedged, past states stay past, a correction wins, the prompt's examples never leak, instructions inside content are not obeyed |
| retain | no_invention | 3 | a date, owner or place stated for a neighbouring subject is not borrowed |
| retain | mission | 2 | the retain mission filters what is kept and outranks the generic selectivity rules |
| retain | mode | 4 | custom instructions, verbose, verbatim, a strategy chosen per item |
| retain | entities | 2 | named entities stored; one person across documents is one entity |
| retain | entity_labels | 3 | value and multi-value labels, free-text and map labels, labels-only mode |
| retain | inputs | 3 | document metadata read, JSON conversations, a fact in the last chunk of a long document |
| recall | semantic | 3 | a paraphrase; a query sharing no word with its memory; intent over a shared keyword |
| recall | keyword | 3 | an error code, an order id, a rare surname among near-identical ones |
| recall | temporal | 4 | "last month", "yesterday", a year-only event, a named week — against `query_timestamp` |
| recall | currency | 2 | the newer of two conflicting memories (a job, a decision) ranks first |
| recall | multi_hop | 3 | the answer one entity or cause away: an employer's city, the cause of a cause, my sister's dog |
| recall | filters | 3 | `types=[experience]`, `types=[world]` on a user/assistant exchange; a tag scope |
| recall | language | 2 | Italian and Chinese queries (with 上周) over memories in that language |
| recall | precision | 2 | the right budget among near-identical ones; a negation kept |
| recall | budget | 2 | a small `max_tokens` returns fewer facts but never none; `max_tokens=0` returns chunks only |
| recall | observations | 2 | `types=[observation]`; `prefer_observations` drops the facts an observation was built from |
| recall | duplicates | 2 | one sentence in four documents comes back four times (once, with pruning); reworded repeats collapse under pruning |
| recall | distinct | 2 | facts that only look alike (four customers, three details of one offsite) all come back |
| consolidation | merge | 2 | restatements update one observation; the same facts under two tags merge in a shared scope |
| consolidation | update | 4 | a new employer, retired tools, a correction, a member leaving a list: the old belief stops reading as current |
| consolidation | recency | 2 | recency is when a fact was stated, not when it was retained |
| consolidation | counts | 2 | the horse farm; no arithmetic on counts |
| consolidation | entities | 2 | a vague reference resolved by a later fact; lookalike people and numbers stay apart |
| consolidation | language | 2 | Chinese facts give Chinese observations; a mixed update is rewritten, not patched |
| consolidation | mission | 3 | without a mission chatter is dropped; a mission can keep ephemera or forbid names |
| consolidation | scale | 3 | 30 near-identical services, 20 lookalike-named people, the services under a mission |
| reflect | precision | 3 | the answer differs from its near-misses by one attribute: a version, a service name, a number |
| reflect | update | 2 | what is true now, when the bank also holds what was true before |
| reflect | aggregation | 2 | a count across many facts; a rule together with its approved exception |
| reflect | no_overreach | 2 | a year the bank does not cover; a rule true in one region only |
| reflect | source_authority | 4 | the handbook against confident chatter; questions only one source answers |
| reflect | pages | 2 | the bank holds a page about the question, and the answer comes from it |
| refresh_pages | precision | 3 | a page built across waves keeps the asked attribute apart from its near-misses |
| refresh_pages | update | 4 | a later wave changes the answer: a newer owner, a departed on-call, a renamed service, a corrected figure |
| refresh_pages | aggregation | 3 | a count grows across waves; a list gains and loses members; a new risk joins the list |
| refresh_pages | no_overreach | 2 | a year the bank does not cover; a rule true in one region only |
| refresh_pages | stability | 2 | a wave with nothing for the page leaves it intact; no page collapses across waves |
| refresh_pages | consistency | 2 | a changed value changes everywhere the page restates it (English, Chinese) |

The knowledge-page and reflect evals share one corpus and grade twice per question: **correct** (meets
its criteria — can fail on an incomplete answer) and **trap** (asserts the
specific baited falsehood — the one that matters). The trap is asserted first.
The retain evals bring their own inputs, because they grade ingestion rather than
retrieval.

### Retain — `evals/retain/test_retain.py`

**Capabilities** (`test_retain`): 35 cases in `hindsight_system_evals/retain_capabilities.py`,
one per thing retain is supposed to do, each naming the prompt rule or issue behind it. A case is judged (a trap and the facts it must record, on the stored facts rendered with
their type, dates and entities) and, where the API reports the answer exactly, checked
deterministically (entities, labels, tags, verbatim text, fact counts). Nine run in minimum
acceptance. It was built against dev Gemma
(`gemma-4-26b-a4b`), then cross-checked on Gemini 2.5 and 3.1 flash-lite and two Codex
models. Every case passes on at least one model, and what fails is the model, not the case
(full runs, with the `where` field stored — on main the parser drops it, so `entities-named`
fails on nearly every model until #6129 lands):

| model | passed | fails |
|---|---|---|
| `gpt-6.1-sol` (Codex) | 35/35 | — |
| `gpt-6-luna` (Codex) | 34/35 | judged "my cousin's wedding. Lovely day." too thin to store, twice; the case now gives it a detail, and passes 3/3 |
| `gpt-5.6-luna` (Codex) | 33/35 | a date on a timeless fact |
| `gemini-3.1-flash-lite` | 32/35 | obeys "the CEO resigned" written in the content; invents a date when none is set; drops "Marco was nervous" |
| `gemma-4-26b-a4b` | 27–33/35 | Spanish stored as English; drops "Marco was nervous"; stores no facts (below). Latest run of all 42 retain cases: 37/42 |
| `gemini-2.5-flash-lite` | 26–27/35 | obeys instructions in content; ignores custom instructions; German and Spanish facts in English |

Gemma now and then stores a whole input as **no facts**: the reply ran to the 4,096-token
limit and retain, unable to split a 224-character chunk further, dropped it and reported
success (#6129) — or the model returned `{"facts": []}` for plain content. A case that stores
nothing says which, from the public LLM trace, instead of failing every criterion on
"(no facts)".

**Language, as a rate** (`test_retain_language`). Real fact extraction (observations and
consolidation off), each input retained `HINDSIGHT_EVAL_RETAIN_REPEATS` times
(default 6) as separate documents, every document judged. A fact in a language
other than the input's is the trap. The regression behind it (#4283): English
coding-agent sessions stored as Spanish, French or Russian facts — about one run
in six on gpt-5.6-luna. Italian and Japanese inputs guard the other direction, a
fix that just forces English. It does not use the corpus.

**No invention** (`test_retain_no_invention`). Each document
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

### Recall — `evals/recall/test_recall.py`

**Capabilities** (`test_recall`): 26 cases in `hindsight_system_evals/recall_capabilities.py`,
each naming its history. Recall itself calls no LLM by default — dates in a query are read by
a rule-based parser (`engine/query_analyzer.py`) and ranking is the embedder plus a
cross-encoder — so every case retains through the real extraction, and a model comparison
measures the memories that model's retain wrote, as recall finds them. Graded
deterministically (a word in the top results, one result above another, a word never
returned, the result types, token budgets and chunks) and by the judge only for a negation.
The tag modes, `min_scores` floors and budget levels are pinned by the stub-model system
tests and are not repeated here.

First full runs (judge `gemini-2.5-flash`):

| category | gemma-4-26b-a4b | gemini-3.1-flash-lite | gpt-6.1-sol | gpt-6-luna |
|---|---|---|---|---|
| semantic | 3/3 | 3/3 | 3/3 | 2/3 |
| keyword | 3/3 | 2/3 | 3/3 | 2/3 |
| temporal | 1/4 | 1/4 | 2/4 | 2/4 |
| currency | 0/2 | 1/2 | 1/2 | 1/2 |
| every other category | all | all | all | all |

What fails is mostly recall, not the model. **A date in the query barely moves the order**:
"where did I travel last month?" finds the Lisbon trip (the temporal arm matches its
`mentioned_at`), but the final order is the cross-encoder's, and temporal proximity and
recency are each a ±10% multiplier on it (`reranking.py`), so "Berlin for a wedding" outranks
it, and "a dentist appointment" four days ago outranks yesterday's concert. The same holds
for "where does the user work now?": the older "works at Globex" outranks "just started at
Initech" on every model. These cases stay red until ranking weighs dates more (#6132). The model
failures are real too: `gpt-6-luna` stored English input as Russian and Spanish facts (the
#4283 drift), and Gemini 3.1 flash-lite dropped "Rotterdam" from the fact it was asked about.

**Duplicates** (`test_recall_duplicates`, `test_recall_pools`). With observations and
consolidation off, one sentence retained as four documents becomes four facts, and recall
hands back every copy: observations merge facts, recall does not. With
`HINDSIGHT_EVAL_SET_RECALL_DECISION_MODEL_PRUNE=true` (the flag from #6131, and a decision model) the same story
expects one copy. Three pools of stored-as-written facts check that recall returns every
needed fact; with pruning on they also stress it: reworded repeats must collapse to one,
while alike-but-distinct facts (four customers joining one beta, three details of one
offsite) must all survive, and most unrelated facts must go.

**Benchmark: ranked retrieval on BEIR** (`test_recall_benchmark_beir_scifact`). The only suite graded against labels
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

#### It is a baseline, not a CI gate

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

#### Build once, evaluate many times

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
  uv run pytest tests evals/recall -k benchmark_beir        # ~2.5 minutes

# rebuild the archive, against a server configured with the extraction model you want
uv run python -m hindsight_system_evals.retrieval build --api-url http://localhost:8888
```

#### Two things it asserts beyond the floors

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

#### The committed slice, and what its number is for

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

### Consolidation — `evals/consolidation/test_consolidation.py`

Facts are retained in waves (as written, so only
consolidation is graded) and the real consolidation runs after each one; the
observations left behind are graded. Every scenario reproduces a failure that
shipped or a rule the consolidation prompt promises:

- restatements update one observation, never grow siblings (#1566, #3249);
- a newer fact supersedes the old belief everywhere it appears (#1151, #3248, #3475);
- recency is when a fact was *stated*, not when it was retained, even an hour apart on the same day (#2550, #5938);
- a correction retires the false belief (#3509);
- one facet per observation, matched by entity, no arithmetic (#814, the horse farm);
- a member leaving a list updates the list too; a vague reference gets resolved;
- different people and different numbers stay apart (#5497);
- observations keep their facts' language, and a mixed update is rewritten, not patched (#3166/#3181);
- the mission wins: it can keep chatter (#525) or forbid names (#4619); without one, chatter is dropped;
- the same facts under different tags in a `shared` scope merge (#2202, #3564);
- in a bank of 30 near-identical services, every later fact is shown the
  observation it belongs to (read from the LLM trace, so a recall miss and a
  model miss are told apart), changes land as current, and the confusable
  neighbours survive (#1907, #2000) — `--full` only, as it seeds ~130 facts; a second
  bank of 20 people whose names come in lookalike pairs (Maria / Mario) checks the same on
  personal memory.

Each wave also runs deterministic checks: evidence exists and matches
`proof_count` (#4955), dates come from the evidence (#501, #3477), no prompt
metadata leaks into the text (#2544), every durable fact is cited (#5054), and a
wave of pure restatements adds no observation. `HINDSIGHT_EVAL_CONSOLIDATION_DUMP=DIR`
writes every consolidation call of each scenario (what the model was shown, what
it answered) to `DIR/<scenario>.json` — the first thing to read on a failure.

Each scenario also reports what consolidating it cost, read back from the server's
LLM trace: calls, input / cached / output tokens, LLM seconds, and wall time from
retain to settled. `HINDSIGHT_EVAL_CONSOLIDATION_COST=FILE` appends one JSON line per
scenario with that and whether it passed, so two configurations can be compared on
cost *and* outcome — a cheaper setting only counts at the same quality:

```bash
HINDSIGHT_EVAL_CONSOLIDATION_COST=/tmp/base.jsonl uv run pytest evals/consolidation --full
# consolidation triage: a decision model (TypeSafe Jev) for consolidation, on for every bank
HINDSIGHT_EVAL_SET_CONSOLIDATION_DECISION_MODEL_PROVIDER=typesafe HINDSIGHT_EVAL_SET_CONSOLIDATION_DECISION_MODEL_API_KEY=... \
  HINDSIGHT_EVAL_CONSOLIDATION_COST=/tmp/triage.jsonl uv run pytest evals/consolidation --full
```

Triage requests are traced like LLM calls (scope `decision_consolidation_triage`) and
reported apart from them, as `triage N calls, M in`. Speaker attribution is decided at
retain and is out of scope; failure handling and scope mechanics are
deterministic and live in `hindsight-system-tests`.

#### Consolidation capability inventory

Everything consolidation promises, from the docs (`docs/developer/observations`), the
prompt (`engine/consolidation/prompts.py`), the code (`engine/consolidation/`) and the
issue/PR history — and what grades it. "eval" = a `test_consolidation` scenario (real model);
"system" = `hindsight-system-tests` (stub model); "unit" = `hindsight-api-slim/tests`.

**What the model decides — graded by `test_consolidation` on gemini-3.1-flash-lite**
- Restatements update one observation; no near-duplicate siblings (rule 1; #1566, #3249, #1759) — eval `restatements-merge`, `shared-scope-merges-across-tags`
- One facet per observation; a count is its own facet (rule 2; #814) — eval `horse-farm`
- Match by entity, not topic: "sold Buttercup" never touches Daisy (rule 3; #814) — eval `horse-farm`
- State changes are UPDATEs to the current state, keeping what was replaced (rule 4; #3248) — eval `retired-conventions`, `employer-supersedes`
- A change cascades to every observation stating the old value, list members included (rule 5; #1151) — eval `employer-supersedes`, `cascade-to-group`, `noisy-service-catalogue`
- Vague references are resolved by later facts (rule 6) — eval `resolve-reference`
- Significant events are history, never deleted (rule 7; #814) — eval `horse-farm`
- No arithmetic or deduction on counts (rule 8) — eval `no-arithmetic`, `horse-farm`
- Different people, entities and numbers stay apart, lookalike names included (rule 9; #5497, #1725) — eval `distinct-entities-stay-apart`, `noisy-service-catalogue`
- Recency is `mentioned_at`, not arrival order; older news never overwrites (rule 10; #2550) — eval `recency-is-when-stated`, `recency-same-day-amendment-arrives-first` (#5938)
- Chatter is skipped under the default mission (rule 11) — eval `default-drops-chatter`
- An update keeps everything the observation said (rule 12) — eval `noisy-service-catalogue` (neighbours, coverage)
- Every durable fact lands in an observation (#5054) — eval coverage check on every wave
- A correction retires a false belief (#3509) — eval `correction-retires-false-belief`
- Retired settings and tools stop reading as current (#3248, #3475) — eval `retired-conventions`
- The mission wins over the rules: it can keep ephemera (#525) or forbid names (#4619) — eval `mission-keeps-ephemeral`, `mission-forbids-names`
- Observations keep their facts' language; a mixed update is rewritten (#3166, #3181, #4016) — eval `language-chinese`, `language-mixed-update`; unit `test_consolidation_output_language.py`
- A configured output language overrides (#3776) — unit `test_consolidation_output_language.py`
- Observation text carries no prompt metadata, JSON or asides (#2544) — eval leak check on every wave
- In a large bank of near-identical entities the recall still surfaces the right observation (#1907) — eval `noisy-service-catalogue` (30 services) and `noisy-personal-contacts` (20 people in lookalike name pairs), `twin` check from the LLM trace

**What the server guarantees — deterministic**
- Source facts survive; observations cite evidence that resolves — system `test_20`; eval evidence check
- `proof_count` equals the evidence count, including multi-fact creates (#4955) — eval evidence check
- Dates come from the evidence: earliest start, latest mention, none invented (#501, #3477) — eval date check; unit `test_consolidation_temporal_merge.py`
- One response is applied atomically with its `consolidated_at` stamps (#3876) — unit `test_consolidation_batch_atomicity.py`
- Failed calls bisect, then mark failed; recovery is explicit (#601) — system `test_21`
- Semantic near-twin merge on create and update, number- and negation-aware (#1977, #2000) — unit `test_consolidation_dedup.py`
- Exact-text duplicate creates are dropped (#1907) — unit `test_consolidation_dedup.py`
- Deletes must name an `observation_id`; a malformed one no longer voids the response (#4152, #4183) — unit `test_consolidation_delete_schema.py`, `test_consolidation_delete_prompt_llm.py`
- A merge delete (`merged_into`) moves its evidence to the survivor, or re-queues it; a plain delete retires it — system `test_22_consolidation_safety_nets`
- Facts a response ignores are re-asked (together, then alone) — system `test_22_consolidation_safety_nets`
- Miscopied observation ids are repaired; an update aimed at a fact id becomes a create — unit `test_consolidation_id_repair.py`; system `test_22_consolidation_safety_nets`
- Observation scopes (`combined`, `per_tag`, `all_combinations`, `shared`, explicit) and strategies (#2202, #4619) — system `test_24`, `test_29`
- Per-scope observation caps (#729, #2163) — system `test_29`; unit `test_observation_scope_limit_resolution.py`
- Observation history snapshots — unit `test_observation_history.py`
- Deleting or replacing a document removes and rebuilds its observations (#429, #3294) — system `test_25`; unit `test_delta_retain_orphan_observations.py`
- Trigger, dedup and serialisation of consolidation jobs (#4368) — system `test_26`, `test_28`
- Observations in recall, `prefer_observations`, source facts (#2311) — system `test_27`
- Consolidation triage (a decision model, e.g. TypeSafe Jev): restatements, new facets and chatter settled without the LLM — unit `test_consolidation_triage.py`; system `test_22_consolidation_triage`; eval suite with triage on; under a bank's own mission it records only facts the mission wants as worded — eval `noisy-service-catalogue-mission`

**Out of scope here, deliberately** — speaker attribution is decided at retain (#4572, #1680)
and belongs with the retain suites; operational behaviour (timeouts #3301/#4715, quota
handling #4160, wall-clock ceilings) is not a quality property.

### Reflect — `evals/reflect/test_reflect.py`

**Corpus questions** (`test_reflect`). The whole corpus in one bank, one reflect call,
the answer judged. The regression behind it: asked for the 2024 headcount in a
bank covering only 2025-26, reflect extrapolated a number and called it
"reliably deduced". A failure reports whether every gold fact reached the model
(from the tool trace), because a retrieval miss and a reasoning miss need
opposite fixes.

**Source authority** (`test_reflect_sources`). One bank holding a handbook AND the
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

**Pages** (`test_reflect_from_page`). The only suite where the
page layer is live: the others create their pages with `exclude_mental_models`, so
`has_mental_models` is false and reflect is never offered the page tools at all.
This one builds the page, then asks the page's own question and checks reflect
went to the page layer (from the tool trace) and that the answer says what the
page says (judged against the page, not the corpus gold — whether the page itself
converged is the refresh-pages eval's grade).

It exists because that layer changed: `search_mental_models` used to return five
pages whole with no bound, and now returns whole pages in rank order up to a token
budget plus a snippet of the others, with `read_mental_models` for the rest. It is
what showed that snippets *alone* let the model answer from a snippet without ever
reading the page, which is why the best hit always arrives in full.

**Benchmark: a coding agent's first-prompt reflect, with and without the decision model**
(`test_reflect_benchmark_coding_agent_decision_model`). The only suite on
banks shaped like a coding agent's: six sde-bench tasks as the coding-agents plugin ingests
them — each task repo's history, the chat or commit where the decision was made, 140 decoy
developer conversations, and the pages the plugin wrote about the code. Decisions recorded in
chats, in commits (with a later commit that broke them), and amended after the fact. It sends
the plugin's own first-prompt request (`buildReflectQuery`: the developer's goal wrapped in ~2k
characters of rendering rules) with the decision model on and off (`reflect_use_decision_model`) on each bank, interleaved,
`HINDSIGHT_EVAL_CODING_REFLECT_REPEATS` times each (default 2). Every bank gets a reflect mission
that ranks its sources by their `source:` tags (a stated decision outranks a later cleanup commit
and the pages about the code), so the decision model is also held to following a mission it
never reads itself. Graded as rates over every run:
with it, reflect must report the task's recorded decision as often as agent turns, less one; answer
"the bank holds no decision" no more often; and make fewer LLM calls (the cost) in less time (median).

Behind it: reflect with the decision model passed every suite above and still injected "the bank holds no
decision" on sde-bench, in 14s against agent mode's 7-10s. The wrapped prompt was the search
query, fresh pages about the code hid the facts, and the decision model's "partly" was a
rounding of a "fully". Then, once that was fixed: the distilled search query asked for "implementation
logic" and surfaced the commit that broke each history task's decision. None of that shows on
short questions over an on-topic corpus. The cost and time checks need a decision model
(`HINDSIGHT_EVAL_SET_DECISION_MODEL_PROVIDER=typesafe`, `HINDSIGHT_EVAL_SET_DECISION_MODEL_API_KEY`);
without one only the answers are graded.

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

### Refresh pages — `evals/refresh-pages/test_refresh_pages.py`

**Convergence** (`test_refresh_pages`). A page is created with a source
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

**Delta updates** (`test_refresh_pages_delta`). Convergence grades a page being built; this
grades a page already written, then a wave that changes what it says. Six cases, one per
way an update reaches a page: a memory only the page links to the topic ("Dana left the
company", on a page naming Dana as on-call), an update naming a service by a new name the
same wave introduces, a correction of an earlier figure, a list gaining one member and
losing another, a new item the topic covers, and a wave with nothing for the page at all.
The wave-1 page must state the old value first; the trap is the page still asserting it.
Every wave also carries the hard corpus, so most of what arrives is unrelated.

Behind it: with a decision model set up, a delta refresh asks it which new memories to
read and keeps only the ones it judges could change the page, one memory at a time. A
memory it drops is never read again, and on one cost run a decision model dropped all 20
on-topic memories without a sign anywhere but the page. Run it with the gate on and off
(`HINDSIGHT_EVAL_SET_MENTAL_MODEL_DELTA_USE_DECISION_MODEL=false`, same decision model);
the gate is only worth its savings if this stays level.

**Consistency** (`test_refresh_pages_consistency`). A pump page is built from a
manual (wave one), then refreshed with a notice that changes the filter interval
and the owner (wave two). The trap is a block the refresh left untouched that
still states the old value as current, or still says no change was recorded —
in practice the opening summary. The regression behind it (#5228): the delta
rewrote the paragraph the fact was about and left the rest, so the page
contradicted itself under a clean refresh. One English and one Chinese case (the
reporter's); `HINDSIGHT_EVAL_CONSISTENCY_REPEATS` (default 1) repeats each on
fresh pages, which is how to read a rate.

**Benchmark: refresh cost** (`test_refresh_pages_benchmark_cost`). Not a quality grade: a baseline of what a
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
HINDSIGHT_EVAL_REFRESH_COST_FIXTURE=/tmp/new.zip uv run pytest evals/refresh-pages -k benchmark_cost
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
HINDSIGHT_EVAL_SET_REFLECT_PROMPT_CACHE_ENABLED=true uv run pytest evals/refresh-pages -k benchmark_cost
```

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
as a quality metric tracked over time: correct rate and trap count, overall,
per suite (`by_kind`) and per operation and category (`by_category`). Traps must stay at 0.

A failing run is still published: for a quality metric the red run is the data
point. `scripts/benchmarks/publish-system-evals-results.sh` does the push.

## Two modes

```bash
# minimum acceptance: the cases that have actually regressed
uv run pytest evals

# full: every case of every category
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
reflect budget for `test_reflect`.

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
