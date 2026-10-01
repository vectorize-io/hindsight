# Source-relative language prevention

This candidate does not deploy or change process configuration. `observe` still accepts
mismatches. The enforceable mode is `llm_language_integrity=reject` with
`llm_output_language` unset. `retry` corrects once but remains fail-open; it is not
prevention. An explicit output-language configuration is the supported deliberate
translation override and bypasses source-language checking.

## Contract

- Retain and consolidation perform one language-correction attempt, then reject
  mismatched **or unchecked** output in reject mode. Detector failures fail closed.
- Per-output text-free verdicts are `copied`, `preserved`, `mismatch`, or `unchecked`,
  including source keys, reason and policy version. Aggregate checked/abstained
  counters do not substitute for these verdicts. Short/ambiguous natural language
  is unchecked, not a validated pass; this can hold legitimate work for review.
- Source-evidenced copied spans and quotations are not translation errors. They
  do not license translating the surrounding account. This is language validity,
  not assertion/fact validation: copying a quotation does not establish its truth.
- Genuine multilingual source prose supports corresponding output-language
  segments. Recognizable code and source-evidenced names are not forced into English.
  Capitalization and backticks alone are not exemptions.
- Consolidation obtains bank-scoped original chunks for incoming facts and the
  visible source facts of recalled observations. Final dedup merges are separately
  checked against originals of incoming facts, the update anchor and the nearest
  twin (including twins not shown in the main recall), before any batch writes. Generated fact text is never a
  fallback language authority. Missing original evidence remains unchecked and
  blocks novel prose in reject mode. Stores that cannot implement original-chunk
  retrieval via `get_chunk_texts` fail closed in strict consolidation instead of
  guessing.
- Original-source language codes accompany instructions so an old translated fact
  is not also the only language cue in the corrective prompt.

The detector remains heuristic, not a semantic proof for every language or code
syntax. Source prose/JSON decoding and language profiles are shared per distinct
chunk per check context. Same-Latin clause checks use bounded lexical-candidate
windows; normal source profiling is not repeated for each fact. Quoted source
prose grants copy evidence, not authority to translate surrounding prose. Retain
validates generated dimensions separately from deterministic formatting labels.

## Recovery and rollout (separate authorization required)

1. Independently review and merge a pinned candidate; build a release and run the
   offline fixtures plus isolated PostgreSQL tests below.
2. Preserve the existing runtime, effective process config and queue/operation state
   for rollback. Deploy/restart only with explicit approval.
3. `llm_language_integrity` is a hierarchical BankConfig field. With authorized
   rollout, update the target bank to `reject` and read back its effective config.
   `llm_output_language` stays process-scoped and must be unset for this guard to
   apply. Null bank modes inherit tenant or process defaults. Retain strategies
   may explicitly override the mode, while a null mode inherits the bank policy.
   Inventory those strategies before claiming prevention for every retain path.
   Other banks retain their existing resolved policies. Do not claim prevention
   while observe/retry or an overriding output language remains effective.
4. Monitor per-output unchecked/mismatch/copied counters, failed operations and
   `consolidation_failed_at` facts / `memories_failed` job results. A consolidation
   job can complete successfully while reporting held facts; operation failure
   alone is not sufficient monitoring.
   Missing chunks, attachments without sufficient source text, or low-confidence
   prose can reject legitimate records. Review these rather than silently retrying
   with language checks disabled. Verbatim ingestion is a source-preserving path.
5. Consolidation rejection happens before the rejected batch's write transaction:
   existing observations and original documents/chunks remain intact. Rejections
   use bounded adaptive bisection, then `consolidation_failed_at` on isolated failed
   facts (not a successful `consolidated_at` stamp). Source facts remain stored;
   they are excluded from pending selection so later healthy work drains. Each
   attempted sub-batch has at most one language correction per generation stage;
   bisection can attempt smaller sub-batches. Detector/infrastructure failures
   still propagate and do not classify all facts as bad. Earlier successful scopes
   retain the existing multi-scope transaction granularity; this is not job-wide
   atomicity. Review held facts through the existing failed-consolidation lifecycle;
   clearing/retrying historical holds still requires separate authorization.
   A rejected synchronous retain leaves its caller responsible for retaining and
   resubmitting the original input. Async operation input is the recovery source;
   do not discard failed operations. Multi-chunk retain has existing streaming
   transaction boundaries, not a new whole-document atomicity guarantee.
6. Rollback means reverting release/policy after separate approval, not deleting
   source data or marking rejected work successful. Existing-record cleanup,
   invalidation, retries and curation are separately authorized work.

## Deterministic tests (no provider generation)

Run from `hindsight-api-slim`, in an isolated test database (never production):

```sh
HINDSIGHT_API_DATABASE_URL=pg0://language-prevention-review:55783 HF_HUB_OFFLINE=1 \
uv run pytest -n 0 tests/test_language_prevention.py \
  tests/test_language_prevention_atomicity.py tests/test_language_integrity.py \
  tests/test_language_integrity_retain.py tests/test_consolidation_retry_budget.py \
  tests/test_consolidation_language_sources.py tests/test_consolidation_batch_atomicity.py \
  tests/test_language_prevention_review.py tests/test_language_prevention_dedup.py \
  tests/test_consolidation_dedup.py tests/test_fact_extraction_retry.py \
  tests/test_worker_retry_knobs.py
```

`test_consolidation_output_language.py` and `test_retain_reflect_output_language.py`
use live-provider fixtures and must not be included in an offline/no-probe run.
The unit tests cover the configured-translation bypass without a provider.
Private historical records must stay outside this repository; replay their actual
fact/observation text against their saved original chunks, including the quoted
French span and English-to-Spanish/German/Chinese drift.
