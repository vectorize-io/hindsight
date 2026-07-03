# Performance Benchmarks

## System Performance Test (`perf-test`)

Orchestrates independent benchmark suites using mock LLM + pg0 to produce
deterministic, LLM-independent performance baselines.

```bash
uv run perf-test                        # all suites, small scale
uv run perf-test --suite retain         # single suite
uv run perf-test --scale medium         # larger scale
uv run perf-test --output results.json  # save JSON results
```

### Suites

| Suite | What it measures |
|-------|-----------------|
| `retain` | Full retain pipeline with mock LLM: fact extraction callback, embedding generation, DB writes, entity linking |
| `recall` | Pre-populated bank recall: 4-way parallel retrieval (semantic, BM25, graph, temporal), RRF fusion, percentile latency |
| `graph-maintenance` | `run_graph_maintenance_job` in isolation after a batch of deletes: relink pass wall-clock broken down by probe (semantic ANN vs temporal), plus entity/cooccurrence prune counts (#1919) |
| `graph-maintenance-contention` | The maintenance cooccurrence sweep **under concurrent retain load** — drives `prune_stale_cooccurrences` against retain-shaped sorted cooccurrence upserts and counts how many maintenance passes the resulting deadlock silently drops (#2529). See below |
| `stats` | `/stats` endpoint (`get_bank_stats`): uncached aggregation latency (node/link counts + entity rollup join) vs. cached latency, plus cache speedup. Runs with the result cache **disabled** (TTL=0) so the headline numbers are the real per-poll cost |

#### `graph-maintenance-contention` (#2529)

The `graph-maintenance` suite runs the maintenance job **in isolation** — no
concurrent writer — so its Pass 2/3 cooccurrence sweep can never overlap a
retain upsert and never deadlocks. That is exactly why continuous perf never
caught #2529: the `DeadlockDetectedError` is a *contention* phenomenon between
`prune_stale_cooccurrences`' unordered `DELETE` scan and retain's sorted
`(entity_id_1, entity_id_2)` cooccurrence upsert (`entity_resolver._flush_pending`),
and no suite drove both at once. In production a losing deadlock silently drops
a maintenance pass — visible only as slow graph bloat, never as a latency number.

This suite closes the gap: it seeds a hot set of stale cooccurrence pairs, then
runs `upsert_workers` retain-shaped upsert loops against `sweep_workers` loops
calling `run_graph_maintenance_job`. The gate metric is the **deadlock escape
rate** — of the deadlocks the sweep hits, what fraction escaped the job and
dropped a pass:

- **Without the fix:** every deadlock escapes → escape rate ≈ **100%** → suite **fails**.
- **With the fix** (the sweep wrapped in a jittered `retry_with_backoff` with a larger retry budget): the sweep retries → escape rate ≈ **0%** → suite **passes**. A vanishingly small tail can still slip through under the deliberately brutal multi-sweep synthetic load here; a realistic single-sweep-per-bank load drops nothing.

The suite fails if the escape rate exceeds `GRAPH_CONTENTION_ESCAPE_RATE_THRESHOLD`
(0.5), or if no deadlock reproduced at all (a hollow run that proves nothing).

Measured (`--scale small`): **main** ≈ 100% escape (every deadlock drops a pass)
→ FAIL; **fixed** 0% escape (deadlocks still occur, none escape) → PASS.

### Scale Configurations

| Scale | Retain items | Recall bank size | Recall iterations | Recall concurrency |
|-------|-------------|-----------------|-------------------|-------------------|
| `tiny` | 20 | 20 | 5 | 1 |
| `small` | 200 | 200 | 20 | 4 |
| `medium` | 1,000 | 1,000 | 50 | 8 |
| `large` | 5,000 | 5,000 | 100 | 16 |
| `huge` | — (`stats` only) | — | — | — |

The `huge` scale is a prod-simulation for the `stats` suite only: it bulk-loads
~500k units and ~17.8M physical `memory_links` rows (semantic + temporal +
caused_by) via COPY, plus a `unit_entities` set whose `LEAST(n-1, 10)` rollup
reproduces the ~110.9k *derived* entity links (entity edges aren't stored). The
other suites fall back to `large` sizing at this scale, so prefer
`--suite stats --scale huge`. Bulk-load takes a few minutes; FK triggers and
non-essential `memory_links` indexes are dropped during COPY and restored after.

```bash
uv run perf-test --suite stats --scale huge
```

### CI

The `perf-test.yml` workflow runs daily and on manual dispatch. Scale is
configurable via workflow input (defaults to `small`). Results are uploaded as
artifacts with 90-day retention.

## Standalone Benchmarks

### `retain_perf.py`

Retain operation benchmark with two modes:

- **In-memory**: Direct `retain_batch_async` call (single file or directory)
- **Async**: Full async workflow with mock LLM, worker poller, and deadlock stress testing

```bash
uv run python hindsight-dev/benchmarks/perf/retain_perf.py \
    --document <file_path> --in-memory
```

### `recall_perf.py`

Large-bank recall load test. Generates synthetic banks with Zipf-distributed
entities, then benchmarks recall latency at scale with per-step breakdown.

```bash
# Generate bank
uv run python hindsight-dev/benchmarks/perf/recall_perf.py generate \
    --bank-id my-bank --scale small

# Benchmark
uv run python hindsight-dev/benchmarks/perf/recall_perf.py benchmark \
    --bank-id my-bank --query "database migration" --iterations 20

# Clean up
uv run python hindsight-dev/benchmarks/perf/recall_perf.py clean \
    --bank-id my-bank
```

Scales: `tiny` (1), `mini` (50), `small` (2K), `medium` (10K), `large` (33K), `very-large` (100K).
