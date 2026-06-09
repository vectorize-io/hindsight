# Observation duplication benchmark

Measures how many **duplicate observations** consolidation produces — a quality
signal for the consolidation pipeline (complements the perf-focused
`consolidation` benchmark).

For each document under `datasets/`, it ingests the content into a fresh bank,
runs consolidation, then reuses the observation-dedup tool
(`hindsight_dev.obs_dedup`) to score:

- **exact duplicates** — observations with identical (normalised) text in a scope
- **near duplicates** — cosine-similarity clusters at thresholds (0.97, 0.92)

Headline metric: **duplication rate** = redundant observations / total. Lower is
better.

## Run

```bash
./scripts/benchmarks/run-obs.sh
# or
cd hindsight-dev && uv run python -m benchmarks.obs.obs_benchmark
```

Requires a real LLM (set `HINDSIGHT_API_LLM_PROVIDER` / `_MODEL` / `_API_KEY`).
Results are written to `benchmarks/results/obs_benchmark_<ts>.json`.

## Extending

Drop more `*.txt` transcripts into `datasets/` — one file per scenario. Keep them
synthetic and PII-free. Documents that restate the same durable facts across many
turns are the ones that stress consolidation's dedup behaviour.
