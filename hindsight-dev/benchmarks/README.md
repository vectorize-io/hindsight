# Hindsight Benchmarks

This directory contains benchmark suites for evaluating Hindsight's memory capabilities.

## Prerequisites

1. Set up your environment variables in `.env` at the project root:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. Make sure you have `uv` installed.

## Available Benchmarks

### LoComo and LongMemEval — see AMB

Both live in [AMB](https://github.com/vectorize-io/agent-memory-benchmark) now,
along with BEAM, PersonaMem and the coding-agent suite. AMB owns their datasets,
prompts, answer model, judge and scoring, and publishes to
[agentmemorybenchmark.ai](https://agentmemorybenchmark.ai); the in-repo copies
were a second implementation of the same two datasets, free to drift from the
published numbers. Run them from `hindsight-system-evals`, against a local server
or any deployment:

```bash
cd hindsight-system-evals
uv run run-amb --dataset locomo --split locomo10
uv run run-amb --dataset longmemeval --split s -- --category single-session-user
uv run run-amb --dataset locomo --split locomo10 --api-url https://api.dev.example
```

### Consolidation Performance

Tests consolidation throughput and identifies bottlenecks.

```bash
./scripts/benchmarks/run-consolidation.sh

# With custom memory count
NUM_MEMORIES=200 ./scripts/benchmarks/run-consolidation.sh
```

### System Performance Test

Runs retain throughput and recall latency benchmarks using mock LLM + pg0.
No external dependencies needed.

```bash
# Run all suites at default (small) scale
./scripts/benchmarks/run-perf-test.sh

# Quick smoke test
./scripts/benchmarks/run-perf-test.sh --scale tiny

# Single suite
./scripts/benchmarks/run-perf-test.sh --suite retain

# Save results
./scripts/benchmarks/run-perf-test.sh --output results.json
```

**Options:**
- `--scale {tiny,small,medium,large}` - Test scale (default: small)
- `--suite {retain,recall}` - Run specific suite (default: all)
- `--output PATH` - Save JSON results to file

See [perf/README.md](perf/README.md) for detailed documentation.

### Multimodal Retain

Is the information in the pictures reaching memory? Retains the same KB articles
twice — once with their images inline, once with the images absent — and asks
questions that only the images can answer. See
[`multimodal_retain/README.md`](multimodal_retain/README.md) for the corpus, the
metrics and the recorded baseline.

Needs a server with a **vision-capable retain LLM**; without one the multimodal
arm is refused with `422` and the report says so rather than reporting zeros.

```bash
uv run python -m benchmarks.multimodal_retain run --api-url http://localhost:8917 --build my-branch

# re-render a saved artifact, no LLM calls
uv run python -m benchmarks.multimodal_retain report benchmarks/results/multimodal_retain/<artifact>.json
```

**Options:**
- `--article NAME` - Run one article (repeatable)
- `--build LABEL` - Label recorded in the artifact
- `--out PATH` - Where to write the artifact
- `--keep-banks` - Leave the benchmark banks behind for inspection

### Token Counting (micro)

Measures the token counting recall does per candidate fact, chunk and
reranker document — wall time, CPU time (all threads) and peak *traced*
allocation, against a set of cheaper spellings of the same count. No DB, no LLM,
no network.

```bash
# All workloads
./scripts/benchmarks/run-token-count-bench.sh

# One call site, more repeats, raw results
./scripts/benchmarks/run-token-count-bench.sh --workload facts_200 --repeats 10 --json tok.json

# Against real text rather than the synthetic generator (a small vocabulary
# flatters every BPE implementation)
./scripts/benchmarks/run-token-count-bench.sh --corpus /path/to/text

# On another vocabulary instead of production's o200k_base
./scripts/benchmarks/run-token-count-bench.sh --encoding cl100k_base

# Include the tiktoken baseline (deliberately not a project dependency)
cd hindsight-dev && uv run --with tiktoken token-count-bench
```

**Options:**
- `--workload NAME` - Run one workload (repeatable); shaped after a real call site
- `--encoding NAME` - Vocabulary to measure (default `o200k_base`, what production
  uses). A 200k vocabulary is a different amount of work per byte, so a ranking
  measured on one does not transfer to another for free.
- `--repeats N` - Timed repeats per variant, best-of (default: 5)
- `--threads N` - `num_threads` for the parallel batch variant
- `--corpus PATH` - Slice workload texts out of a real text file
- `--no-conformance` - Skip the count-agreement check on adversarial inputs
- `--json PATH` - Save raw results

Every variant is checked against the production count first, on inputs that have
broken token counting before (special-token literals, unicode, empty). A variant
that counts differently — or raises — is reported as such, never as a speedup.

## Results

Results are saved in JSON format in each benchmark's `results/` directory.
