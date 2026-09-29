#!/bin/bash
# Microbenchmark the token counting on the recall path (no DB, no LLM).
#
# Usage:
#   ./scripts/benchmarks/run-token-count-bench.sh                       # all workloads
#   ./scripts/benchmarks/run-token-count-bench.sh --repeats 10
#   ./scripts/benchmarks/run-token-count-bench.sh --workload facts_200
#   ./scripts/benchmarks/run-token-count-bench.sh --encoding cl100k_base
#   ./scripts/benchmarks/run-token-count-bench.sh --json /tmp/tok.json
#
# To include the tiktoken baseline this repo migrated away from (no longer a
# dependency):  cd hindsight-dev && uv run --with tiktoken token-count-bench

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT/hindsight-dev"

exec uv run token-count-bench "$@"
