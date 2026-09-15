#!/bin/bash
# Microbenchmark fact extraction prompt & schema hoisting on retain paths.
#
# Usage:
#   ./scripts/benchmarks/run-fact-extraction-bench.sh
#   ./scripts/benchmarks/run-fact-extraction-bench.sh --repeats 10
#   ./scripts/benchmarks/run-fact-extraction-bench.sh --json /tmp/fact_extraction_bench.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT/hindsight-dev"

exec uv run fact-extraction-bench "$@"
