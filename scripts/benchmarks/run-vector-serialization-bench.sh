#!/bin/bash
# Microbenchmark pgvector float vector serialization on retain and import paths.
#
# Usage:
#   ./scripts/benchmarks/run-vector-serialization-bench.sh
#   ./scripts/benchmarks/run-vector-serialization-bench.sh --repeats 10
#   ./scripts/benchmarks/run-vector-serialization-bench.sh --workload batch_200_openai_1536
#   ./scripts/benchmarks/run-vector-serialization-bench.sh --json /tmp/vec_bench.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT/hindsight-dev"

exec uv run vector-serialization-bench "$@"
