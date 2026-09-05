#!/bin/bash
# Microbenchmark entity resolution and in-batch deduplication on retain paths.
#
# Usage:
#   ./scripts/benchmarks/run-entity-resolver-bench.sh
#   ./scripts/benchmarks/run-entity-resolver-bench.sh --repeats 10
#   ./scripts/benchmarks/run-entity-resolver-bench.sh --json /tmp/entity_bench.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT/hindsight-dev"

exec uv run entity-resolver-bench "$@"
