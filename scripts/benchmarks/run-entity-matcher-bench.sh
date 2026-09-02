#!/bin/bash
# Microbenchmark candidate entity string similarity matching on retain entity resolution paths.
#
# Usage:
#   ./scripts/benchmarks/run-entity-matcher-bench.sh
#   ./scripts/benchmarks/run-entity-matcher-bench.sh --repeats 10
#   ./scripts/benchmarks/run-entity-matcher-bench.sh --workload large_batch_1000
#   ./scripts/benchmarks/run-entity-matcher-bench.sh --json /tmp/entity_matcher_bench.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT/hindsight-dev"

exec uv run entity-matcher-bench "$@"
