#!/bin/bash
# Microbenchmark the within-batch semantic link pass on the retain path (#3977).
#
# Usage:
#   ./scripts/benchmarks/run-semantic-within-batch-bench.sh
#   ./scripts/benchmarks/run-semantic-within-batch-bench.sh --sizes 500 5000 --workloads clustered

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT/hindsight-dev"

exec uv run semantic-within-batch-bench "$@"
