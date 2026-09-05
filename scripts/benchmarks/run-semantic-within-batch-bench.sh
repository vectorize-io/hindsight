#!/bin/bash
# Microbenchmark within-batch semantic link calculation on retain path.
#
# Usage:
#   ./scripts/benchmarks/run-semantic-within-batch-bench.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT/hindsight-dev"

exec uv run semantic-within-batch-bench "$@"
