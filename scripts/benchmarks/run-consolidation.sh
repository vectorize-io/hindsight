#!/bin/bash
set -e

# Consolidation Performance Benchmark Runner
# Measures consolidation throughput (op/sec) and identifies bottlenecks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source .env if it exists
if [ -f "$REPO_ROOT/.env" ]; then
    source "$REPO_ROOT/.env"
    echo "Loaded environment from .env"
fi

# Default configuration
NUM_MEMORIES="${NUM_MEMORIES:-100}"

# Enable observations (required for consolidation)
export HINDSIGHT_API_ENABLE_OBSERVATIONS=true

echo "Running consolidation benchmark with configuration:"
echo "  NUM_MEMORIES=$NUM_MEMORIES"
echo "  HINDSIGHT_API_LLM_PROVIDER=${HINDSIGHT_API_LLM_PROVIDER:-not set}"
echo "  HINDSIGHT_API_LLM_MODEL=${HINDSIGHT_API_LLM_MODEL:-not set}"
echo ""

# Run benchmark
cd "$REPO_ROOT"
uv run python -m benchmarks.consolidation.consolidation_benchmark

echo ""
echo "Benchmark complete! Check benchmarks/results/ for detailed results."
