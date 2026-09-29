#!/bin/bash
set -e

# Observation Duplication Benchmark Runner
# Measures the duplicate-observation rate consolidation produces over a dataset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source .env if it exists
if [ -f "$REPO_ROOT/.env" ]; then
    source "$REPO_ROOT/.env"
    echo "Loaded environment from .env"
fi

# Enable observations (required for consolidation)
export HINDSIGHT_API_ENABLE_OBSERVATIONS=true
# Note: the benchmark uses SyncTaskBackend (inline/serial task execution) and disables
# auto-consolidation per-bank, so its explicit drain loop is the sole consolidator with no
# background worker racing it. See benchmarks/obs/obs_benchmark.py for the reasoning.

echo "Running observation duplication benchmark with configuration:"
echo "  HINDSIGHT_API_LLM_PROVIDER=${HINDSIGHT_API_LLM_PROVIDER:-not set}"
echo "  HINDSIGHT_API_LLM_MODEL=${HINDSIGHT_API_LLM_MODEL:-not set}"
echo ""

# Run benchmark
cd "$REPO_ROOT"
uv run python -m benchmarks.obs.obs_benchmark

echo ""
echo "Benchmark complete! Check benchmarks/results/ for detailed results."
