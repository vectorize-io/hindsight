#!/bin/bash
# Run system performance tests (mock LLM + pg0)
#
# Usage:
#   ./scripts/benchmarks/run-perf-test.sh                    # all suites, small scale
#   ./scripts/benchmarks/run-perf-test.sh --scale tiny       # quick smoke test
#   ./scripts/benchmarks/run-perf-test.sh --suite retain     # single suite
#   ./scripts/benchmarks/run-perf-test.sh --output results.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT/hindsight-dev"

exec uv run perf-test "$@"
