#!/bin/bash
# Microbenchmark bank transfer export archive streaming with multimodal attachments.
#
# Usage:
#   ./scripts/benchmarks/run-transfer-streaming-bench.sh
#   ./scripts/benchmarks/run-transfer-streaming-bench.sh --repeats 5
#   ./scripts/benchmarks/run-transfer-streaming-bench.sh --workload multimodal_medium
#   ./scripts/benchmarks/run-transfer-streaming-bench.sh --json /tmp/transfer_bench.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT/hindsight-dev"

exec uv run transfer-streaming-bench "$@"
