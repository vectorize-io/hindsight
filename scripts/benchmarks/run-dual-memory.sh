#!/bin/bash
# D experiment: three-arm dual-memory comparison (HINDSIGHT_GRAPHITI_EVAL_4WAY.md stage 1).
#
#   ./scripts/benchmarks/run-dual-memory.sh --arms hindsight,graphiti,dual --conversations 3
#   ./scripts/benchmarks/run-dual-memory.sh --mock          # plumbing smoke test, no API key
#
# Needs in .env: HINDSIGHT_API_LLM_PROVIDER/_API_KEY/_MODEL (answering+judging+ingestion).
# The graphiti/dual arms additionally need:
#   uv pip install 'graphiti-core[falkordb]'   (in hindsight-dev)
#   docker run -d -p 6379:6379 falkordb/falkordb
#   GRAPHITI_LLM_* env vars if the Graphiti LLM differs from HINDSIGHT_API_LLM_*.
set -e

cd "$(dirname "$0")/../.."

ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  echo "📄 Loading environment from $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

cd hindsight-dev
exec uv run python -m benchmarks.dual_memory.run_dual_memory "$@"
