#!/bin/bash
set -e

# Prelude Retrieval Eval Runner
# Measures whether query WORDING moves what reflect's forced hierarchical prelude
# retrieves: the question verbatim (floor) vs a hand-written ideal query (ceiling).
# See hindsight-dev/benchmarks/prelude/README.md.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Deliberately NOT sourcing .env: the corpus must be authored rather than
# extracted, so this always runs against the mock provider. Neither arm calls an
# LLM, so no API key is needed and a stray provider setting would only make the
# fixture non-deterministic.
export HINDSIGHT_API_LLM_PROVIDER=mock
export HINDSIGHT_API_LLM_MODEL=mock
export HINDSIGHT_API_LLM_API_KEY=unused
export HINDSIGHT_API_DATABASE_URL="${HINDSIGHT_API_DATABASE_URL:-pg0}"

echo "Running prelude retrieval eval:"
echo "  db=${HINDSIGHT_API_DATABASE_URL} provider=mock (corpus is authored, not extracted)"
echo ""

cd "$REPO_ROOT/hindsight-dev"
uv run python -m benchmarks.prelude.floor_ceiling "$@"
