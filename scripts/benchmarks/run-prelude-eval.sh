#!/bin/bash
set -e

# Prelude Retrieval Eval Runner
# Measures whether query WORDING moves what reflect's forced hierarchical prelude
# retrieves: the question verbatim (floor) vs a hand-written ideal query (ceiling).
# See hindsight-dev/benchmarks/prelude/README.md.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --answers runs the ANSWER tier: real reflect end to end, graded by an LLM
# judge. Everything else runs the deterministic retrieval tier.
MODE="retrieval"
if [ "${1:-}" = "--answers" ]; then
    MODE="answers"
    shift
fi

export HINDSIGHT_API_DATABASE_URL="${HINDSIGHT_API_DATABASE_URL:-pg0}"

if [ "$MODE" = "answers" ]; then
    # Needs a real reflect model AND a judge key, so .env matters here. The corpus
    # is still seeded deterministically: the fixture scripts retain's extraction
    # regardless of which provider reflect runs on.
    if [ -f "$REPO_ROOT/.env" ]; then
        # `set -a` so the sourced values are EXPORTED: the eval runs in a child
        # process and plain shell variables never reach it.
        set -a
        source "$REPO_ROOT/.env"
        set +a
        echo "Loaded environment from .env"
    fi
    echo "Running prelude ANSWER eval (real reflect + LLM judge):"
    echo "  db=${HINDSIGHT_API_DATABASE_URL}"
    echo "  reflect=${HINDSIGHT_API_LLM_PROVIDER:-not set}/${HINDSIGHT_API_LLM_MODEL:-not set}"
    echo "  judge=${HINDSIGHT_TEST_JUDGE_PROVIDER:-gemini}/${HINDSIGHT_TEST_JUDGE_MODEL:-gemini-2.5-flash-lite}"
    echo ""
    cd "$REPO_ROOT/hindsight-dev"
    exec uv run python -m benchmarks.prelude.answer_eval "$@"
fi

# Retrieval tier. Deliberately NOT sourcing .env: the corpus must be authored
# rather than extracted, so this always runs against the mock provider. Neither
# arm calls an LLM, so no API key is needed and a stray provider setting would
# only make the fixture non-deterministic.
export HINDSIGHT_API_LLM_PROVIDER=mock
export HINDSIGHT_API_LLM_MODEL=mock
export HINDSIGHT_API_LLM_API_KEY=unused

echo "Running prelude RETRIEVAL eval (deterministic, no LLM):"
echo "  db=${HINDSIGHT_API_DATABASE_URL} provider=mock (corpus is authored, not extracted)"
echo ""

cd "$REPO_ROOT/hindsight-dev"
uv run python -m benchmarks.prelude.floor_ceiling "$@"
