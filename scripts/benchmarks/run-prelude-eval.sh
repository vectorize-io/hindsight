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
elif [ "${1:-}" = "--pages" ]; then
    MODE="pages"
    shift
fi

# A DEDICATED pg0 instance, not the shared default. The eval builds ~1000 rows
# and holds a pool while it does; sharing an instance with other sessions gets it
# "sorry, too many clients already" partway through a build, or a start race when
# two things call ensure_running at once. Override to point at a real database.
# Remember what the CALLER chose, before .env gets a say below. Anything set on
# the command line has to survive sourcing .env — comparing two reflect models is
# the whole point of the answer tier, and a .env value silently winning means you
# benchmark the same model twice and never notice.
_CALLER_DB="${HINDSIGHT_API_DATABASE_URL:-}"
_CALLER_MODEL="${HINDSIGHT_API_LLM_MODEL:-}"
_CALLER_PROVIDER="${HINDSIGHT_API_LLM_PROVIDER:-}"
export HINDSIGHT_API_DATABASE_URL="${_CALLER_DB:-pg0://prelude-eval}"

if [ "$MODE" = "answers" ] || [ "$MODE" = "pages" ]; then
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
        # .env is for the LLM credentials, not the database. It typically points
        # at a shared pg0 that other sessions are using, and this eval builds
        # hundreds of rows while holding a pool — which is how the shared
        # instance ends up refusing connections mid-build. The caller's explicit
        # choice wins; otherwise go back to the dedicated instance.
        export HINDSIGHT_API_DATABASE_URL="${_CALLER_DB:-pg0://prelude-eval}"
        [ -n "$_CALLER_MODEL" ] && export HINDSIGHT_API_LLM_MODEL="$_CALLER_MODEL"
        [ -n "$_CALLER_PROVIDER" ] && export HINDSIGHT_API_LLM_PROVIDER="$_CALLER_PROVIDER"
    fi
    if [ "$MODE" = "pages" ]; then
        echo "Running prelude KNOWLEDGE-PAGE eval (create page -> ingest in waves -> judge the page):"
    else
        echo "Running prelude ANSWER eval (real reflect + LLM judge):"
    fi
    echo "  db=${HINDSIGHT_API_DATABASE_URL}"
    echo "  reflect=${HINDSIGHT_API_LLM_PROVIDER:-not set}/${HINDSIGHT_API_LLM_MODEL:-not set}"
    echo "  judge=${HINDSIGHT_TEST_JUDGE_PROVIDER:-gemini}/${HINDSIGHT_TEST_JUDGE_MODEL:-gemini-2.5-flash-lite}"
    echo ""
    cd "$REPO_ROOT/hindsight-dev"
    if [ "$MODE" = "pages" ]; then
        exec uv run python -m benchmarks.prelude.kp_eval "$@"
    fi
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
