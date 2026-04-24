#!/data/data/com.termux/files/usr/bin/bash
# Start the Hindsight API server on Android/Termux.
#
# Prerequisites:
#   - Run setup-termux.sh first
#   - Run install-python-deps.sh first
#   - Set LLM API key in environment or .env file
#
# Usage: bash scripts/android/start-api.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
API_DIR="$REPO_ROOT/hindsight-api-slim"

# Ensure wake lock
termux-wake-lock 2>/dev/null || true

# Start PostgreSQL if not running
if ! pg_isready -q 2>/dev/null; then
    echo ">>> Starting PostgreSQL..."
    pg_ctl -D "$HOME/pg-data" -l "$HOME/pg.log" start
    sleep 2
fi

# Activate virtualenv
source "$API_DIR/.venv/bin/activate"

# Set Android-specific defaults
export HINDSIGHT_API_DATABASE_URL="${HINDSIGHT_API_DATABASE_URL:-postgresql://$(whoami)@localhost/hindsight}"
export HINDSIGHT_API_HOST="${HINDSIGHT_API_HOST:-0.0.0.0}"
export HINDSIGHT_API_PORT="${HINDSIGHT_API_PORT:-8741}"

# Disable local ML models by default on Android (too heavy for phone)
export HINDSIGHT_API_EMBEDDINGS_PROVIDER="${HINDSIGHT_API_EMBEDDINGS_PROVIDER:-tei}"
export HINDSIGHT_API_RERANKER_PROVIDER="${HINDSIGHT_API_RERANKER_PROVIDER:-tei}"

# Load .env if present
if [ -f "$REPO_ROOT/.env" ]; then
    echo ">>> Loading .env..."
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

echo "=== Starting Hindsight API ==="
echo "  Database: $HINDSIGHT_API_DATABASE_URL"
echo "  Listen: $HINDSIGHT_API_HOST:$HINDSIGHT_API_PORT"
echo "  LLM Provider: ${HINDSIGHT_API_LLM_PROVIDER:-not set}"
echo ""

cd "$API_DIR"
python -m uvicorn hindsight_api.main:create_app --factory \
    --host "$HINDSIGHT_API_HOST" \
    --port "$HINDSIGHT_API_PORT"
