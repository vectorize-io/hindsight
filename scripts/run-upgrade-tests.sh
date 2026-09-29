#!/bin/bash
#
# Run upgrade tests locally
#
# Usage:
#   ./scripts/run-upgrade-tests.sh
#
# Environment variables:
#   HINDSIGHT_API_LLM_PROVIDER - LLM provider (default: groq)
#   HINDSIGHT_API_LLM_API_KEY  - LLM API key (or GROQ_API_KEY)
#   HINDSIGHT_API_LLM_MODEL    - LLM model
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Running Hindsight Upgrade Tests ==="
echo ""

# Check for LLM API key
if [ -z "${HINDSIGHT_API_LLM_API_KEY:-}" ] && [ -z "${GROQ_API_KEY:-}" ]; then
    echo "Warning: No LLM API key found. Set HINDSIGHT_API_LLM_API_KEY or GROQ_API_KEY"
fi

# Load .env if present
if [ -f "$ROOT_DIR/.env" ]; then
    echo "Loading .env file..."
    set -a
    source "$ROOT_DIR/.env"
    set +a
fi

cd "$ROOT_DIR/hindsight-dev"

# Run tests
echo "Running upgrade tests..."
uv run pytest upgrade_tests/ -v "$@"
