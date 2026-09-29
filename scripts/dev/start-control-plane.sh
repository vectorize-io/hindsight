#!/bin/bash
set -e

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR" || exit 1

# Check if .env exists in workspace root
if [ ! -f "$ROOT_DIR/.env" ]; then
  echo "⚠️  Warning: .env not found in workspace root at $ROOT_DIR/.env"
  echo "📝 Please create a .env file if you need to set HINDSIGHT_CP_DATAPLANE_API_URL"
  echo "   Default will use http://localhost:8888"
  echo ""
fi

echo "🔨 Building TypeScript SDK first to ensure it's up to date..."
npm run build -w @vectorize-io/hindsight-client
echo "✅ SDK built successfully"
echo ""

echo "🚀 Starting Control Plane (Next.js dev server)..."
# Save caller-provided values before .env can overwrite them
_CALLER_PORT="${PORT:-}"
_CALLER_DATAPLANE_URL="${HINDSIGHT_CP_DATAPLANE_API_URL:-}"

if [ -f "$ROOT_DIR/.env" ]; then
  echo "📄 Loading environment from $ROOT_DIR/.env"
  # Load env vars from root .env file
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

# Map prefixed env vars to Next.js standard vars
export HOSTNAME="${HINDSIGHT_CP_HOSTNAME:-0.0.0.0}"
# Caller-provided values take priority over .env
export PORT="${_CALLER_PORT:-${HINDSIGHT_CP_PORT:-9999}}"
export HINDSIGHT_CP_DATAPLANE_API_URL="${_CALLER_DATAPLANE_URL:-${HINDSIGHT_CP_DATAPLANE_API_URL:-http://localhost:8888}}"

# Run dev server
npm run dev -w @vectorize-io/hindsight-control-plane