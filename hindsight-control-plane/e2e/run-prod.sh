#!/usr/bin/env bash
# Run Playwright e2e tests against the production deployment.
#
# Prerequisites:
#   1. Copy .env.e2e → .env.local and fill in the tenant API keys
#   2. Client certificates available at ../openclaw-infra/{ca,client}.{crt,key}
#
# Usage:
#   ./e2e/run-prod.sh              # run all tests
#   ./e2e/run-prod.sh --headed     # run with browser visible
#   ./e2e/run-prod.sh --debug      # run in debug mode
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$CP_DIR"

# ── Validate .env.local has tenant keys ──────────────────────────────────────
if [[ ! -f .env.local ]]; then
  echo "ERROR: .env.local not found."
  echo "Copy .env.e2e to .env.local and fill in the tenant API keys."
  echo "  cp .env.e2e .env.local"
  exit 1
fi

if grep -q "KEY_HERE" .env.local; then
  echo "ERROR: .env.local still has placeholder keys."
  echo "Fill in the real tenant API keys from the prod instance:"
  echo "  ssh openclaw@34.208.169.77 'grep HINDSIGHT_KEY_ /opt/openclaw/.env'"
  exit 1
fi

# ── Validate certs exist ─────────────────────────────────────────────────────
# CP_DIR is hindsight-control-plane/, go up to code/ then into openclaw-infra/
INFRA_DIR="${MTLS_INFRA_DIR:-$CP_DIR/../../../openclaw-infra}"
for f in ca.crt client.crt client.key; do
  if [[ ! -f "$INFRA_DIR/$f" ]]; then
    echo "ERROR: Missing $INFRA_DIR/$f"
    echo "Ensure openclaw-infra repo is at ../openclaw-infra relative to hindsight-contrib/"
    exit 1
  fi
done

# ── Start mTLS proxy in background ──────────────────────────────────────────
echo "Starting mTLS proxy..."
MTLS_INFRA_DIR="$INFRA_DIR" node e2e/mtls-proxy.mjs &
PROXY_PID=$!

cleanup() {
  echo "Stopping mTLS proxy (pid $PROXY_PID)..."
  kill "$PROXY_PID" 2>/dev/null || true
  wait "$PROXY_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for proxy to be ready
for i in $(seq 1 10); do
  if curl -sf http://localhost:18888/ >/dev/null 2>&1 || [[ $i -eq 10 ]]; then
    break
  fi
  sleep 0.5
done
echo "mTLS proxy ready on :18888"

# ── Check if dev server is running ───────────────────────────────────────────
if ! curl -sf http://localhost:9999/ >/dev/null 2>&1; then
  echo ""
  echo "WARNING: Dev server not detected on :9999."
  echo "Start it in another terminal:  npm run dev"
  echo "Waiting for dev server..."
  for i in $(seq 1 30); do
    if curl -sf http://localhost:9999/ >/dev/null 2>&1; then
      break
    fi
    if [[ $i -eq 30 ]]; then
      echo "ERROR: Dev server not responding after 15s. Start it with: npm run dev"
      exit 1
    fi
    sleep 0.5
  done
fi
echo "Dev server ready on :9999"

# ── Run Playwright ───────────────────────────────────────────────────────────
echo ""
echo "Running Playwright tests..."
npx playwright test "$@"
