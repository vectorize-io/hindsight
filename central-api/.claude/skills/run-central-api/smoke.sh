#!/usr/bin/env bash
# Smoke-tests the running central-api.
# Usage: ./smoke.sh [BASE_URL]
# Defaults to http://localhost:8000

set -euo pipefail

BASE="${1:-http://localhost:8000}"

pass() { echo "  PASS $1"; }
fail() { echo "  FAIL $1: $2"; exit 1; }

check() {
  local label="$1" url="$2" expected="$3"
  local body
  body=$(curl -sf "$url") || fail "$label" "curl failed"
  echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); assert '$expected' in str(d), repr(d)" \
    || fail "$label" "expected '$expected' not found in: $body"
  pass "$label"
}

check_post() {
  local label="$1" url="$2" data="$3" expected="$4"
  local body
  body=$(curl -sf -X POST "$url" -H "Content-Type: application/json" -d "$data") \
    || fail "$label" "curl failed"
  echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); assert '$expected' in str(d), repr(d)" \
    || fail "$label" "expected '$expected' not found in: $body"
  pass "$label"
}

echo "Smoking $BASE ..."
check      "GET /"                  "$BASE/"                       "central-api"
check      "GET /health"            "$BASE/health"                 "ok"
check      "GET /api/health/engines" "$BASE/api/health/engines"   "memlord"
check_post "POST /api/memory/search" "$BASE/api/memory/search" \
           '{"query":"hello","engine":"memlord","top_k":3}' "count"
check_post "POST /api/context/build" "$BASE/api/context/build" \
           '{"query":"test","engine":"memlord"}' "audit_trace_id"

echo ""
echo "All checks passed."
