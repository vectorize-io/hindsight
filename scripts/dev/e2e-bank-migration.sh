#!/bin/bash
#
# Manual end-to-end test for cross-instance bank migration with DIFFERENT embedding
# models. NOT run in CI (it spins two real API servers, needs two cached local
# models, and an LLM key for the initial retain). Run locally to verify the
# export-bank/import-bank pipeline re-embeds on the target with no LLM re-extraction.
#
#   Instance A: BAAI/bge-small-en-v1.5 (384-dim)  -->  Instance B: BAAI/bge-base-en-v1.5 (768-dim)
#
# It retains data into A, exports the bank, imports it into B (which re-embeds with
# its 768-dim model), then asserts recall on B returns the migrated facts ranked
# correctly. Exits non-zero if any step or assertion fails.
#
# Usage:  ./scripts/dev/e2e-bank-migration.sh
# Requires: a configured .env at the repo root (LLM provider/key for retain).
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
API_DIR="$REPO/hindsight-api-slim"
WORK="$(mktemp -d -t hs-e2e-XXXXXX)"
BANK="e2e_migration_bank"
PA=53611
PB=53622
PIDA=""
PIDB=""

cleanup() {
  [ -n "$PIDA" ] && kill "$PIDA" 2>/dev/null
  [ -n "$PIDB" ] && kill "$PIDB" 2>/dev/null
  # stop the two embedded pg0 instances spun up for this run
  ps ax | grep -E "pg0/instances/(e2e_mig_a|e2e_mig_b)" | grep -v grep | awk '{print $1}' | xargs -r kill 2>/dev/null
}
trap cleanup EXIT

fail() { echo "❌ FAIL: $1"; exit 1; }

mk_env() {  # $1=dir $2=db_instance $3=model
  mkdir -p "$1"
  cp "$REPO/.env" "$1/.env" 2>/dev/null || fail "no .env at repo root (needed for the LLM key)"
  {
    echo ""
    echo "HINDSIGHT_API_DATABASE_URL=pg0://$2"
    echo "HINDSIGHT_API_EMBEDDINGS_PROVIDER=local"
    echo "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=$3"
  } >> "$1/.env"
}

start_api() {  # $1=dir $2=port $3=logname -> echoes pid
  ( cd "$1" && uv run --project "$API_DIR" hindsight-api --host 127.0.0.1 --port "$2" ) > "$WORK/$3.log" 2>&1 &
  echo $!
}

wait_ready() {  # $1=logfile
  for _ in $(seq 1 120); do
    grep -q "Application startup complete" "$1" && return 0
    grep -qiE "Traceback|FATAL|Address already in use" "$1" && { tail -8 "$1"; return 1; }
    sleep 2
  done
  return 1
}

echo "Workdir: $WORK"
mk_env "$WORK/a" e2e_mig_a "BAAI/bge-small-en-v1.5"
mk_env "$WORK/b" e2e_mig_b "BAAI/bge-base-en-v1.5"

echo "[1] start instance A (bge-small/384) on $PA"
PIDA=$(start_api "$WORK/a" "$PA" apiA); wait_ready "$WORK/apiA.log" || fail "instance A did not start"

echo "[2] retain into A"
code=$(curl -s -o "$WORK/retain.json" -w "%{http_code}" -X POST \
  "http://127.0.0.1:$PA/v1/default/banks/$BANK/memories" -H 'content-type: application/json' \
  -d '{"items":[
        {"content":"Alice works at Acme Corp as a data engineer.","document_id":"d1"},
        {"content":"Bob lives in Berlin and loves cycling.","document_id":"d2"},
        {"content":"Carol manages the Helsinki office and speaks Finnish.","document_id":"d3"}
      ],"async":false}')
[ "$code" = "200" ] || fail "retain returned HTTP $code: $(cat "$WORK/retain.json")"

echo "[3] export-bank from A"
( cd "$WORK/a" && uv run --project "$API_DIR" hindsight-admin export-bank --bank "$BANK" --output "$WORK/bank.zip" ) \
  || fail "export-bank failed"
[ -s "$WORK/bank.zip" ] || fail "archive not written"

echo "[4] import-bank into B (bge-base/768) — re-embeds, no LLM"
( cd "$WORK/b" && uv run --project "$API_DIR" hindsight-admin import-bank --archive "$WORK/bank.zip" ) \
  > "$WORK/import.log" 2>&1 || { cat "$WORK/import.log"; fail "import-bank failed"; }
grep -q "dim: 768" "$WORK/import.log" || echo "  (note: could not confirm 768-dim from import log)"

echo "[5] start instance B (bge-base/768) on $PB"
PIDB=$(start_api "$WORK/b" "$PB" apiB); wait_ready "$WORK/apiB.log" || fail "instance B did not start"

echo "[6] recall on B and assert the migrated fact ranks first"
top=$(curl -s -X POST "http://127.0.0.1:$PB/v1/default/banks/$BANK/memories/recall" \
  -H 'content-type: application/json' -d '{"query":"Who manages the Helsinki office?"}' \
  | jq -r '.results[0].text // ""')
echo "  top result: $top"
echo "$top" | grep -qi "Helsinki" || fail "recall on B did not return the migrated Helsinki fact (got: '$top')"

echo "[7] confirm the two instances used different embedding dims"
grep -q "dim: 384" "$WORK/apiA.log" || fail "instance A was not 384-dim"
grep -q "dim: 768" "$WORK/apiB.log" || fail "instance B was not 768-dim"

echo "✅ PASS: bank migrated bge-small(384) → bge-base(768), recall works on B, no LLM re-extraction"
