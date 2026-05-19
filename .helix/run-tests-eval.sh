#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEST_USER="${HINDSIGHT_TEST_USER:-hindsight}"

# Helix CI runs the container as root so git can write a bind-mounted .git.
# pg0 initdb refuses root, so re-exec this script as the test user when needed.
if [[ "$(id -u)" -eq 0 ]]; then
  exec su "$TEST_USER" -s /bin/bash "$0" "$@"
fi

# Start embedded PostgreSQL when no external database URL is provided.
# pg0's Python API can report running=false inside Docker even when Postgres is
# up, so start via the pg0 CLI and export the connection URI for pytest.
if [[ -z "${HINDSIGHT_API_DATABASE_URL:-}" ]]; then
  PG0_BIN="$(uv run --directory hindsight-api-slim python -c 'from pg0 import _find_pg0; print(_find_pg0())')"
  PG0_NAME="${HINDSIGHT_TEST_PG0_INSTANCE:-hindsight-test}"
  PG0_PORT="${HINDSIGHT_TEST_PG_PORT:-5556}"
  PG0_USER="${HINDSIGHT_TEST_PG_USER:-hindsight}"
  PG0_PASS="${HINDSIGHT_TEST_PG_PASS:-hindsight}"
  PG0_DB="${HINDSIGHT_TEST_PG_DATABASE:-hindsight}"

  pg0_start_out="$("$PG0_BIN" start \
    --name "$PG0_NAME" \
    --port "$PG0_PORT" \
    --username "$PG0_USER" \
    --password "$PG0_PASS" \
    --database "$PG0_DB" 2>&1)" || true

  uri="$(echo "$pg0_start_out" | grep -m1 'Connection URI:' | sed 's/.*Connection URI: //' || true)"
  if [[ -z "$uri" ]]; then
    uri="postgresql://${PG0_USER}:${PG0_PASS}@127.0.0.1:${PG0_PORT}/${PG0_DB}"
  fi
  export HINDSIGHT_API_DATABASE_URL="$uri"
fi

PYTEST_ARGS=(
  -m "not oracle"
  --timeout 300
  -n 4
  --dist loadgroup
  -v
)

run_pytest() {
  uv run --directory hindsight-api-slim pytest "$@"
}

# Helix CI runs this script with no args after git checkout to verify the
# container/environment — not to execute the full suite (many retain/recall
# tests need a real LLM or patched mock responses).
if [[ $# -eq 0 || -z "${1:-}" ]]; then
  run_pytest \
    tests/test_async_batch_retain.py::test_duplicate_document_ids_rejected_async \
    tests/test_base_path.py::test_base_path_health_endpoint \
    -m "not oracle" \
    --timeout 300 \
    -n 0 \
    -v
  exit 0
fi

IFS=',' read -ra TEST_FILES <<< "$1"
normalized=()
for f in "${TEST_FILES[@]}"; do
  f="${f#hindsight-api-slim/}"
  normalized+=("$f")
done
run_pytest "${normalized[@]}" "${PYTEST_ARGS[@]}"
