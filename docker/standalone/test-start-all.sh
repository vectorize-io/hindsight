#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HINDSIGHT_START_ALL_SOURCE_ONLY=true
source "$SCRIPT_DIR/start-all.sh"
unset HINDSIGHT_START_ALL_SOURCE_ONLY

TMP_DIR="$(mktemp -d)"
HTTP_SERVER_PID=""

cleanup() {
    if [ -n "$HTTP_SERVER_PID" ]; then
        kill "$HTTP_SERVER_PID" 2>/dev/null || true
        wait "$HTTP_SERVER_PID" 2>/dev/null || true
    fi
    chmod -R u+rwx "$TMP_DIR" 2>/dev/null || true
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

assert_contains() {
    local output="$1"
    local expected="$2"

    if [[ "$output" != *"$expected"* ]]; then
        echo "Expected output to contain: $expected"
        echo "Actual output:"
        echo "$output"
        exit 1
    fi
}

assert_not_contains() {
    local output="$1"
    local unexpected="$2"

    if [[ "$output" == *"$unexpected"* ]]; then
        echo "Expected output not to contain: $unexpected"
        echo "Actual output:"
        echo "$output"
        exit 1
    fi
}

assert_empty() {
    local output="$1"

    if [ -n "$output" ]; then
        echo "Expected no output, got:"
        echo "$output"
        exit 1
    fi
}

# =============================================================================
# http_probe
# =============================================================================
HTTP_PORT_FILE="$TMP_DIR/http-port"
if python3 -c 'import sys; sys.exit(0)' >/dev/null 2>&1; then
    TEST_PYTHON_BIN="$(command -v python3)"
elif python -c 'import sys; sys.exit(0)' >/dev/null 2>&1; then
    TEST_PYTHON_BIN="$(command -v python)"
else
    echo "A working Python interpreter is required for HTTP probe tests"
    exit 1
fi

"$TEST_PYTHON_BIN" - "$HTTP_PORT_FILE" <<'PY' &
import http.server
import pathlib
import sys


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(204 if self.path == "/ok" else 404)
        self.end_headers()

    def log_message(self, *_args):
        pass


server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
pathlib.Path(sys.argv[1]).write_text(str(server.server_port), encoding="ascii")
for _ in range(4):
    server.handle_request()
PY
HTTP_SERVER_PID=$!

for _ in $(seq 1 50); do
    [ -s "$HTTP_PORT_FILE" ] && break
    sleep 0.1
done
if [ ! -s "$HTTP_PORT_FILE" ]; then
    echo "HTTP probe test server did not start"
    exit 1
fi

HTTP_TEST_URL="http://127.0.0.1:$(cat "$HTTP_PORT_FILE")"
PYTHON_BIN="$TEST_PYTHON_BIN"
WGET_BIN="$(command -v wget)"

(
    PATH="$TMP_DIR/empty-path"
    python3() { "$PYTHON_BIN" "$@"; }
    http_probe "$HTTP_TEST_URL/ok" 2
    if http_probe "$HTTP_TEST_URL/missing" 2; then
        echo "Python HTTP probe should fail on a 404 response"
        exit 1
    fi
)

(
    PATH="$TMP_DIR/empty-path"
    wget() { "$WGET_BIN" "$@"; }
    http_probe "$HTTP_TEST_URL/ok" 2
    if http_probe "$HTTP_TEST_URL/missing" 2; then
        echo "wget HTTP probe should fail on a 404 response"
        exit 1
    fi
)

wait "$HTTP_SERVER_PID"
HTTP_SERVER_PID=""
echo "start-all HTTP probe checks passed"

mkdir -p "$TMP_DIR/empty"
assert_empty "$(check_pg0_data_integrity "$TMP_DIR/empty")"

mkdir -p "$TMP_DIR/direct"
touch "$TMP_DIR/direct/PG_VERSION"
direct_output="$(check_pg0_data_integrity "$TMP_DIR/direct")"
assert_contains "$direct_output" "Existing pg0 data directory detected"
assert_not_contains "$direct_output" "WARNING"

mkdir -p "$TMP_DIR/legacy/instance"
touch "$TMP_DIR/legacy/instance/PG_VERSION"
legacy_output="$(check_pg0_data_integrity "$TMP_DIR/legacy")"
assert_contains "$legacy_output" "Existing pg0 data directory detected"
assert_not_contains "$legacy_output" "WARNING"

mkdir -p "$TMP_DIR/nested/instances/hindsight/data"
touch "$TMP_DIR/nested/instances/hindsight/data/PG_VERSION"
nested_output="$(check_pg0_data_integrity "$TMP_DIR/nested")"
assert_contains "$nested_output" "Existing pg0 data directory detected"
assert_not_contains "$nested_output" "WARNING"

mkdir -p "$TMP_DIR/nonempty/instances/hindsight"
touch "$TMP_DIR/nonempty/instances/hindsight/instance.json"
nonempty_output="$(check_pg0_data_integrity "$TMP_DIR/nonempty")"
assert_contains "$nonempty_output" "WARNING: pg0 data directory exists"

echo "start-all pg0 integrity checks passed"

# =============================================================================
# resolve_api_startup_wait_seconds (#3733)
# =============================================================================
assert_equals() {
    local actual="$1"
    local expected="$2"

    if [ "$actual" != "$expected" ]; then
        echo "Expected: $expected"
        echo "Actual:   $actual"
        exit 1
    fi
}

# Neither knob set: the wrapper default.
assert_equals "$(HINDSIGHT_API_STARTUP_WAIT_SECONDS= HINDSIGHT_API_MODEL_INIT_TIMEOUT= resolve_api_startup_wait_seconds)" "300"

# The documented knob raised: the wrapper waits at least that long, so raising
# it actually takes effect instead of being cut short at the default.
assert_equals "$(HINDSIGHT_API_MODEL_INIT_TIMEOUT=7200 resolve_api_startup_wait_seconds)" "7230"

# Floats are accepted — the API parses its cap as one.
assert_equals "$(HINDSIGHT_API_MODEL_INIT_TIMEOUT=7200.0 resolve_api_startup_wait_seconds)" "7230"

# A shorter cap never shortens the wrapper wait.
assert_equals "$(HINDSIGHT_API_MODEL_INIT_TIMEOUT=60 resolve_api_startup_wait_seconds)" "300"

# Garbage falls back to the default rather than breaking startup.
assert_equals "$(HINDSIGHT_API_MODEL_INIT_TIMEOUT=abc resolve_api_startup_wait_seconds)" "300"

# An explicit wrapper setting always wins.
assert_equals "$(HINDSIGHT_API_STARTUP_WAIT_SECONDS=45 HINDSIGHT_API_MODEL_INIT_TIMEOUT=7200 resolve_api_startup_wait_seconds)" "45"

echo "start-all API startup wait checks passed"

# =============================================================================
# check_pg0_writable (#1483)
# These rely on filesystem permissions, which root bypasses; skip under root.
# =============================================================================
if [ "$(id -u)" != "0" ]; then
    # Writable directory: returns 0, prints nothing, leaves no artifact behind.
    mkdir -p "$TMP_DIR/writable"
    writable_output="$(check_pg0_writable "$TMP_DIR/writable")"
    assert_empty "$writable_output"
    if [ -e "$TMP_DIR/writable/.hindsight-write-test" ]; then
        echo "check_pg0_writable left its write-test file behind"
        exit 1
    fi

    # Non-writable directory: returns 1 with actionable guidance.
    mkdir -p "$TMP_DIR/readonly"
    chmod 000 "$TMP_DIR/readonly"
    set +e
    readonly_output="$(check_pg0_writable "$TMP_DIR/readonly" 2>&1)"
    readonly_rc=$?
    set -e
    chmod 755 "$TMP_DIR/readonly"
    if [ "$readonly_rc" -eq 0 ]; then
        echo "check_pg0_writable should fail on a non-writable directory"
        exit 1
    fi
    assert_contains "$readonly_output" "not writable"
    assert_contains "$readonly_output" "hindsight-data:/home/hindsight/.pg0"
    assert_contains "$readonly_output" "--user"

    # External database configured: skip the check regardless of dir perms.
    mkdir -p "$TMP_DIR/extdb"
    chmod 000 "$TMP_DIR/extdb"
    set +e
    HINDSIGHT_API_DATABASE_URL="postgres://x" check_pg0_writable "$TMP_DIR/extdb" >/dev/null 2>&1
    extdb_rc=$?
    set -e
    chmod 755 "$TMP_DIR/extdb"
    if [ "$extdb_rc" -ne 0 ]; then
        echo "check_pg0_writable should skip when an external database is configured"
        exit 1
    fi

    echo "start-all pg0 writability checks passed"
else
    echo "⚠️  Running as root; skipping pg0 writability checks (permissions are bypassed)."
fi
