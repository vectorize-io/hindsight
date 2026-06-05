#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HINDSIGHT_START_ALL_SOURCE_ONLY=true
source "$SCRIPT_DIR/start-all.sh"
unset HINDSIGHT_START_ALL_SOURCE_ONLY

TMP_DIR="$(mktemp -d)"
trap 'chmod -R u+rwx "$TMP_DIR" 2>/dev/null || true; rm -rf "$TMP_DIR"' EXIT

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
