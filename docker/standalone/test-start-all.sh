#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HINDSIGHT_START_ALL_SOURCE_ONLY=true
source "$SCRIPT_DIR/start-all.sh"
unset HINDSIGHT_START_ALL_SOURCE_ONLY

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

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
