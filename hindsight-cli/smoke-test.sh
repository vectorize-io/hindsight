#!/bin/bash
# CLI smoke test - verifies basic CLI functionality against a running API server
#
# Prerequisites:
#   - hindsight CLI must be in PATH or HINDSIGHT_CLI env var set
#   - API server must be running at HINDSIGHT_API_URL (default: http://localhost:8888)
#
# Usage:
#   ./hindsight-cli/smoke-test.sh
#   HINDSIGHT_CLI=/path/to/hindsight ./hindsight-cli/smoke-test.sh

set -e

# Configuration
HINDSIGHT_CLI="${HINDSIGHT_CLI:-hindsight}"
export HINDSIGHT_API_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"
TEST_BANK="cli-smoke-test-$(date +%s)"
TEST_KB="$TEST_BANK-kb"
KB_DIR="$(mktemp -d)"

echo "=== Hindsight CLI Smoke Test ==="
echo "CLI: $HINDSIGHT_CLI"
echo "API URL: $HINDSIGHT_API_URL"
echo "Test bank: $TEST_BANK"
echo ""

# Helper function
run_test() {
    local name="$1"
    shift
    echo -n "Testing: $name... "
    if "$@" > /tmp/cli-test-output.txt 2>&1; then
        echo "OK"
        return 0
    else
        echo "FAILED"
        echo "  Command: $*"
        echo "  Output:"
        cat /tmp/cli-test-output.txt | sed 's/^/    /'
        return 1
    fi
}

run_test_output() {
    local name="$1"
    local expected="$2"
    shift 2
    echo -n "Testing: $name... "
    if "$@" > /tmp/cli-test-output.txt 2>&1; then
        if grep -qi "$expected" /tmp/cli-test-output.txt; then
            echo "OK"
            return 0
        else
            echo "FAILED (expected '$expected' not found)"
            echo "  Command: $*"
            echo "  Output:"
            cat /tmp/cli-test-output.txt | sed 's/^/    /'
            return 1
        fi
    else
        echo "FAILED"
        echo "  Command: $*"
        echo "  Output:"
        cat /tmp/cli-test-output.txt | sed 's/^/    /'
        return 1
    fi
}

cleanup() {
    echo ""
    echo "Cleaning up test bank..."
    "$HINDSIGHT_CLI" bank delete "$TEST_BANK" -y 2>/dev/null || true
    "$HINDSIGHT_CLI" kb delete "$TEST_KB" -y 2>/dev/null || true
    rm -rf "$KB_DIR"
}
trap cleanup EXIT

FAILED=0

# Test 1: Version
run_test "version" "$HINDSIGHT_CLI" --version || FAILED=1

# Test 2: Help
run_test "help" "$HINDSIGHT_CLI" --help || FAILED=1

# Test 3: Configure help
run_test "configure help" "$HINDSIGHT_CLI" configure --help || FAILED=1

# Test 4: List banks (JSON output)
run_test "list banks" "$HINDSIGHT_CLI" bank list -o json || FAILED=1

# Test 5: Set bank name (creates the bank)
run_test "set bank name" "$HINDSIGHT_CLI" bank name "$TEST_BANK" "CLI Smoke Test Bank" || FAILED=1

# Test 6: Get bank disposition
run_test_output "get bank disposition" "CLI Smoke Test Bank" "$HINDSIGHT_CLI" bank disposition "$TEST_BANK" || FAILED=1

# Test 7: Retain memory
run_test "retain memory" "$HINDSIGHT_CLI" memory retain "$TEST_BANK" "Alice is a software engineer who loves Rust programming" || FAILED=1

# Test 8: Retain more memories
run_test "retain more memories" "$HINDSIGHT_CLI" memory retain "$TEST_BANK" "Bob is Alice's colleague who prefers Python" || FAILED=1

# Test 9: Recall memories
run_test_output "recall memories" "Alice" "$HINDSIGHT_CLI" memory recall "$TEST_BANK" "Who is Alice?" || FAILED=1

# Test 10: Reflect on memories
run_test_output "reflect" "Alice" "$HINDSIGHT_CLI" memory reflect "$TEST_BANK" "What do you know about Alice?" || FAILED=1

# Test 11: Get bank stats
run_test "bank stats" "$HINDSIGHT_CLI" bank stats "$TEST_BANK" || FAILED=1

# Test 12: List entities
run_test "list entities" "$HINDSIGHT_CLI" entity list "$TEST_BANK" || FAILED=1

# Test 13: List documents
run_test "list documents" "$HINDSIGHT_CLI" document list "$TEST_BANK" || FAILED=1

# Test 14: Clear memories
run_test "clear memories" "$HINDSIGHT_CLI" memory clear "$TEST_BANK" || FAILED=1

# Test 15: List operations
run_test "list operations" "$HINDSIGHT_CLI" operation list "$TEST_BANK" || FAILED=1

# --- Coverage-critical commands (added to ensure CLI exercises every endpoint) ---

# Test: Set disposition directly (PUT /profile)
run_test "bank set-disposition" "$HINDSIGHT_CLI" bank set-disposition "$TEST_BANK" \
    --skepticism 3 --literalism 3 --empathy 3 || FAILED=1

# Test: Recover consolidation (no-op when nothing stalled, but exercises the endpoint)
run_test "bank consolidation-recover" "$HINDSIGHT_CLI" bank consolidation-recover "$TEST_BANK" || FAILED=1

# Test: Bank template schema
run_test "bank template-schema" "$HINDSIGHT_CLI" bank template-schema -o json || FAILED=1

# Test: Export bank template
run_test "bank export-template" "$HINDSIGHT_CLI" bank export-template "$TEST_BANK" -o json || FAILED=1

# Test: Audit log list + stats
run_test "audit list" "$HINDSIGHT_CLI" audit list "$TEST_BANK" -o json || FAILED=1
run_test "audit stats" "$HINDSIGHT_CLI" audit stats "$TEST_BANK" -o json || FAILED=1

# Test: Webhook lifecycle (list / create / update / deliveries / delete)
run_test "webhook list (empty)" "$HINDSIGHT_CLI" webhook list "$TEST_BANK" -o json || FAILED=1

WEBHOOK_OUT=$("$HINDSIGHT_CLI" webhook create "$TEST_BANK" https://example.invalid/hook -o json 2>/tmp/cli-test-output.txt || true)
if echo "$WEBHOOK_OUT" | grep -q '"id"'; then
    echo "Testing: webhook create... OK"
    WEBHOOK_ID=$(echo "$WEBHOOK_OUT" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)
    run_test "webhook update" "$HINDSIGHT_CLI" webhook update "$TEST_BANK" "$WEBHOOK_ID" --enabled false || FAILED=1
    run_test "webhook deliveries" "$HINDSIGHT_CLI" webhook deliveries "$TEST_BANK" "$WEBHOOK_ID" -o json || FAILED=1
    run_test "webhook delete" "$HINDSIGHT_CLI" webhook delete "$TEST_BANK" "$WEBHOOK_ID" -y || FAILED=1
else
    echo "Testing: webhook create... FAILED"
    cat /tmp/cli-test-output.txt | sed 's/^/    /'
    FAILED=1
fi

# Test: Knowledge bank lifecycle (create / ingest / search / documents / config / link / delete)
SAMPLE_PDF="$(dirname "$0")/../hindsight-docs/examples/api/sample.pdf"
mkdir -p "$KB_DIR/manuals"
cp "$SAMPLE_PDF" "$KB_DIR/manuals/sample.pdf"
printf '# Warranty\n\nThe X200 warranty lasts two years.\n' > "$KB_DIR/manuals/warranty.md"
run_test "kb create" "$HINDSIGHT_CLI" kb create "$TEST_KB" || FAILED=1
run_test_output "kb list" "$TEST_KB" "$HINDSIGHT_CLI" kb list -q "$TEST_KB" -o json || FAILED=1
run_test_output "kb ingest folder --wait" '"ready"' "$HINDSIGHT_CLI" kb ingest "$TEST_KB" "$KB_DIR/manuals" --tags x200 --wait -o json || FAILED=1
run_test "kb ingest-text" "$HINDSIGHT_CLI" kb ingest-text "$TEST_KB" "Return an X200 within 30 days." --document-id returns --tags policy || FAILED=1
run_test_output "kb search" '"returns"' "$HINDSIGHT_CLI" kb search "$TEST_KB" "return an X200 within 30 days" -o json || FAILED=1
run_test_output "kb search --tags" 'warranty' "$HINDSIGHT_CLI" kb search "$TEST_KB" "warranty" --tags x200 --tags-match all_strict --trace -o json || FAILED=1
run_test_output "kb get" '"document_count": 3' "$HINDSIGHT_CLI" kb get "$TEST_KB" -o json || FAILED=1
run_test_output "kb list-documents" '"returns"' "$HINDSIGHT_CLI" kb list-documents "$TEST_KB" --tags policy -o json || FAILED=1
run_test_output "kb get-document" '"passages"' "$HINDSIGHT_CLI" kb get-document "$TEST_KB" returns -o json || FAILED=1
run_test_output "kb list-tags" '"x200"' "$HINDSIGHT_CLI" kb list-tags "$TEST_KB" -o json || FAILED=1
PDF_ID=$("$HINDSIGHT_CLI" kb list-documents "$TEST_KB" -q sample.pdf -o json | sed -n 's/.*"document_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)
run_test "kb download" "$HINDSIGHT_CLI" kb download "$TEST_KB" "$PDF_ID" --out "$KB_DIR/download.pdf" || FAILED=1
run_test "kb download is the original" cmp "$SAMPLE_PDF" "$KB_DIR/download.pdf" || FAILED=1
run_test "kb passage-media" "$HINDSIGHT_CLI" kb passage-media "$TEST_KB" "$PDF_ID" 0 --out "$KB_DIR/page.png" || FAILED=1
run_test_output "kb set-config" '"text_passage_tokens": 256' "$HINDSIGHT_CLI" kb set-config "$TEST_KB" --set text_passage_tokens=256 -o json || FAILED=1
run_test_output "kb config --overrides-only" '256' "$HINDSIGHT_CLI" kb config "$TEST_KB" --overrides-only -o json || FAILED=1
run_test "kb reset-config" "$HINDSIGHT_CLI" kb reset-config "$TEST_KB" -y || FAILED=1
run_test_output "bank set-config --knowledge-banks" "$TEST_KB" "$HINDSIGHT_CLI" bank set-config "$TEST_BANK" --knowledge-banks "$TEST_KB" -o json || FAILED=1
run_test "kb delete-document" "$HINDSIGHT_CLI" kb delete-document "$TEST_KB" returns || FAILED=1
run_test "kb delete" "$HINDSIGHT_CLI" kb delete "$TEST_KB" -y || FAILED=1

# Test 16: Delete bank
run_test "delete bank" "$HINDSIGHT_CLI" bank delete "$TEST_BANK" -y || FAILED=1

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== All smoke tests passed! ==="
    exit 0
else
    echo "=== Some smoke tests failed ==="
    exit 1
fi
