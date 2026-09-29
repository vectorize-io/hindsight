#!/bin/bash
#
# Slim variant smoke test: retain + recall
#
# Verifies that a Hindsight API endpoint can store and retrieve memories.
# Used by both Docker slim and pip slim CI jobs.
#
# Usage:
#   ./scripts/smoke-test-slim.sh [base_url]
#
# Arguments:
#   base_url  - API base URL (default: http://localhost:8888)
#
# Exit codes:
#   0 - Success
#   1 - Failure
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

BASE_URL="${1:-http://localhost:8888}"
BANK_ID="smoke-test-$$"

echo "Running retain/recall smoke test against: $BASE_URL"
echo "Bank: $BANK_ID"

# Retain
echo ""
echo "--- Retain ---"
RETAIN_RESPONSE=$(curl -sf -X POST "$BASE_URL/v1/default/banks/$BANK_ID/memories" \
  -H "Content-Type: application/json" \
  -d '{"items": [{"content": "Alice is a software engineer who loves Python and distributed systems."}]}')
echo "$RETAIN_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RETAIN_RESPONSE"

SUCCESS=$(echo "$RETAIN_RESPONSE" | python3 -c "import sys, json; d = json.load(sys.stdin); print(d.get('success', False))" 2>/dev/null || echo "False")
if [ "$SUCCESS" != "True" ]; then
  echo -e "${RED}FAIL: retain did not return success=true${NC}"
  exit 1
fi
echo "Retain: OK"

# Recall
echo ""
echo "--- Recall ---"
RECALL_RESPONSE=$(curl -sf -X POST "$BASE_URL/v1/default/banks/$BANK_ID/memories/recall" \
  -H "Content-Type: application/json" \
  -d '{"query": "What does Alice do?"}')
echo "$RECALL_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RECALL_RESPONSE"

RESULTS_COUNT=$(echo "$RECALL_RESPONSE" | python3 -c "import sys, json; d = json.load(sys.stdin); print(len(d.get('results', [])))" 2>/dev/null || echo "0")
if [ -z "$RESULTS_COUNT" ] || [ "$RESULTS_COUNT" -eq 0 ]; then
  echo -e "${RED}FAIL: recall returned no results${NC}"
  exit 1
fi
echo "Recall: OK ($RESULTS_COUNT results)"

echo ""
echo -e "${GREEN}PASS: retain/recall smoke test${NC}"
