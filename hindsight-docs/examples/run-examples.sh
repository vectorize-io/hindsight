#!/bin/bash
# Run all documentation example scripts
# Usage: ./examples/run-examples.sh [python|node|cli|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed=0
failed=0
skipped=0

run_python_examples() {
    echo -e "${YELLOW}Running Python examples...${NC}"
    for f in "$SCRIPT_DIR"/api/*.py; do
        if [ -f "$f" ]; then
            echo -n "  $(basename "$f"): "
            if python "$f" 2>&1; then
                echo -e "${GREEN}PASSED${NC}"
                ((passed++))
            else
                echo -e "${RED}FAILED${NC}"
                ((failed++))
            fi
        fi
    done
}

run_node_examples() {
    echo -e "${YELLOW}Running Node.js examples...${NC}"
    for f in "$SCRIPT_DIR"/api/*.mjs; do
        if [ -f "$f" ]; then
            echo -n "  $(basename "$f"): "
            if node "$f" 2>&1; then
                echo -e "${GREEN}PASSED${NC}"
                ((passed++))
            else
                echo -e "${RED}FAILED${NC}"
                ((failed++))
            fi
        fi
    done
}

run_cli_examples() {
    echo -e "${YELLOW}Running CLI examples...${NC}"

    # Check if hindsight CLI is available
    if ! command -v hindsight &> /dev/null; then
        echo -e "  ${YELLOW}SKIPPED (hindsight CLI not installed)${NC}"
        for f in "$SCRIPT_DIR"/api/*.sh; do
            if [ -f "$f" ]; then
                ((skipped++))
            fi
        done
        return
    fi

    for f in "$SCRIPT_DIR"/api/*.sh; do
        if [ -f "$f" ]; then
            echo -n "  $(basename "$f"): "
            if bash "$f" 2>&1; then
                echo -e "${GREEN}PASSED${NC}"
                ((passed++))
            else
                echo -e "${RED}FAILED${NC}"
                ((failed++))
            fi
        fi
    done
}

# Wait for server to be ready
wait_for_server() {
    echo "Waiting for Hindsight server at $HINDSIGHT_URL..."
    for i in {1..30}; do
        if curl -s "$HINDSIGHT_URL/health" > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        sleep 1
    done
    echo "Server not available after 30 seconds"
    return 1
}

# Main
case "${1:-all}" in
    python)
        wait_for_server
        run_python_examples
        ;;
    node)
        wait_for_server
        run_node_examples
        ;;
    cli)
        wait_for_server
        run_cli_examples
        ;;
    all)
        wait_for_server
        run_python_examples
        echo ""
        run_node_examples
        echo ""
        run_cli_examples
        ;;
    *)
        echo "Usage: $0 [python|node|cli|all]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo -e "Results: ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}, ${YELLOW}$skipped skipped${NC}"
echo "========================================"

if [ $failed -gt 0 ]; then
    exit 1
fi
