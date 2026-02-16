#!/bin/bash
set +e  # Don't exit on errors - we want to collect all failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLES_DIR="$PROJECT_ROOT/hindsight-docs/examples/api"
LOG_DIR="/tmp/doc-example-logs"

mkdir -p "$LOG_DIR"

TOTAL_PASSED=0
TOTAL_FAILED=0
FAILED_EXAMPLES=()

# Parse language filter from command line
LANGUAGE_FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --lang)
            LANGUAGE_FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--lang <python|node|cli|go>]"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "Running Documentation Examples"
if [ -n "$LANGUAGE_FILTER" ]; then
    echo "Language filter: $LANGUAGE_FILTER"
fi
echo "======================================"
echo ""

# Function to run a single example
run_example() {
    local file="$1"
    local runner="$2"
    local workdir="${3:-$PROJECT_ROOT}"

    local basename=$(basename "$file")
    local logfile="$LOG_DIR/$basename.log"

    echo -n "Running $basename... "

    pushd "$workdir" > /dev/null 2>&1
    if $runner "$file" > "$logfile" 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        TOTAL_PASSED=$((TOTAL_PASSED + 1))
        rm -f "$logfile"  # Clean up successful test logs
        popd > /dev/null 2>&1
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        FAILED_EXAMPLES+=("$basename:$logfile")
        popd > /dev/null 2>&1
        return 1
    fi
}

# Run Python examples
if [ -z "$LANGUAGE_FILTER" ] || [ "$LANGUAGE_FILTER" = "python" ]; then
    echo "======================================"
    echo "Python Examples"
    echo "======================================"
    cd "$PROJECT_ROOT/hindsight-clients/python"
    for f in "$EXAMPLES_DIR"/*.py; do
        [ -e "$f" ] || continue  # Skip if no files match
        run_example "$f" "uv run python" "$PROJECT_ROOT/hindsight-clients/python"
    done
    echo ""
fi

# Run Node.js examples
if [ -z "$LANGUAGE_FILTER" ] || [ "$LANGUAGE_FILTER" = "node" ]; then
    echo "======================================"
    echo "Node.js Examples"
    echo "======================================"
    cd "$PROJECT_ROOT"
    for f in "$EXAMPLES_DIR"/*.mjs; do
        [ -e "$f" ] || continue  # Skip if no files match
        run_example "$f" "node" "$PROJECT_ROOT"
    done
    echo ""
fi

# Run CLI examples
if [ -z "$LANGUAGE_FILTER" ] || [ "$LANGUAGE_FILTER" = "cli" ]; then
    echo "======================================"
    echo "CLI Examples"
    echo "======================================"
    cd "$PROJECT_ROOT"
    for f in "$EXAMPLES_DIR"/*.sh; do
        [ -e "$f" ] || continue  # Skip if no files match
        run_example "$f" "bash" "$PROJECT_ROOT"
    done
    echo ""
fi

# Run Go examples
if [ -z "$LANGUAGE_FILTER" ] || [ "$LANGUAGE_FILTER" = "go" ]; then
    echo "======================================"
    echo "Go Examples"
    echo "======================================"
    cd "$PROJECT_ROOT/hindsight-clients/go"
    for f in "$EXAMPLES_DIR"/*.go; do
        [ -e "$f" ] || continue  # Skip if no files match
        run_example "$f" "go run" "$PROJECT_ROOT/hindsight-clients/go"
    done
    echo ""
fi

# Print summary
echo "======================================"
echo "Summary"
echo "======================================"
echo -e "${GREEN}Passed: $TOTAL_PASSED${NC}"
echo -e "${RED}Failed: $TOTAL_FAILED${NC}"
echo ""

# If there are failures, show the logs
if [ $TOTAL_FAILED -gt 0 ]; then
    echo "======================================"
    echo "Failed Example Logs"
    echo "======================================"
    for entry in "${FAILED_EXAMPLES[@]}"; do
        IFS=':' read -r name logfile <<< "$entry"
        echo ""
        echo -e "${YELLOW}=== $name ===${NC}"
        cat "$logfile"
    done
    echo ""
    echo -e "${RED}$TOTAL_FAILED example(s) failed${NC}"
    exit 1
fi

echo -e "${GREEN}All examples passed!${NC}"
exit 0
