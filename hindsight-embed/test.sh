#!/bin/bash
#
# Smoke test for hindsight-embed CLI with daemon mode
# Tests retain and recall operations via the background daemon
# Verifies daemon lifecycle, memory retention, and recall functionality
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$(cd "$SCRIPT_DIR/../hindsight-api" && pwd)"

echo "=== Hindsight Embed Smoke Test (Daemon Mode) ==="

# Check required environment (load from config if not set)
if [ -f ~/.hindsight/config.env ]; then
    source ~/.hindsight/config.env
fi

if [ -z "$HINDSIGHT_API_LLM_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: HINDSIGHT_API_LLM_API_KEY or OPENAI_API_KEY is required"
    exit 1
fi

# Use a unique bank ID for this test run
BANK_ID="test-$$-$(date +%s)"
echo "Using bank ID: $BANK_ID"
echo "Script dir: $SCRIPT_DIR"
echo "API dir: $API_DIR"

# Show environment info for debugging
echo ""
echo "Environment:"
echo "  HINDSIGHT_API_LLM_PROVIDER: ${HINDSIGHT_API_LLM_PROVIDER:-not set}"
echo "  HINDSIGHT_API_LLM_MODEL: ${HINDSIGHT_API_LLM_MODEL:-not set}"
echo "  HINDSIGHT_API_LLM_API_KEY: ${HINDSIGHT_API_LLM_API_KEY:+set (hidden)}"
echo "  Python: $(python3 --version 2>&1)"
echo "  uv: $(uv --version 2>&1)"

# Stop any existing daemon
echo ""
echo "Stopping any existing daemon..."
uv run --project "$SCRIPT_DIR" hindsight-embed daemon stop 2>/dev/null || true
sleep 1

# Test 1: Retain (this should start the daemon)
echo ""
echo "Test 1: Retaining a memory (first call - daemon will start)..."
START_TIME=$(python3 -c "import time; print(time.time())")
set +e  # Temporarily disable exit on error to capture output
OUTPUT=$(uv run --project "$SCRIPT_DIR" hindsight-embed memory retain "$BANK_ID" "The user's favorite color is blue" 2>&1)
EXIT_CODE=$?
set -e
END_TIME=$(python3 -c "import time; print(time.time())")
DURATION=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
echo "$OUTPUT"
echo "Duration: ${DURATION}s"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -ne 0 ]; then
    echo "FAIL: Command exited with code $EXIT_CODE"
    echo ""
    echo "Checking daemon logs..."
    if [ -f ~/.hindsight/daemon.log ]; then
        echo "=== daemon.log ==="
        tail -50 ~/.hindsight/daemon.log
    else
        echo "No daemon.log found"
    fi
    if [ -f ~/.hindsight/daemon.stderr ]; then
        echo ""
        echo "=== daemon.stderr ==="
        cat ~/.hindsight/daemon.stderr
    else
        echo "No daemon.stderr found"
    fi
    echo ""
    echo "Checking for hindsight-api..."
    which hindsight-api 2>/dev/null || echo "hindsight-api not in PATH"
    which uvx 2>/dev/null || echo "uvx not in PATH"
    which uv 2>/dev/null || echo "uv not in PATH"
    exit 1
fi

if ! echo "$OUTPUT" | grep -qi "retained"; then
    echo "FAIL: Expected 'retained' in output"
    exit 1
fi
echo "PASS: Memory retained successfully"

# Test 2: Recall (daemon already running - should be faster)
echo ""
echo "Test 2: Recalling memories (daemon already running)..."
START_TIME=$(python3 -c "import time; print(time.time())")
set +e
OUTPUT=$(uv run --project "$SCRIPT_DIR" hindsight-embed memory recall "$BANK_ID" "What is the user's favorite color?" 2>&1)
EXIT_CODE=$?
set -e
END_TIME=$(python3 -c "import time; print(time.time())")
DURATION=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
echo "$OUTPUT"
echo "Duration: ${DURATION}s"
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "FAIL: Command exited with code $EXIT_CODE"
    exit 1
fi
if ! echo "$OUTPUT" | grep -qi "blue"; then
    echo "FAIL: Expected 'blue' in recall output"
    exit 1
fi
echo "PASS: Memory recalled successfully"

# Test 3: Retain with context (daemon should still be running)
echo ""
echo "Test 3: Retaining memory with context..."
START_TIME=$(python3 -c "import time; print(time.time())")
set +e
OUTPUT=$(uv run --project "$SCRIPT_DIR" hindsight-embed memory retain "$BANK_ID" "User prefers Python over JavaScript" --context work 2>&1)
EXIT_CODE=$?
set -e
END_TIME=$(python3 -c "import time; print(time.time())")
DURATION=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
echo "$OUTPUT"
echo "Duration: ${DURATION}s"
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "FAIL: Command exited with code $EXIT_CODE"
    exit 1
fi
if ! echo "$OUTPUT" | grep -qi "retained"; then
    echo "FAIL: Expected 'retained' in output"
    exit 1
fi
echo "PASS: Memory with context retained successfully"

# Test 4: Recall with JSON output
echo ""
echo "Test 4: Recalling with JSON output..."
START_TIME=$(python3 -c "import time; print(time.time())")
set +e
JSON_OUTPUT=$(uv run --project "$SCRIPT_DIR" hindsight-embed memory recall "$BANK_ID" "programming preferences" -o json 2>&1)
EXIT_CODE=$?
set -e
END_TIME=$(python3 -c "import time; print(time.time())")
DURATION=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
echo "$JSON_OUTPUT"
echo "Duration: ${DURATION}s"
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "FAIL: Command exited with code $EXIT_CODE"
    exit 1
fi
if ! echo "$JSON_OUTPUT" | grep -qi "python"; then
    echo "FAIL: Expected 'Python' in recall output"
    exit 1
fi
if ! echo "$JSON_OUTPUT" | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
    echo "FAIL: Expected valid JSON output"
    exit 1
fi
echo "PASS: Memory recalled with JSON format successfully"

# Test 5: Check daemon is running
echo ""
echo "Test 5: Verifying daemon is running..."
if curl -s http://127.0.0.1:8888/health | grep -q "healthy"; then
    echo "PASS: Daemon is running and healthy"
else
    echo "FAIL: Daemon is not running"
    exit 1
fi

# Test 6: Daemon status command
echo ""
echo "Test 6: Testing daemon status command..."
STATUS_OUTPUT=$(uv run --project "$SCRIPT_DIR" hindsight-embed daemon status 2>&1)
echo "$STATUS_OUTPUT"
if ! echo "$STATUS_OUTPUT" | grep -qi "running"; then
    echo "FAIL: Expected 'running' in daemon status output"
    exit 1
fi
echo "PASS: Daemon status command works"

# Cleanup: Stop daemon
echo ""
echo "Stopping daemon..."
uv run --project "$SCRIPT_DIR" hindsight-embed daemon stop 2>/dev/null || true

echo ""
echo "=== All tests passed! ==="
