#!/bin/bash
# Stop the secondary Ollama LLM lane (port 11435)
# Leaves the primary embeddings lane (port 11434) running

set -e

LLM_PORT="${OLLAMA_LLM_PORT:-11435}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Stopping Ollama LLM Lane (port $LLM_PORT)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Find process listening on LLM port
PID=$(lsof -ti :$LLM_PORT 2>/dev/null || echo "")

if [ -z "$PID" ]; then
    echo -e "${YELLOW}⚠ No process found on port $LLM_PORT${NC}"
    echo "  LLM lane is not running"
    exit 0
fi

echo -e "${BLUE}Found Ollama process:${NC}"
echo "  • PID: $PID"
echo "  • Port: $LLM_PORT"
echo ""

# Send SIGTERM for graceful shutdown
echo -e "${BLUE}Sending SIGTERM for graceful shutdown...${NC}"
kill -TERM "$PID" 2>/dev/null || {
    echo -e "${RED}✗ Failed to send SIGTERM to PID $PID${NC}"
    exit 1
}

# Wait for process to exit (max 10 seconds)
echo -n "Waiting for process to exit"
for i in {1..10}; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo ""
        echo -e "${GREEN}✓ LLM lane stopped successfully${NC}"
        echo ""
        exit 0
    fi
    echo -n "."
    sleep 1
done

# If still running after 10 seconds, force kill
echo ""
echo -e "${YELLOW}⚠ Process still running after 10 seconds, forcing shutdown...${NC}"
kill -KILL "$PID" 2>/dev/null || {
    echo -e "${RED}✗ Failed to force kill PID $PID${NC}"
    exit 1
}

sleep 1

if ! kill -0 "$PID" 2>/dev/null; then
    echo -e "${GREEN}✓ LLM lane stopped (forced)${NC}"
    echo ""
else
    echo -e "${RED}✗ Failed to stop LLM lane${NC}"
    exit 1
fi
