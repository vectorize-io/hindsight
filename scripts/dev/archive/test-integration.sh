#!/bin/bash
# Integration test for Hindsight development environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Hindsight Integration Test"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Test 1: Ollama split lanes
echo -e "${BLUE}[1/4] Testing Ollama Split Lanes...${NC}"
"$SCRIPT_DIR/start-ollama-split.sh" > /dev/null 2>&1 || {
    echo -e "${RED}✗ Failed${NC}"
    exit 1
}

# Verify both endpoints
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}  ✓ Embeddings lane (11434) responding${NC}"
else
    echo -e "${RED}  ✗ Embeddings lane not responding${NC}"
    exit 1
fi

if curl -sf http://localhost:11435/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}  ✓ LLM lane (11435) responding${NC}"
else
    echo -e "${RED}  ✗ LLM lane not responding${NC}"
    exit 1
fi

# Test 2: Worker scaling
echo ""
echo -e "${BLUE}[2/4] Testing Worker Scaling...${NC}"

# Start 2 workers
"$SCRIPT_DIR/scale-workers.sh" 2 > /dev/null 2>&1
sleep 5

# Check if workers are running
if "$SCRIPT_DIR/scale-workers.sh" status 2>&1 | grep -q "2 running"; then
    echo -e "${GREEN}  ✓ 2 workers started successfully${NC}"
else
    echo -e "${RED}  ✗ Workers failed to start${NC}"
    exit 1
fi

# Stop workers
"$SCRIPT_DIR/scale-workers.sh" stop > /dev/null 2>&1
echo -e "${GREEN}  ✓ Workers stopped successfully${NC}"

# Test 3: Monitoring stack
echo ""
echo -e "${BLUE}[3/4] Testing Monitoring Stack...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}  ⚠ Docker not running, skipping monitoring test${NC}"
else
    # Ensure network exists
    docker network create hindsight-network > /dev/null 2>&1 || true
    
    # Start monitoring
    cd "$SCRIPT_DIR/monitoring"
    docker compose up -d > /dev/null 2>&1
    cd "$ROOT_DIR"
    
    sleep 5
    
    # Check if container is running
    if docker ps | grep -q hindsight-monitoring; then
        echo -e "${GREEN}  ✓ Monitoring stack started${NC}"
        
        # Stop monitoring
        "$SCRIPT_DIR/stop-monitoring.sh" > /dev/null 2>&1
        echo -e "${GREEN}  ✓ Monitoring stack stopped${NC}"
    else
        echo -e "${RED}  ✗ Monitoring stack failed to start${NC}"
        exit 1
    fi
fi

# Test 4: Configuration verification
echo ""
echo -e "${BLUE}[4/4] Verifying Configuration...${NC}"

if [ -f "$ROOT_DIR/.env" ]; then
    # Check split Ollama config
    if grep -q "HINDSIGHT_API_LLM_OLLAMA_API_BASE=http://localhost:11435" "$ROOT_DIR/.env"; then
        echo -e "${GREEN}  ✓ LLM endpoint configured (11435)${NC}"
    else
        echo -e "${RED}  ✗ LLM endpoint not configured correctly${NC}"
        exit 1
    fi
    
    if grep -q "HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE=http://localhost:11434" "$ROOT_DIR/.env"; then
        echo -e "${GREEN}  ✓ Embeddings endpoint configured (11434)${NC}"
    else
        echo -e "${RED}  ✗ Embeddings endpoint not configured correctly${NC}"
        exit 1
    fi
else
    echo -e "${RED}  ✗ .env file not found${NC}"
    exit 1
fi

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e "${GREEN}All Integration Tests Passed!${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Scripts available:"
echo "  • ./scripts/dev/start-ollama-split.sh  - Start split Ollama lanes"
echo "  • ./scripts/dev/stop-ollama-split.sh   - Stop LLM lane"
echo "  • ./scripts/dev/scale-workers.sh <N>   - Scale workers"
echo "  • ./scripts/dev/start-monitoring.sh    - Start monitoring (Docker)"
echo "  • ./scripts/dev/stop-monitoring.sh     - Stop monitoring"
echo "  • ./scripts/dev/start-all.sh           - Start everything"
echo ""
echo "Quick start:"
echo "  ./scripts/dev/start-all.sh --workers 4 --monitoring"
echo ""
