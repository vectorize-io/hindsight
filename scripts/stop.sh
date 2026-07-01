#!/bin/bash
# Stop all Hindsight services gracefully
# Stops: Workers, Control Plane, API, Ollama split lanes, and optionally Monitoring (Docker)
#
# Usage:
#   ./stop-all.sh               # Stop everything except Docker monitoring
#   ./stop-all.sh --monitoring  # Stop everything including Docker monitoring
#   ./stop-all.sh --force       # Force kill if graceful shutdown fails

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/dev" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
STOP_MONITORING=false
FORCE_KILL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --monitoring)
            STOP_MONITORING=true
            shift
            ;;
        --force)
            FORCE_KILL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --monitoring    Also stop Docker monitoring stack (Grafana LGTM)"
            echo "  --force         Force kill processes if graceful shutdown fails"
            echo "  --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Stop all services (keep Docker monitoring)"
            echo "  $0 --monitoring       # Stop everything including monitoring"
            echo "  $0 --force            # Force kill stubborn processes"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option '$1'${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Header
echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e "${CYAN}Hindsight - Stopping All Services${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Track what we stopped
STOPPED_COUNT=0
FAILED_COUNT=0

# Helper function to stop process by PID
stop_process() {
    local pid=$1
    local name=$2
    local timeout=${3:-5}
    
    if ! kill -0 "$pid" 2>/dev/null; then
        echo -e "${YELLOW}  ⚠ Already stopped${NC}"
        return 0
    fi
    
    echo -n "  • Sending SIGTERM..."
    kill -TERM "$pid" 2>/dev/null || {
        echo -e " ${RED}FAILED${NC}"
        return 1
    }
    
    # Wait for graceful shutdown
    for i in $(seq 1 "$timeout"); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo -e " ${GREEN}OK${NC}"
            ((STOPPED_COUNT++))
            return 0
        fi
        sleep 1
    done
    
    # Timeout - force kill if requested
    if [ "$FORCE_KILL" = true ]; then
        echo -e " ${YELLOW}TIMEOUT, forcing...${NC}"
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
        if ! kill -0 "$pid" 2>/dev/null; then
            echo -e "  ${GREEN}✓ Killed${NC}"
            ((STOPPED_COUNT++))
            return 0
        fi
    else
        echo -e " ${RED}TIMEOUT (use --force to kill)${NC}"
        ((FAILED_COUNT++))
        return 1
    fi
}

# Step 1: Stop Workers
echo -e "${BLUE}[1/5] Stopping Workers...${NC}"
if [ -f "/tmp/hindsight-workers.state" ]; then
    WORKER_COUNT=$(wc -l < /tmp/hindsight-workers.state | tr -d ' ')
    echo "  • Found $WORKER_COUNT worker(s)"
    
    "$SCRIPT_DIR/scale-workers.sh" stop > /dev/null 2>&1 && {
        echo -e "${GREEN}  ✓ Workers stopped${NC}"
        ((STOPPED_COUNT += WORKER_COUNT))
    } || {
        echo -e "${RED}  ✗ Failed to stop workers${NC}"
        ((FAILED_COUNT++))
    }
else
    echo -e "${YELLOW}  ⚠ No workers running${NC}"
fi

# Step 2: Stop Control Plane
echo ""
echo -e "${BLUE}[2/5] Stopping Control Plane UI...${NC}"
CP_PID=$(pgrep -f "next dev.*9998" || true)
if [ -n "$CP_PID" ]; then
    echo "  • PID: $CP_PID"
    stop_process "$CP_PID" "Control Plane" 10
else
    echo -e "${YELLOW}  ⚠ Not running${NC}"
fi

# Step 3: Stop Hindsight API
echo ""
echo -e "${BLUE}[3/5] Stopping Hindsight API...${NC}"
API_PID=$(pgrep -f "hindsight-api.*--port" || true)
if [ -n "$API_PID" ]; then
    echo "  • PID: $API_PID"
    stop_process "$API_PID" "API" 10
else
    echo -e "${YELLOW}  ⚠ Not running${NC}"
fi

# Step 4: Stop Ollama Split Lanes
echo ""
echo -e "${BLUE}[4/5] Stopping Ollama Split Lanes...${NC}"

# Check if stop script exists
if [ -f "$SCRIPT_DIR/stop-ollama-split.sh" ]; then
    "$SCRIPT_DIR/stop-ollama-split.sh" && {
        echo -e "${GREEN}  ✓ Ollama lanes stopped${NC}"
        ((STOPPED_COUNT += 2))  # Both lanes
    } || {
        echo -e "${RED}  ✗ Failed to stop Ollama lanes${NC}"
        ((FAILED_COUNT++))
    }
else
    # Manual stop if script doesn't exist
    echo "  • Checking Ollama instances..."
    
    # Stop LLM lane (11435)
    LLM_PID=$(lsof -ti :11435 2>/dev/null | head -1 || true)
    if [ -n "$LLM_PID" ]; then
        echo "  • Stopping LLM lane (PID: $LLM_PID, Port: 11435)"
        stop_process "$LLM_PID" "Ollama LLM" 10
    else
        echo -e "${YELLOW}  ⚠ LLM lane not running${NC}"
    fi
    
    # Note: Keep embeddings lane (11434) running as it's the primary instance
    echo -e "${CYAN}  ℹ Keeping primary Ollama (11434) running${NC}"
fi

# Step 5: Optionally stop Docker monitoring
if [ "$STOP_MONITORING" = true ]; then
    echo ""
    echo -e "${BLUE}[5/5] Stopping Monitoring Stack (Docker)...${NC}"
    
    if command -v docker &> /dev/null; then
        if docker ps | grep -q hindsight-monitoring; then
            cd "$SCRIPT_DIR/monitoring"
            docker compose down > /dev/null 2>&1 && {
                echo -e "${GREEN}  ✓ Monitoring stack stopped${NC}"
                ((STOPPED_COUNT++))
            } || {
                echo -e "${RED}  ✗ Failed to stop monitoring${NC}"
                ((FAILED_COUNT++))
            }
            cd "$ROOT_DIR"
        else
            echo -e "${YELLOW}  ⚠ Monitoring container not running${NC}"
        fi
    else
        echo -e "${YELLOW}  ⚠ Docker not available${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}[5/5] Skipping Monitoring Stack (use --monitoring to stop Docker)${NC}"
fi

# Final summary
echo ""
echo "════════════════════════════════════════════════════════════════"
if [ $FAILED_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All Services Stopped Successfully${NC}"
else
    echo -e "${YELLOW}⚠ Stopped with warnings${NC}"
fi
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  • Stopped: $STOPPED_COUNT service(s)"
if [ $FAILED_COUNT -gt 0 ]; then
    echo -e "  • ${RED}Failed: $FAILED_COUNT service(s)${NC}"
    if [ "$FORCE_KILL" = false ]; then
        echo ""
        echo -e "${YELLOW}Tip: Use --force to kill stubborn processes${NC}"
    fi
fi
echo ""

# Show what's still running (if anything)
echo -e "${CYAN}Remaining Hindsight processes:${NC}"
REMAINING=$(ps aux | grep -E "(hindsight|ollama|next dev)" | grep -v grep | grep -v "stop-all" || true)
if [ -z "$REMAINING" ]; then
    echo "  • None (all clean)"
else
    echo "$REMAINING" | while read -r line; do
        echo "  • $line"
    done
    
    if [ "$FORCE_KILL" = false ]; then
        echo ""
        echo -e "${YELLOW}To force kill remaining processes, run:${NC}"
        echo "  $0 --force"
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Exit with appropriate code
if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
fi

exit 0
