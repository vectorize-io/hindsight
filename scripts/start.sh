#!/bin/bash
# Production startup script for Hindsight
# Order: Database → API → Control Plane → Monitoring → Workers (scaling)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "════════════════════════════════════════════════════════════════"
echo "Starting Hindsight Production Environment"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════
# Step 1: Check Database (PostgreSQL)
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}[1/5] Checking Database...${NC}"

PG_PORT=5433
if lsof -ti:$PG_PORT > /dev/null 2>&1; then
    echo -e "${GREEN}  ✓ PostgreSQL running on port $PG_PORT${NC}"
else
    echo -e "${RED}  ✗ PostgreSQL NOT running on port $PG_PORT${NC}"
    echo -e "${YELLOW}  ⚠ Start PostgreSQL first before running this script${NC}"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════
# Step 2: Start Hindsight API
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${BLUE}[2/5] Starting Hindsight API...${NC}"

API_PORT=8888
if lsof -ti:$API_PORT > /dev/null 2>&1; then
    echo -e "${YELLOW}  ⚠ API already running on port $API_PORT${NC}"
else
    bash "$SCRIPT_DIR/dev/start-ollama-split.sh"
    
    nohup bash "$SCRIPT_DIR/dev/start-api.sh" > /tmp/hindsight-api.log 2>&1 &
    API_PID=$!
    
    echo "  • Starting API (PID: $API_PID)"
    echo "  • Waiting for API health check..."
    
    for i in {1..30}; do
        if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
            echo -e "${GREEN}  ✓ API healthy${NC}"
            break
        fi
        sleep 1
    done
fi

# ═══════════════════════════════════════════════════════════════════
# Step 3: Start Control Plane (Dashboard)
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${BLUE}[3/5] Starting Control Plane (Dashboard)...${NC}"

CP_PORT=9998
if lsof -ti:$CP_PORT > /dev/null 2>&1; then
    echo -e "${YELLOW}  ⚠ Control Plane already running on port $CP_PORT${NC}"
else
    bash "$SCRIPT_DIR/dev/start-control-plane-production.sh" > /dev/null 2>&1 &
    CP_PID=$!
    
    echo "  • Starting Control Plane (PID: $CP_PID)"
    echo "  • Waiting for Control Plane..."
    
    sleep 5
    echo -e "${GREEN}  ✓ Control Plane started${NC}"
fi

# ═══════════════════════════════════════════════════════════════════
# Step 4: Start Monitoring & Check Conditions
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${BLUE}[4/5] Monitoring & Status Check...${NC}"

# Check Docker
if docker ps > /dev/null 2>&1; then
    echo -e "${GREEN}  ✓ Docker running${NC}"
    
    # Check if monitoring container exists
    if docker ps | grep -q hindsight-monitoring; then
        echo -e "${GREEN}  ✓ Monitoring stack running (Grafana: http://localhost:3000)${NC}"
    else
        echo -e "${YELLOW}  ⚠ Monitoring stack not running${NC}"
        echo "  • Start with: bash $SCRIPT_DIR/dev/start-monitoring.sh"
    fi
else
    echo -e "${YELLOW}  ⚠ Docker not running (monitoring unavailable)${NC}"
fi

# Service status
echo ""
echo "Service Status:"
curl -s http://localhost:$API_PORT/health 2>&1 | grep -q "healthy" && echo -e "${GREEN}  ✓ API: http://localhost:$API_PORT${NC}" || echo -e "${RED}  ✗ API: DOWN${NC}"
curl -s http://localhost:$CP_PORT > /dev/null 2>&1 && echo -e "${GREEN}  ✓ Control Plane: http://localhost:$CP_PORT${NC}" || echo -e "${RED}  ✗ Control Plane: DOWN${NC}"

# ═══════════════════════════════════════════════════════════════════
# Step 5: Scaling (Workers) - LAST
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${BLUE}[5/5] Scaling Workers...${NC}"

# Get worker count from .env or default to 2
WORKER_COUNT=$(grep "^HINDSIGHT_API_WORKERS=" "$ROOT_DIR/.env" 2>/dev/null | cut -d'=' -f2 || echo "2")
WORKER_COUNT=${WORKER_COUNT:-2}

echo "  • Starting $WORKER_COUNT workers..."
bash "$SCRIPT_DIR/dev/scale-workers.sh" "$WORKER_COUNT"

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ Hindsight Started${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Services:"
echo "  • API:           http://localhost:$API_PORT"
echo "  • Control Plane: http://localhost:$CP_PORT"
echo "  • Workers:       $WORKER_COUNT running"
echo ""
echo "Logs:"
echo "  • API:           tail -f /tmp/hindsight-api.log"
echo "  • Control Plane: tail -f /tmp/hindsight-control-plane.log"
echo "  • Workers:       tail -f $ROOT_DIR/logs/worker-*.log"
echo ""
echo "Status:"
echo "  • bash $SCRIPT_DIR/dev/status.sh --watch"
echo ""
