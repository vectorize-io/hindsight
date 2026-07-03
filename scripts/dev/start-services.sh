#!/usr/bin/env bash
# Start all Hindsight services properly in background
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"

echo "Starting Hindsight services..."
echo ""

# Kill any zombie processes first
echo "[0/4] Cleaning up zombie processes..."
pkill -9 -f "next dev.*hindsight-control-plane|next start.*hindsight-control-plane" 2>/dev/null || true
pkill -9 -f "hindsight-api|uvicorn.*hindsight" 2>/dev/null || true
pkill -9 -f "hindsight.*worker" 2>/dev/null || true
lsof -ti:8888 | xargs kill -9 2>/dev/null || true
lsof -ti:9998 | xargs kill -9 2>/dev/null || true
lsof -ti:10001 | xargs kill -9 2>/dev/null || true
sleep 2
echo "  ✓ Cleanup complete"
echo ""

# 1. Ensure Ollama lanes are running
echo "[1/4] Starting Ollama lanes..."
"$SCRIPT_DIR/start-ollama-split.sh"
echo ""

# 2. Start API
echo "[2/4] Starting API..."
cd "$PROJECT_ROOT"
nohup "$SCRIPT_DIR/start-api.sh" --port 8888 > "$PROJECT_ROOT/logs/hindsight-api.log" 2>&1 &
API_PID=$!
echo "  API starting (PID: $API_PID)"
echo "  Log: $PROJECT_ROOT/logs/hindsight-api.log"
echo "  Waiting for API to be ready..."
for i in {1..30}; do
    if curl -sf http://localhost:8888/health > /dev/null 2>&1; then
        echo "  ✓ API is ready"
        break
    fi
    sleep 1
done
echo ""

# 3. Start Control Plane
echo "[3/4] Starting Control Plane..."
cd "$PROJECT_ROOT"
nohup "$SCRIPT_DIR/start-control-plane.sh" > "$PROJECT_ROOT/logs/hindsight-control-plane.log" 2>&1 &
CP_PID=$!
echo "  Control Plane starting (PID: $CP_PID)"
echo "  Log: $PROJECT_ROOT/logs/hindsight-control-plane.log"
echo "  Waiting for Control Plane to be ready..."
for i in {1..30}; do
    if curl -sf http://localhost:9998 > /dev/null 2>&1; then
        echo "  ✓ Control Plane is ready"
        break
    fi
    sleep 1
done
echo ""

# 4. Start Workers
echo "[4/4] Starting Workers..."
# Read worker count from .env (default to 2 if not set)
WORKER_COUNT=$(grep "^HINDSIGHT_API_WORKERS=" "$PROJECT_ROOT/.env" | cut -d'=' -f2 | tr -d ' ')
WORKER_COUNT=${WORKER_COUNT:-2}
echo "  Worker count from .env: $WORKER_COUNT"
"$SCRIPT_DIR/scale-workers.sh" "$WORKER_COUNT"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "✅ All services started"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Services:"
echo "  • API:           http://localhost:8888"
echo "  • Control Plane: http://localhost:9998"
echo "  • Grafana:       http://localhost:3000"
echo "  • Workers:       $WORKER_COUNT running (ports 9001-900$((WORKER_COUNT)))"
echo ""
echo "Logs:"
echo "  • API:           tail -f $PROJECT_ROOT/logs/hindsight-api.log"
echo "  • Control Plane: tail -f $PROJECT_ROOT/logs/hindsight-control-plane.log"
echo "  • Workers:       tail -f $PROJECT_ROOT/logs/worker-*.log"
echo "  • All:           tail -f $PROJECT_ROOT/logs/*.log"
echo ""
echo "Status:"
echo "  • $SCRIPT_DIR/status.sh --watch"
echo ""
