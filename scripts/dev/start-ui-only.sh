#!/usr/bin/env bash
# Start ONLY what's needed for Control Plane UI - NO WORKERS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "════════════════════════════════════════════════════════════════"
echo "Starting Hindsight Control Plane (UI Only - No Workers)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 1. Ensure Ollama lanes are running
echo "[1/3] Starting Ollama lanes..."
"$SCRIPT_DIR/start-ollama-split.sh"
echo ""

# 2. Start API (needed for Control Plane data)
echo "[2/3] Starting API..."
cd "$PROJECT_ROOT"
nohup "$SCRIPT_DIR/start-api.sh" --port 8888 > /tmp/hindsight-api.log 2>&1 &
API_PID=$!
echo "  API starting (PID: $API_PID)"
echo "  Waiting for API to be ready (loading models takes ~10s)..."
for i in {1..30}; do
    if curl -sf http://localhost:8888/health > /dev/null 2>&1; then
        echo "  ✓ API is ready"
        break
    fi
    sleep 1
done
echo ""

# 3. Start Control Plane
echo "[3/3] Starting Control Plane..."
cd "$PROJECT_ROOT"
nohup "$SCRIPT_DIR/start-control-plane.sh" > /tmp/hindsight-control-plane.log 2>&1 &
CP_PID=$!
echo "  Control Plane starting (PID: $CP_PID)"
echo "  Waiting for Control Plane to be ready..."
for i in {1..30}; do
    if curl -sf http://localhost:9998 > /dev/null 2>&1; then
        echo "  ✓ Control Plane is ready"
        break
    fi
    sleep 1
done
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "✅ Control Plane UI Ready"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Access:"
echo "  • Control Plane: http://localhost:9998"
echo "  • API:           http://localhost:8888"
echo "  • Grafana:       http://localhost:3000"
echo ""
echo "Logs:"
echo "  • API:           tail -f /tmp/hindsight-api.log"
echo "  • Control Plane: tail -f /tmp/hindsight-control-plane.log"
echo ""
echo "Note: Workers NOT started (use start-services.sh to start workers)"
echo ""
