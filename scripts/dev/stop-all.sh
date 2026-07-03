#!/usr/bin/env bash
# Stop all Hindsight services and kill zombie processes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Stopping all Hindsight services..."
echo ""

# Function to kill processes by pattern with retries
kill_processes() {
    local pattern="$1"
    local description="$2"
    local max_retries=3
    
    echo "[$description] Searching for processes..."
    
    for attempt in $(seq 1 $max_retries); do
        # Find PIDs matching pattern
        PIDS=$(pgrep -f "$pattern" 2>/dev/null || true)
        
        if [ -z "$PIDS" ]; then
            echo "  ✓ No $description processes running"
            return 0
        fi
        
        echo "  Found PIDs: $PIDS"
        
        if [ $attempt -eq 1 ]; then
            # First attempt: graceful SIGTERM
            echo "  Attempt $attempt: Sending SIGTERM..."
            echo "$PIDS" | xargs kill -15 2>/dev/null || true
            sleep 2
        elif [ $attempt -eq 2 ]; then
            # Second attempt: SIGINT
            echo "  Attempt $attempt: Sending SIGINT..."
            echo "$PIDS" | xargs kill -2 2>/dev/null || true
            sleep 1
        else
            # Final attempt: Force kill
            echo "  Attempt $attempt: Force killing with SIGKILL..."
            echo "$PIDS" | xargs kill -9 2>/dev/null || true
            sleep 1
        fi
    done
    
    # Final check
    REMAINING=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$REMAINING" ]; then
        echo "  ⚠️  Warning: Some processes still running: $REMAINING"
    else
        echo "  ✓ All $description processes stopped"
    fi
}

# Function to kill by port
kill_port() {
    local port="$1"
    local description="$2"
    
    echo "[$description] Checking port $port..."
    PIDS=$(lsof -ti:$port 2>/dev/null || true)
    
    if [ -z "$PIDS" ]; then
        echo "  ✓ Port $port is free"
        return 0
    fi
    
    echo "  Found PIDs on port $port: $PIDS"
    echo "  Killing..."
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
    sleep 1
    
    # Verify
    REMAINING=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$REMAINING" ]; then
        echo "  ⚠️  Warning: Port $port still in use by PID: $REMAINING"
    else
        echo "  ✓ Port $port is now free"
    fi
}

echo "═══════════════════════════════════════════════════════════════"
echo "Stopping services..."
echo "═══════════════════════════════════════════════════════════════"
echo ""

# 1. Stop Control Plane (Next.js)
kill_processes "next dev.*hindsight-control-plane|next start.*hindsight-control-plane|node.*server.js.*hindsight" "Control Plane"
kill_port 9998 "Control Plane port"
kill_port 9999 "Control Plane alt port"
kill_port 10001 "Control Plane production port"
echo ""

# 2. Stop Hindsight API
kill_processes "hindsight-api|hindsight_api.main|uvicorn.*hindsight" "Hindsight API"
kill_port 8888 "Hindsight API port"
echo ""

# 3. Stop Workers
kill_processes "hindsight.*worker|worker.*hindsight" "Hindsight Workers"
# Worker metrics ports (9001-9010)
for port in {9001..9010}; do
    PIDS=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "  Killing worker on port $port (PID: $PIDS)"
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
    fi
done
echo ""

# 4. Clean up worker state file
if [ -f /tmp/hindsight-workers.state ]; then
    echo "[Worker State] Removing /tmp/hindsight-workers.state"
    rm -f /tmp/hindsight-workers.state
    echo "  ✓ Worker state cleaned"
else
    echo "[Worker State] No state file found"
fi
echo ""

# 5. Stop MCP servers (Memlord)
kill_processes "memlord.*mcp|python.*memlord" "MCP Servers"
kill_port 8005 "Memlord MCP port"
echo ""

# 6. Optional: Stop Ollama split lanes
read -p "Stop Ollama split lanes? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    "$SCRIPT_DIR/stop-ollama-split.sh" || true
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "✅ All services stopped"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "To verify, run: $SCRIPT_DIR/status.sh"
echo ""
