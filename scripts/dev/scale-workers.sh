#!/bin/bash
# Scale Hindsight worker processes (bare metal)
# Usage:
#   ./scale-workers.sh <count>    # Start N workers
#   ./scale-workers.sh stop       # Stop all workers
#   ./scale-workers.sh status     # Show running workers
#   ./scale-workers.sh restart <count>  # Restart with new count

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATE_FILE="/tmp/hindsight-workers.state"
LOGS_DIR="$ROOT_DIR/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Load .env for default worker count
if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
fi

DEFAULT_WORKER_COUNT="${HINDSIGHT_API_WORKERS:-4}"
BASE_HTTP_PORT=9000

# Function to check if worker is running
is_worker_running() {
    local pid=$1
    kill -0 "$pid" 2>/dev/null
}

# Function to get worker health
check_worker_health() {
    local port=$1
    curl -sf "http://localhost:$port/health" > /dev/null 2>&1
}

# Function to stop all workers
stop_workers() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "Stopping Hindsight Workers"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    if [ ! -f "$STATE_FILE" ]; then
        echo -e "${YELLOW}⚠ No workers running (state file not found)${NC}"
        return 0
    fi
    
    local stopped=0
    local failed=0
    
    while IFS='|' read -r worker_id pid port; do
        if [ -z "$pid" ]; then
            continue
        fi
        
        echo -e "${BLUE}Stopping $worker_id (PID: $pid, Port: $port)...${NC}"
        
        if is_worker_running "$pid"; then
            kill -TERM "$pid" 2>/dev/null || {
                echo -e "${RED}  ✗ Failed to send SIGTERM${NC}"
                ((failed++))
                continue
            }
            
            # Wait for graceful shutdown (max 5 seconds)
            for i in {1..5}; do
                if ! is_worker_running "$pid"; then
                    echo -e "${GREEN}  ✓ Stopped${NC}"
                    ((stopped++))
                    break
                fi
                sleep 1
                
                if [ $i -eq 5 ]; then
                    echo -e "${YELLOW}  ⚠ Force killing...${NC}"
                    kill -KILL "$pid" 2>/dev/null || true
                    ((stopped++))
                fi
            done
        else
            echo -e "${YELLOW}  ⚠ Already stopped${NC}"
        fi
    done < "$STATE_FILE"
    
    rm -f "$STATE_FILE"
    
    echo ""
    echo -e "${GREEN}✓ Stopped $stopped worker(s)${NC}"
    if [ $failed -gt 0 ]; then
        echo -e "${RED}✗ Failed to stop $failed worker(s)${NC}"
    fi
    echo ""
}

# Function to show worker status
show_status() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "Hindsight Workers - Status"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    if [ ! -f "$STATE_FILE" ]; then
        echo -e "${YELLOW}⚠ No workers running${NC}"
        echo ""
        return 0
    fi
    
    local running=0
    local stopped=0
    
    while IFS='|' read -r worker_id pid port; do
        if [ -z "$pid" ]; then
            continue
        fi
        
        if is_worker_running "$pid"; then
            if check_worker_health "$port"; then
                echo -e "${GREEN}✓ $worker_id${NC} (PID: $pid, Port: $port) - HEALTHY"
            else
                echo -e "${YELLOW}⚠ $worker_id${NC} (PID: $pid, Port: $port) - RUNNING (health check failed)"
            fi
            ((running++))
        else
            echo -e "${RED}✗ $worker_id${NC} (PID: $pid, Port: $port) - STOPPED"
            ((stopped++))
        fi
    done < "$STATE_FILE"
    
    echo ""
    echo "Summary: $running running, $stopped stopped"
    echo ""
}

# Function to start workers
start_workers() {
    local count=$1
    
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "Starting $count Hindsight Worker(s)"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    # Stop existing workers first
    if [ -f "$STATE_FILE" ]; then
        echo -e "${BLUE}Stopping existing workers...${NC}"
        stop_workers
    fi
    
    # Ensure logs directory exists
    mkdir -p "$LOGS_DIR"
    
    # Start new workers
    > "$STATE_FILE"  # Clear state file
    
    for i in $(seq 1 "$count"); do
        local worker_id="worker-$i"
        local http_port=$((BASE_HTTP_PORT + i))
        local log_file="$LOGS_DIR/$worker_id.log"
        
        echo -e "${BLUE}Starting $worker_id...${NC}"
        echo "  • HTTP Port: $http_port"
        echo "  • Log File: $log_file"
        
        # Start worker in background
        # CRITICAL: Export all .env variables to worker process
        # Workers MUST see HINDSIGHT_API_LLM_OLLAMA_API_BASE and split Ollama config
        
        # Ensure we're in a valid directory before starting worker
        if [ ! -d "$ROOT_DIR" ]; then
            echo -e "${RED}Error: Root directory does not exist: $ROOT_DIR${NC}"
            exit 1
        fi
        
        # Load environment and start worker (no subshell to prevent SIGHUP)
        set -a
        source "$ROOT_DIR/.env"
        set +a
        
        # Start worker with nohup and detach properly
        HINDSIGHT_API_WORKER_ID="$worker_id" \
        HINDSIGHT_API_WORKER_HTTP_PORT="$http_port" \
        nohup hindsight-worker \
            --worker-id "$worker_id" \
            --http-port "$http_port" \
            > "$log_file" 2>&1 &
        
        local pid=$!
        
        # Save to state file
        echo "$worker_id|$pid|$http_port" >> "$STATE_FILE"
        
        # Wait briefly to check if it started
        sleep 1
        
        if is_worker_running "$pid"; then
            echo -e "${GREEN}  ✓ Started (PID: $pid)${NC}"
        else
            echo -e "${RED}  ✗ Failed to start${NC}"
            echo "    Check log: $log_file"
        fi
        
        echo ""
    done
    
    echo "════════════════════════════════════════════════════════════════"
    echo -e "${GREEN}Workers Started${NC}"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Endpoints:"
    
    while IFS='|' read -r worker_id pid port; do
        if [ -n "$pid" ] && is_worker_running "$pid"; then
            echo "  • $worker_id: http://localhost:$port/metrics"
        fi
    done < "$STATE_FILE"
    
    echo ""
    echo "Logs directory: $LOGS_DIR"
    echo ""
}

# Main command handling
case "${1:-}" in
    stop)
        stop_workers
        ;;
    status)
        show_status
        ;;
    restart)
        count="${2:-$DEFAULT_WORKER_COUNT}"
        stop_workers
        start_workers "$count"
        ;;
    [0-9]*)
        count="$1"
        start_workers "$count"
        ;;
    "")
        echo "Usage: $0 <count|stop|status|restart>"
        echo ""
        echo "Commands:"
        echo "  $0 <count>         Start N workers"
        echo "  $0 stop            Stop all workers"
        echo "  $0 status          Show worker status"
        echo "  $0 restart <count> Restart with new count"
        echo ""
        echo "Default worker count: $DEFAULT_WORKER_COUNT (from HINDSIGHT_API_WORKERS in .env)"
        exit 1
        ;;
    *)
        echo -e "${RED}Error: Invalid command '$1'${NC}"
        echo "Usage: $0 <count|stop|status|restart>"
        exit 1
        ;;
esac
