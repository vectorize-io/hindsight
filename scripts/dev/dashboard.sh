#!/usr/bin/env bash
# Hindsight Development Dashboard
# Simple terminal UI for monitoring and controlling services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Clear screen and show header
show_header() {
    clear
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}           Hindsight Development Dashboard${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Check service status
check_ollama_status() {
    local port=$1
    local name=$2
    
    if curl -s "http://localhost:$port/api/tags" > /dev/null 2>&1; then
        local model_count=$(curl -s "http://localhost:$port/api/tags" | grep -o '"name"' | wc -l | tr -d ' ')
        echo -e "${GREEN}✓${NC} $name (port $port) - ${model_count} models"
    else
        echo -e "${RED}✗${NC} $name (port $port) - NOT RUNNING"
    fi
}

check_api_status() {
    if curl -s http://localhost:8888/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Hindsight API (port 8888) - RUNNING"
    else
        if pgrep -f "hindsight-api" > /dev/null; then
            echo -e "${YELLOW}⚠${NC} Hindsight API (port 8888) - STARTING..."
        else
            echo -e "${RED}✗${NC} Hindsight API (port 8888) - NOT RUNNING"
        fi
    fi
}

check_workers_status() {
    if [ -f /tmp/hindsight-workers.state ]; then
        local worker_count=$(grep -c "^worker-" /tmp/hindsight-workers.state 2>/dev/null || echo "0")
        local running_count=0
        
        while IFS=: read -r worker_id pid port; do
            if ps -p "$pid" > /dev/null 2>&1; then
                ((running_count++))
            fi
        done < /tmp/hindsight-workers.state
        
        if [ "$running_count" -gt 0 ]; then
            echo -e "${GREEN}✓${NC} Workers - $running_count running"
        else
            echo -e "${YELLOW}⚠${NC} Workers - 0 running (configured: $worker_count)"
        fi
    else
        echo -e "${YELLOW}⚠${NC} Workers - NOT CONFIGURED"
    fi
}

check_monitoring_status() {
    if docker ps --filter "name=hindsight-monitoring" --format "{{.Status}}" | grep -q "Up"; then
        echo -e "${GREEN}✓${NC} Monitoring Stack (Grafana) - RUNNING"
        echo -e "    ${CYAN}→${NC} http://localhost:3000"
    else
        echo -e "${RED}✗${NC} Monitoring Stack - NOT RUNNING"
    fi
}

# Show service status
show_status() {
    echo -e "${BOLD}Service Status:${NC}"
    echo ""
    check_ollama_status 11434 "Ollama Embeddings"
    check_ollama_status 11435 "Ollama LLM"
    check_api_status
    check_workers_status
    check_monitoring_status
    echo ""
}

# Show recent errors
show_errors() {
    echo -e "${BOLD}Recent Errors:${NC}"
    echo ""
    
    # Check API log
    if [ -f /tmp/hindsight-api.log ]; then
        local api_errors=$(grep -i "error\|exception\|failed" /tmp/hindsight-api.log 2>/dev/null | tail -3)
        if [ -n "$api_errors" ]; then
            echo -e "${RED}API Errors:${NC}"
            echo "$api_errors" | sed 's/^/  /'
            echo ""
        fi
    fi
    
    # Check worker logs
    if [ -d "$PROJECT_ROOT/logs" ]; then
        local worker_errors=$(grep -h -i "error\|exception\|failed" "$PROJECT_ROOT/logs"/worker-*.log 2>/dev/null | tail -3)
        if [ -n "$worker_errors" ]; then
            echo -e "${RED}Worker Errors:${NC}"
            echo "$worker_errors" | sed 's/^/  /'
            echo ""
        fi
    fi
    
    if [ -z "$api_errors" ] && [ -z "$worker_errors" ]; then
        echo -e "${GREEN}No recent errors${NC}"
        echo ""
    fi
}

# Show menu
show_menu() {
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}Actions:${NC}"
    echo ""
    echo "  [1] View API logs (live)"
    echo "  [2] View worker logs (live)"
    echo "  [3] View all errors"
    echo ""
    echo "  [4] Restart API"
    echo "  [5] Restart workers"
    echo "  [6] Scale workers"
    echo ""
    echo "  [7] Start monitoring stack"
    echo "  [8] Stop monitoring stack"
    echo ""
    echo "  [9] Start Ollama LLM lane"
    echo "  [0] Stop Ollama LLM lane"
    echo ""
    echo "  [s] Show status (refresh)"
    echo "  [q] Quit"
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -n "Select action: "
}

# View API logs
view_api_logs() {
    echo -e "${BLUE}Viewing API logs (Ctrl+C to return)...${NC}"
    echo ""
    tail -f /tmp/hindsight-api.log 2>/dev/null || echo "No API log found"
}

# View worker logs
view_worker_logs() {
    echo -e "${BLUE}Viewing worker logs (Ctrl+C to return)...${NC}"
    echo ""
    if [ -d "$PROJECT_ROOT/logs" ]; then
        tail -f "$PROJECT_ROOT/logs"/worker-*.log 2>/dev/null || echo "No worker logs found"
    else
        echo "No worker logs found"
    fi
}

# View all errors
view_all_errors() {
    echo -e "${BLUE}All recent errors:${NC}"
    echo ""
    
    if [ -f /tmp/hindsight-api.log ]; then
        echo -e "${BOLD}API Errors:${NC}"
        grep -i "error\|exception\|failed" /tmp/hindsight-api.log 2>/dev/null | tail -10 || echo "  No errors"
        echo ""
    fi
    
    if [ -d "$PROJECT_ROOT/logs" ]; then
        echo -e "${BOLD}Worker Errors:${NC}"
        grep -h -i "error\|exception\|failed" "$PROJECT_ROOT/logs"/worker-*.log 2>/dev/null | tail -10 || echo "  No errors"
        echo ""
    fi
    
    echo -n "Press Enter to continue..."
    read
}

# Restart API
restart_api() {
    echo -e "${YELLOW}Restarting API...${NC}"
    
    # Stop API
    pkill -f "hindsight-api" || true
    sleep 2
    
    # Start API using proper script (loads .env, uses correct uv environment)
    cd "$PROJECT_ROOT"
    nohup "$SCRIPT_DIR/start-api.sh" --port 8888 > /tmp/hindsight-api.log 2>&1 &
    
    echo -e "${GREEN}API restart initiated (using start-api.sh)${NC}"
    sleep 2
}

# Restart workers
restart_workers() {
    echo -e "${YELLOW}Restarting workers...${NC}"
    
    "$SCRIPT_DIR/scale-workers.sh" stop
    sleep 2
    
    echo -n "How many workers? [4]: "
    read worker_count
    worker_count=${worker_count:-4}
    
    "$SCRIPT_DIR/scale-workers.sh" "$worker_count"
    
    echo -e "${GREEN}Workers restarted${NC}"
    sleep 2
}

# Scale workers
scale_workers() {
    echo -n "How many workers? [4]: "
    read worker_count
    worker_count=${worker_count:-4}
    
    "$SCRIPT_DIR/scale-workers.sh" "$worker_count"
    
    echo -e "${GREEN}Workers scaled to $worker_count${NC}"
    sleep 2
}

# Start monitoring
start_monitoring() {
    echo -e "${YELLOW}Starting monitoring stack...${NC}"
    
    cd "$PROJECT_ROOT/monitoring"
    docker compose up -d
    
    echo -e "${GREEN}Monitoring stack started${NC}"
    echo -e "Grafana UI: ${CYAN}http://localhost:3000${NC}"
    sleep 2
}

# Stop monitoring
stop_monitoring() {
    echo -e "${YELLOW}Stopping monitoring stack...${NC}"
    
    "$SCRIPT_DIR/stop-monitoring.sh"
    
    echo -e "${GREEN}Monitoring stack stopped${NC}"
    sleep 2
}

# Start Ollama LLM lane
start_ollama_llm() {
    echo -e "${YELLOW}Starting Ollama LLM lane...${NC}"
    
    "$SCRIPT_DIR/start-ollama-split.sh"
    
    echo -e "${GREEN}Ollama LLM lane started${NC}"
    sleep 2
}

# Stop Ollama LLM lane
stop_ollama_llm() {
    echo -e "${YELLOW}Stopping Ollama LLM lane...${NC}"
    
    "$SCRIPT_DIR/stop-ollama-split.sh"
    
    echo -e "${GREEN}Ollama LLM lane stopped${NC}"
    sleep 2
}

# Main loop
main() {
    while true; do
        show_header
        show_status
        show_errors
        show_menu
        
        read -r choice
        
        case $choice in
            1) view_api_logs ;;
            2) view_worker_logs ;;
            3) view_all_errors ;;
            4) restart_api ;;
            5) restart_workers ;;
            6) scale_workers ;;
            7) start_monitoring ;;
            8) stop_monitoring ;;
            9) start_ollama_llm ;;
            0) stop_ollama_llm ;;
            s|S) continue ;;
            q|Q) 
                echo ""
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice${NC}"
                sleep 1
                ;;
        esac
    done
}

# Run dashboard
main
