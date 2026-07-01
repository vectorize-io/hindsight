#!/usr/bin/env bash
# Hindsight Status Monitor - Pure visibility, no actions
# Safe read-only dashboard for monitoring services

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
NC='\033[0m'

# Check service status
check_ollama_status() {
    local port=$1
    local name=$2
    
    if curl -s "http://localhost:$port/api/tags" > /dev/null 2>&1; then
        local model_count=$(curl -s "http://localhost:$port/api/tags" | grep -o '"name"' | wc -l | tr -d ' ')
        echo -e "${GREEN}✓${NC} $name (port $port) - ${model_count} models"
        return 0
    else
        echo -e "${RED}✗${NC} $name (port $port) - NOT RUNNING"
        return 1
    fi
}

check_api_status() {
    if curl -s http://localhost:8888/health > /dev/null 2>&1; then
        local pid=$(pgrep -f "hindsight-api" | head -1)
        echo -e "${GREEN}✓${NC} Hindsight API (port 8888) - RUNNING (PID: $pid)"
        return 0
    else
        if pgrep -f "hindsight-api" > /dev/null; then
            echo -e "${YELLOW}⚠${NC} Hindsight API (port 8888) - STARTING..."
            return 1
        else
            echo -e "${RED}✗${NC} Hindsight API (port 8888) - NOT RUNNING"
            return 1
        fi
    fi
}

check_workers_status() {
    if [ -f /tmp/hindsight-workers.state ]; then
        local running_workers=()
        local stopped_workers=()
        
        while IFS=: read -r worker_id pid port; do
            if ps -p "$pid" > /dev/null 2>&1; then
                running_workers+=("$worker_id (PID: $pid, port: $port)")
            else
                stopped_workers+=("$worker_id (stopped)")
            fi
        done < /tmp/hindsight-workers.state
        
        if [ ${#running_workers[@]} -gt 0 ]; then
            echo -e "${GREEN}✓${NC} Workers - ${#running_workers[@]} running"
            for worker in "${running_workers[@]}"; do
                echo -e "    ${CYAN}→${NC} $worker"
            done
        fi
        
        if [ ${#stopped_workers[@]} -gt 0 ]; then
            echo -e "${YELLOW}⚠${NC} Workers - ${#stopped_workers[@]} stopped"
            for worker in "${stopped_workers[@]}"; do
                echo -e "    ${YELLOW}→${NC} $worker"
            done
        fi
        
        [ ${#running_workers[@]} -gt 0 ]
    else
        echo -e "${YELLOW}⚠${NC} Workers - NOT CONFIGURED"
        return 1
    fi
}

check_monitoring_status() {
    if docker ps --filter "name=hindsight-monitoring" --format "{{.Status}}" 2>/dev/null | grep -q "Up"; then
        local status=$(docker ps --filter "name=hindsight-monitoring" --format "{{.Status}}")
        echo -e "${GREEN}✓${NC} Monitoring Stack - $status"
        echo -e "    ${CYAN}→${NC} Grafana: http://localhost:3000"
        echo -e "    ${CYAN}→${NC} OTLP: http://localhost:4317 (gRPC), http://localhost:4318 (HTTP)"
        return 0
    else
        echo -e "${RED}✗${NC} Monitoring Stack - NOT RUNNING"
        return 1
    fi
}

# Show configuration
show_config() {
    echo -e "${BOLD}Configuration:${NC}"
    echo ""
    
    if [ -f "$PROJECT_ROOT/.env" ]; then
        echo -e "${CYAN}Ollama Endpoints:${NC}"
        grep "OLLAMA_API_BASE" "$PROJECT_ROOT/.env" | sed 's/^/  /'
        echo ""
        
        echo -e "${CYAN}Models:${NC}"
        grep "OLLAMA_MODEL" "$PROJECT_ROOT/.env" | sed 's/^/  /'
        echo ""
        
        echo -e "${CYAN}Workers:${NC}"
        grep "WORKERS" "$PROJECT_ROOT/.env" | sed 's/^/  /'
        echo ""
    else
        echo -e "${RED}No .env file found${NC}"
        echo ""
    fi
}

# Show recent errors
show_errors() {
    local has_errors=false
    
    # Check API log
    if [ -f /tmp/hindsight-api.log ]; then
        local api_errors=$(grep -i "error\|exception\|failed" /tmp/hindsight-api.log 2>/dev/null | tail -5)
        if [ -n "$api_errors" ]; then
            echo -e "${RED}API Errors (last 5):${NC}"
            echo "$api_errors" | sed 's/^/  /'
            echo ""
            has_errors=true
        fi
    fi
    
    # Check worker logs
    if [ -d "$PROJECT_ROOT/logs" ]; then
        local worker_errors=$(grep -h -i "error\|exception\|failed" "$PROJECT_ROOT/logs"/worker-*.log 2>/dev/null | tail -5)
        if [ -n "$worker_errors" ]; then
            echo -e "${RED}Worker Errors (last 5):${NC}"
            echo "$worker_errors" | sed 's/^/  /'
            echo ""
            has_errors=true
        fi
    fi
    
    if [ "$has_errors" = false ]; then
        echo -e "${GREEN}✓ No recent errors${NC}"
        echo ""
    fi
}

# Show log locations
show_log_locations() {
    echo -e "${BOLD}Log Files:${NC}"
    echo ""
    
    if [ -f /tmp/hindsight-api.log ]; then
        local api_size=$(du -h /tmp/hindsight-api.log | cut -f1)
        echo -e "${CYAN}API:${NC} /tmp/hindsight-api.log ($api_size)"
    fi
    
    if [ -d "$PROJECT_ROOT/logs" ]; then
        for log in "$PROJECT_ROOT/logs"/worker-*.log; do
            if [ -f "$log" ]; then
                local size=$(du -h "$log" | cut -f1)
                echo -e "${CYAN}Worker:${NC} $log ($size)"
            fi
        done
    fi
    
    echo ""
    echo -e "${BLUE}Tip:${NC} tail -f /tmp/hindsight-api.log"
    echo -e "${BLUE}Tip:${NC} tail -f $PROJECT_ROOT/logs/worker-*.log"
    echo ""
}

# Show management commands
show_commands() {
    echo -e "${BOLD}Management Commands:${NC}"
    echo ""
    echo -e "${CYAN}Start/Stop:${NC}"
    echo "  ./scripts/dev/start-all.sh --monitoring --workers 4"
    echo "  ./scripts/dev/stop-monitoring.sh"
    echo "  ./scripts/dev/stop-ollama-split.sh"
    echo ""
    echo -e "${CYAN}Workers:${NC}"
    echo "  ./scripts/dev/scale-workers.sh 4        # Start 4 workers"
    echo "  ./scripts/dev/scale-workers.sh status   # Check status"
    echo "  ./scripts/dev/scale-workers.sh stop     # Stop all"
    echo ""
    echo -e "${CYAN}Ollama:${NC}"
    echo "  ./scripts/dev/start-ollama-split.sh     # Start LLM lane"
    echo "  ./scripts/dev/stop-ollama-split.sh      # Stop LLM lane"
    echo ""
    echo -e "${CYAN}Testing:${NC}"
    echo "  ./scripts/dev/test-integration.sh       # Run integration tests"
    echo ""
}

# Main status display
main() {
    local watch_mode=false
    local interval=5
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -w|--watch)
                watch_mode=true
                shift
                ;;
            -i|--interval)
                interval="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  -w, --watch           Watch mode (auto-refresh)"
                echo "  -i, --interval SEC    Refresh interval in seconds (default: 5)"
                echo "  -h, --help            Show this help"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    while true; do
        clear
        echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
        echo -e "${BOLD}${CYAN}           Hindsight Status Monitor${NC}"
        echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
        echo ""
        
        echo -e "${BOLD}Service Status:${NC}"
        echo ""
        check_ollama_status 11434 "Ollama Embeddings"
        check_ollama_status 11435 "Ollama LLM"
        check_api_status
        check_workers_status
        check_monitoring_status
        echo ""
        
        echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
        echo ""
        
        show_config
        
        echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
        echo ""
        
        echo -e "${BOLD}Recent Errors:${NC}"
        echo ""
        show_errors
        
        echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
        echo ""
        
        show_log_locations
        
        echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
        echo ""
        
        show_commands
        
        echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
        
        if [ "$watch_mode" = true ]; then
            echo ""
            echo -e "${BLUE}Refreshing in ${interval}s... (Ctrl+C to exit)${NC}"
            sleep "$interval"
        else
            break
        fi
    done
}

main "$@"
