#!/bin/bash
# CollabMind Backplane Health Check
# Usage: ./health-check.sh [--watch]
# Verifies all services defined in MANIFEST.yaml

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

WATCH=false
if [[ "${1:-}" == "--watch" ]]; then
  WATCH=true
fi

check_http() {
  local name=$1
  local url=$2
  local expected_code=${3:-200}
  
  if curl -sf -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -q "$expected_code"; then
    echo -e "  ${GREEN}✅${NC} $name ($url)"
    return 0
  else
    echo -e "  ${RED}❌${NC} $name ($url)"
    return 1
  fi
}

check_port() {
  local name=$1
  local port=$2
  
  if lsof -i :$port -P 2>/dev/null | grep -q LISTEN; then
    local pid=$(lsof -i :$port -P 2>/dev/null | awk 'NR>1 {print $2}' | head -1)
    echo -e "  ${GREEN}✅${NC} $name (port $port, pid $pid)"
    return 0
  else
    echo -e "  ${RED}❌${NC} $name (port $port — not listening)"
    return 1
  fi
}

check_docker() {
  local name=$1
  local container=$2
  
  if docker ps --filter "name=$container" --format "{{.Names}}" 2>/dev/null | grep -q .; then
    echo -e "  ${GREEN}✅${NC} $name ($container)"
    return 0
  else
    echo -e "  ${RED}❌${NC} $name ($container — not running)"
    return 1
  fi
}

do_checks() {
  local all_ok=true
  
  echo -e "\n${CYAN}═══════════════════════════════════════${NC}"
  echo -e "${CYAN}  CollabMind Backplane Health Check${NC}"
  echo -e "${CYAN}  $(date)${NC}"
  echo -e "${CYAN}═══════════════════════════════════════${NC}"

  echo -e "\n${YELLOW}📡 Ports & HTTP Services${NC}"
  check_http "Hindsight API" "http://localhost:8888/health" || all_ok=false
  check_port "Control Plane UI" 9998 || all_ok=false
  check_port "Memlord MCP" 8005 || all_ok=false
  check_port "Ollama Embeddings" 11434 || all_ok=false
  check_port "LM Studio" 1234 || all_ok=false
  check_port "Grafana" 3000 || all_ok=false

  echo -e "\n${YELLOW}🐳 Docker Containers${NC}"
  check_docker "Memlord Service" "memlord-service" || all_ok=false
  check_docker "Memlord Postgres" "memlord-postgres" || all_ok=false
  check_docker "Memlord Qdrant" "memlord-qdrant" || all_ok=false
  check_docker "Monitoring" "hindsight-monitoring" 2>/dev/null || \
    echo -e "  ${YELLOW}⚠️${NC} Monitoring stack (optional — not running)"

  echo -e "\n${YELLOW}💾 Databases${NC}"
  check_port "pg0 (Hindsight)" 5433 || all_ok=false
  check_port "Memlord PG" 5435 || all_ok=false
  check_port "Qdrant" 6333 || all_ok=false

  echo -e "\n${YELLOW}🤖 LLM Providers${NC}"
  if curl -sf http://192.168.1.144:1234/v1/models > /dev/null 2>&1; then
    echo -e "  ${GREEN}✅${NC} LM Studio (192.168.1.144:1234)"
  else
    echo -e "  ${RED}❌${NC} LM Studio (192.168.1.144:1234 — not reachable)"
    all_ok=false
  fi

  echo -e "\n${YELLOW}📊 Summary${NC}"
  if $all_ok; then
    echo -e "  ${GREEN}✅ All core services healthy${NC}"
  else
    echo -e "  ${RED}⚠️  Some services need attention${NC}"
  fi
  echo ""
}

if $WATCH; then
  while true; do
    clear 2>/dev/null || true
    do_checks
    sleep 5
  done
else
  do_checks
fi
