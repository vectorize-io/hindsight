#!/bin/bash
# Stop the Hindsight monitoring stack (Grafana LGTM)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITORING_DIR="$SCRIPT_DIR/monitoring"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Stopping Hindsight Monitoring Stack"
echo "════════════════════════════════════════════════════════════════"
echo ""

cd "$MONITORING_DIR"

# Check if monitoring stack is running
if ! docker compose ps | grep -q "hindsight-monitoring"; then
    echo -e "${YELLOW}⚠ Monitoring stack is not running${NC}"
    echo ""
    exit 0
fi

echo -e "${BLUE}Stopping Grafana LGTM stack...${NC}"
docker compose down

echo ""
echo -e "${GREEN}✓ Monitoring stack stopped${NC}"
echo ""
