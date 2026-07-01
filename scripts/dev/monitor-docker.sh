#!/bin/bash
# Auto-restart Docker Desktop if it crashes
# Usage: ./monitor-docker.sh (runs in foreground)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "════════════════════════════════════════════════════════════════"
echo "Docker Monitor - Auto-restart on crash"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Monitoring Docker Desktop..."
echo "Press Ctrl+C to stop monitoring"
echo ""

CONSECUTIVE_FAILURES=0
MAX_FAILURES=3

while true; do
  if docker ps > /dev/null 2>&1; then
    # Docker is running
    if [ $CONSECUTIVE_FAILURES -gt 0 ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Docker recovered"
      CONSECUTIVE_FAILURES=0
    fi
  else
    # Docker is down
    CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Docker crashed (failure $CONSECUTIVE_FAILURES/$MAX_FAILURES)"
    
    if [ $CONSECUTIVE_FAILURES -ge $MAX_FAILURES ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 Restarting Docker Desktop..."
      open -a "Docker"
      
      # Wait for Docker to start
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏳ Waiting for Docker to start..."
      for i in {1..60}; do
        if docker ps > /dev/null 2>&1; then
          echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Docker started successfully"
          CONSECUTIVE_FAILURES=0
          break
        fi
        sleep 2
      done
      
      if [ $CONSECUTIVE_FAILURES -ge $MAX_FAILURES ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Docker failed to start after 120 seconds"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting 60 seconds before next attempt..."
        sleep 60
      fi
    fi
  fi
  
  # Check every 10 seconds
  sleep 10
done
