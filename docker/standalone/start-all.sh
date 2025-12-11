#!/bin/bash
set -e

# Service flags (default to true if not set)
ENABLE_API="${HINDSIGHT_ENABLE_API:-true}"
ENABLE_CP="${HINDSIGHT_ENABLE_CP:-true}"

# Copy pre-cached PostgreSQL data if runtime directory is empty (first run with volume)
if [ "$ENABLE_API" = "true" ]; then
    PG0_CACHE="/home/hindsight/.pg0-cache"
    PG0_HOME="/home/hindsight/.pg0"
    if [ -d "$PG0_CACHE" ] && [ "$(ls -A $PG0_CACHE 2>/dev/null)" ]; then
        if [ ! "$(ls -A $PG0_HOME 2>/dev/null)" ]; then
            echo "📦 Copying pre-cached PostgreSQL data..."
            cp -r "$PG0_CACHE"/* "$PG0_HOME"/ 2>/dev/null || true
        fi
    fi
fi

# Track PIDs for wait
PIDS=()

# Start API if enabled
if [ "$ENABLE_API" = "true" ]; then
    cd /app/api
    hindsight-api 2>&1 | sed -u 's/^/[api] /' &
    API_PID=$!
    PIDS+=($API_PID)

    # Wait for API to be ready
    for i in {1..60}; do
        if curl -sf http://localhost:8888/health &>/dev/null; then
            break
        fi
        sleep 1
    done
else
    echo "API disabled (HINDSIGHT_ENABLE_API=false)"
fi

# Start Control Plane if enabled
if [ "$ENABLE_CP" = "true" ]; then
    echo "🎛️  Starting Control Plane..."
    cd /app/control-plane
    PORT=9999 node server.js 2>&1 | grep -v -E "^[[:space:]]*(▲|✓|-|$)" | sed -u 's/^/[control-plane] /' &
    CP_PID=$!
    PIDS+=($CP_PID)
else
    echo "Control Plane disabled (HINDSIGHT_ENABLE_CP=false)"
fi

# Print status
echo ""
echo "✅ Hindsight is running!"
echo ""
echo "📍 Access:"
if [ "$ENABLE_CP" = "true" ]; then
    echo "   Control Plane: http://localhost:9999"
fi
if [ "$ENABLE_API" = "true" ]; then
    echo "   API:           http://localhost:8888"
fi
echo ""

# Check if any services are running
if [ ${#PIDS[@]} -eq 0 ]; then
    echo "❌ No services enabled! Set HINDSIGHT_ENABLE_API=true or HINDSIGHT_ENABLE_CP=true"
    exit 1
fi

# Wait for any process to exit
wait -n

# Exit with status of first exited process
exit $?
