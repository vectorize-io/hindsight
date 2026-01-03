#!/bin/bash
set -e

# Service flags (default to true if not set)
ENABLE_API="${HINDSIGHT_ENABLE_API:-true}"
ENABLE_CP="${HINDSIGHT_ENABLE_CP:-true}"

# =============================================================================
# Dependency waiting (opt-in via HINDSIGHT_WAIT_FOR_DEPS=true)
#
# Problem: When running with LM Studio, the LLM may take time to load models.
# If Hindsight starts before LM Studio is ready, it fails on LLM verification.
# This wait loop ensures dependencies are ready before starting.
# =============================================================================
if [ "${HINDSIGHT_WAIT_FOR_DEPS:-false}" = "true" ]; then
    LLM_BASE_URL="${HINDSIGHT_API_LLM_BASE_URL:-http://host.docker.internal:1234/v1}"
    MAX_RETRIES="${HINDSIGHT_RETRY_MAX:-0}"  # 0 = infinite
    RETRY_INTERVAL="${HINDSIGHT_RETRY_INTERVAL:-10}"

    # Check if external database is configured (skip check for embedded pg0)
    SKIP_DB_CHECK=false
    if [ -z "${HINDSIGHT_API_DATABASE_URL}" ]; then
        SKIP_DB_CHECK=true
    else
        DB_CHECK_HOST=$(echo "$HINDSIGHT_API_DATABASE_URL" | sed -E 's|.*@([^:/]+):([0-9]+)/.*|\1 \2|')
    fi

    check_db() {
        if $SKIP_DB_CHECK; then
            return 0
        fi
        if command -v pg_isready &> /dev/null; then
            pg_isready -h $(echo $DB_CHECK_HOST | cut -d' ' -f1) -p $(echo $DB_CHECK_HOST | cut -d' ' -f2) &>/dev/null
        else
            python3 -c "import socket; s=socket.socket(); s.settimeout(5); exit(0 if s.connect_ex(('$(echo $DB_CHECK_HOST | cut -d' ' -f1)', $(echo $DB_CHECK_HOST | cut -d' ' -f2))) == 0 else 1)" 2>/dev/null
        fi
    }

    check_llm() {
        curl -sf "${LLM_BASE_URL}/models" --connect-timeout 5 &>/dev/null
    }

    echo "⏳ Waiting for dependencies to be ready..."
    attempt=1

    while true; do
        db_ok=false
        llm_ok=false

        if check_db; then
            db_ok=true
        fi

        if check_llm; then
            llm_ok=true
        fi

        if $db_ok && $llm_ok; then
            echo "✅ Dependencies ready!"
            break
        fi

        if [ "$MAX_RETRIES" -ne 0 ] && [ "$attempt" -ge "$MAX_RETRIES" ]; then
            echo "❌ Max retries ($MAX_RETRIES) reached. Dependencies not available."
            exit 1
        fi

        echo "   Attempt $attempt: DB=$( $db_ok && echo 'ok' || echo 'waiting' ), LLM=$( $llm_ok && echo 'ok' || echo 'waiting' )"
        sleep "$RETRY_INTERVAL"
        ((attempt++))
    done
fi

# Track PIDs for wait
PIDS=()

# Start API if enabled
if [ "$ENABLE_API" = "true" ]; then
    cd /app/api
    # Run API directly - Python's PYTHONUNBUFFERED=1 handles output buffering
    hindsight-api &
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
    PORT=9999 node server.js &
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
