#!/bin/bash
set -e

# =============================================================================
# Embedded pg0 data integrity check (#675)
#
# When using embedded pg0, check if the data directory has existing PostgreSQL
# data before starting. If the directory exists but appears empty/corrupt
# (e.g., missing PG_VERSION file), log a warning. This helps diagnose data
# loss scenarios where a container restart caused the data directory to be
# wiped despite a volume mount being present.
# =============================================================================
PG0_DATA_DIR="${HOME}/.pg0"
if [ -d "$PG0_DATA_DIR" ]; then
    # Look for actual PostgreSQL data directories (pg0 creates subdirs per instance)
    if compgen -G "$PG0_DATA_DIR"/*/PG_VERSION > /dev/null 2>&1; then
        echo "✅ Existing pg0 data directory detected at $PG0_DATA_DIR"
    elif [ "$(ls -A "$PG0_DATA_DIR" 2>/dev/null)" ]; then
        echo "⚠️  WARNING: pg0 data directory exists at $PG0_DATA_DIR but no PG_VERSION found."
        echo "   This may indicate data corruption or an incomplete previous shutdown."
        echo "   If you see all migrations running from scratch after this, your data may have been lost."
        echo "   See: https://github.com/vectorize-io/hindsight/issues/675"
    fi
fi

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

# =============================================================================
# Graceful shutdown handler (#675)
#
# Docker sends SIGTERM on `docker stop`/`docker restart`. Without a trap, child
# processes (hindsight-api + pg0, control-plane) are killed abruptly. For the
# embedded pg0 database this can cause data loss when the data directory is on
# a Docker volume that gets remounted after restart.
#
# The trap forwards SIGTERM to all tracked child PIDs so that:
#   - hindsight-api receives the signal and can run its shutdown hooks
#   - pg0 gets a clean PostgreSQL shutdown (checkpoint + WAL flush)
#   - The control-plane Node.js process exits cleanly
# =============================================================================
# Guard against concurrent cleanup (e.g., child crash + SIGTERM arriving together)
SHUTTING_DOWN=false

cleanup() {
    if $SHUTTING_DOWN; then return; fi
    SHUTTING_DOWN=true

    echo ""
    echo "🛑 Received shutdown signal, stopping services gracefully..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    # Give processes time to shut down cleanly (pg0 needs to flush WAL).
    # NOTE: Docker's default stop_grace_period is 10s. If you use the default,
    # either set stop_grace_period: 30s in your compose file / docker stop -t 30,
    # or Docker will SIGKILL the container before this timeout expires.
    local timeout=30
    for ((i=1; i<=timeout; i++)); do
        local all_stopped=true
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                all_stopped=false
                break
            fi
        done
        if $all_stopped; then
            echo "✅ All services stopped cleanly"
            exit 0
        fi
        sleep 1
    done
    # Force kill if still running after timeout
    echo "⚠️  Timeout reached, forcing shutdown..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    exit 1
}
trap cleanup SIGTERM SIGINT

# Track PIDs for wait
PIDS=()

# Start API if enabled
if [ "$ENABLE_API" = "true" ]; then
    cd /app/api
    API_HEALTH_URL="${HINDSIGHT_API_HEALTH_URL:-http://localhost:8888/health}"
    API_STARTUP_WAIT_SECONDS="${HINDSIGHT_API_STARTUP_WAIT_SECONDS:-300}"

    # Run API directly - Python's PYTHONUNBUFFERED=1 handles output buffering
    hindsight-api &
    API_PID=$!
    PIDS+=($API_PID)

    # Wait for API to be ready
    api_ready=false
    for ((i=1; i<=API_STARTUP_WAIT_SECONDS; i++)); do
        if ! kill -0 "$API_PID" 2>/dev/null; then
            wait "$API_PID"
            exit $?
        fi
        if curl -sf "$API_HEALTH_URL" &>/dev/null; then
            api_ready=true
            break
        fi
        sleep 1
    done

    if [ "$api_ready" != "true" ]; then
        echo "❌ API did not become healthy within ${API_STARTUP_WAIT_SECONDS}s"
        exit 1
    fi
else
    echo "API disabled (HINDSIGHT_ENABLE_API=false)"
fi

# Start Control Plane if enabled
if [ "$ENABLE_CP" = "true" ]; then
    echo "🎛️  Starting Control Plane..."
    cd /app/control-plane
    export HOSTNAME="${HINDSIGHT_CP_HOSTNAME:-0.0.0.0}"
    PORT="${HINDSIGHT_CP_PORT:-9999}" node server.js &
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
    echo "   Control Plane: http://localhost:${HINDSIGHT_CP_PORT:-9999}"
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

# Wait for any process to exit (use wait -n with trap-safe loop)
while true; do
    # wait -n returns when any child exits; it also returns on signal delivery
    # (the trap handler will run and exit, so this loop is just for robustness).
    # `&& true` prevents `set -e` from killing the script when wait -n returns
    # non-zero (child exited with error or no backgrounded children remain).
    wait -n && true
    # Check if any tracked PID has exited
    for pid in "${PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            exit_code=$?
            echo "⚠️  Service (PID $pid) exited with code $exit_code"
            # Trigger cleanup for remaining services
            cleanup
        fi
    done
done
