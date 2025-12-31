#!/bin/bash
# Retry wrapper - waits for dependencies before starting hindsight

LLM_BASE_URL="${HINDSIGHT_API_LLM_BASE_URL:-http://host.docker.internal:1234/v1}"
DB_HOST="${HINDSIGHT_API_DATABASE_URL:-postgresql://hindsight:hindsight@hindsight-db:5432/hindsight}"
MAX_RETRIES="${HINDSIGHT_RETRY_MAX:-0}"  # 0 = infinite
RETRY_INTERVAL="${HINDSIGHT_RETRY_INTERVAL:-10}"

# Extract DB host from URL for checking
DB_CHECK_HOST=$(echo "$DB_HOST" | sed -E 's|.*@([^:/]+):([0-9]+)/.*|\1 \2|')

check_db() {
    if command -v pg_isready &> /dev/null; then
        pg_isready -h $(echo $DB_CHECK_HOST | cut -d' ' -f1) -p $(echo $DB_CHECK_HOST | cut -d' ' -f2) &>/dev/null
    else
        # Fallback: try connecting with Python
        python3 -c "import socket; s=socket.socket(); s.settimeout(5); exit(0 if s.connect_ex(('$(echo $DB_CHECK_HOST | cut -d' ' -f1)', $(echo $DB_CHECK_HOST | cut -d' ' -f2))) == 0 else 1)" 2>/dev/null
    fi
}

check_llm() {
    curl -sf "${LLM_BASE_URL}/models" --connect-timeout 5 &>/dev/null
}

echo "Waiting for dependencies to be ready..."
attempt=1

while true; do
    echo "Attempt $attempt of $( [ "$MAX_RETRIES" -eq 0 ] && echo 'unlimited' || echo "$MAX_RETRIES" )..."

    db_ok=false
    llm_ok=false

    if check_db; then
        echo "  [OK] Database accessible"
        db_ok=true
    else
        echo "  [..] Database not accessible"
    fi

    if check_llm; then
        echo "  [OK] LLM Studio accessible"
        llm_ok=true
    else
        echo "  [..] LLM Studio not accessible (${LLM_BASE_URL})"
    fi

    if $db_ok && $llm_ok; then
        echo ""
        echo "All dependencies ready! Starting Hindsight..."
        echo ""
        exec /app/start-all.sh
    fi

    if [ "$MAX_RETRIES" -ne 0 ] && [ "$attempt" -ge "$MAX_RETRIES" ]; then
        echo "Max retries reached. Exiting."
        exit 1
    fi

    echo "Waiting ${RETRY_INTERVAL}s before retry..."
    sleep "$RETRY_INTERVAL"
    ((attempt++))
done
