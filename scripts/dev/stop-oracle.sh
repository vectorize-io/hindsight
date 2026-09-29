#!/usr/bin/env bash
# Stop and remove the local Oracle 23ai Free container started by start-oracle.sh.

set -euo pipefail

CONTAINER_NAME="hindsight-oracle"

if [ -n "$(docker ps -aq -f name="^${CONTAINER_NAME}$")" ]; then
    echo "→ Removing ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
    echo "✓ Stopped."
else
    echo "No ${CONTAINER_NAME} container running."
fi
