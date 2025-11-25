#!/bin/bash
set -e

cd "$(dirname "$0")"

SERVICE=$1

if [ -z "$SERVICE" ]; then
    echo "📋 Showing logs for all services..."
    echo ""
    docker compose logs -f
else
    echo "📋 Showing logs for $SERVICE..."
    echo ""
    docker compose logs -f "$SERVICE"
fi
