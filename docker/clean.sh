#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "🧹 Cleaning Services"
echo "============================"
echo ""
echo "This will:"
echo "  - Stop all services"
echo "  - Remove containers"
echo "  - Remove volumes (ALL DATA WILL BE LOST)"
echo ""
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🗑️  Removing services and data..."
docker compose down -v

echo ""
echo "✅ All services and data removed"
echo ""
