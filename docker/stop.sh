#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "🛑 Stopping Services"
echo "============================"
echo ""

docker compose down

echo ""
echo "✅ All services stopped"
echo ""
echo "💡 To remove data volumes as well, run:"
echo "   docker compose down -v"
echo ""
