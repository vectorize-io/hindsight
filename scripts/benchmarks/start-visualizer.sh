#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

echo "🎨 Starting Benchmark Visualizer..."
echo ""
echo "Server will be available at: http://localhost:8001"
echo ""

uv run python hindsight-dev/benchmarks/visualizer/main.py
