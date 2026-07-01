#!/bin/bash
# Hindsight API Startup Script
# Starts the Hindsight API server using the root .env configuration

set -e

# Change to the hindsight root directory
cd "$(dirname "$0")"

echo "════════════════════════════════════════════════════════════════"
echo "Starting Hindsight API"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  • Working Directory: $(pwd)"
echo "  • Environment File: .env"
echo "  • LLM Endpoint: $(grep HINDSIGHT_API_LLM_OLLAMA_API_BASE .env | grep -v '^#' | cut -d= -f2)"
echo "  • Embeddings Endpoint: $(grep HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE .env | grep -v '^#' | cut -d= -f2)"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Start the API
exec hindsight-api
