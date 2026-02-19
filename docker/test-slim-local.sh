#!/bin/bash
#
# Local Test Script for Slim Docker Images
#
# This script makes it easy to test slim images locally with external providers.
# It expects API keys to be set in environment variables.
#
# Usage:
#   export OPENAI_API_KEY=sk-xxx
#   export COHERE_API_KEY=xxx
#   ./docker/test-slim-local.sh
#
# Or inline:
#   OPENAI_API_KEY=sk_xxx COHERE_API_KEY=xxx ./docker/test-slim-local.sh
#

set -euo pipefail

# Check for required API keys
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable is required"
    echo "Set it with: export OPENAI_API_KEY=sk-xxx"
    exit 1
fi

if [ -z "${COHERE_API_KEY:-}" ]; then
    echo "❌ Error: COHERE_API_KEY environment variable is required"
    echo "Set it with: export COHERE_API_KEY=xxx"
    exit 1
fi

# Configuration
IMAGE="${1:-hindsight-slim:test}"
echo "Testing image: $IMAGE"
echo ""

# Set up LLM and external providers
export HINDSIGHT_API_LLM_PROVIDER=openai
export HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY
export HINDSIGHT_API_LLM_MODEL=gpt-4o-mini
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
export HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=$OPENAI_API_KEY
export HINDSIGHT_API_RERANKER_PROVIDER=cohere
export HINDSIGHT_API_COHERE_API_KEY=$COHERE_API_KEY

# Run the test
exec "$(dirname "$0")/test-image.sh" "$IMAGE" standalone
