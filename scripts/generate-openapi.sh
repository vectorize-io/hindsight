#!/bin/bash
set -e

# Script to generate OpenAPI specification and update documentation
# This runs the generate-openapi command from hindsight-dev and regenerates docs

cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

echo "Generating OpenAPI specification..."
cd hindsight-dev
uv run generate-openapi

echo ""
echo "Building documentation..."
cd "$ROOT_DIR/hindsight-docs"
npm run build

echo ""
echo "OpenAPI spec and documentation generated successfully!"
