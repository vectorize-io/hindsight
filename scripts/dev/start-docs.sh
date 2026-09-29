#!/bin/bash

# Start the Hindsight documentation server
# This script starts a local Docusaurus development server for the documentation

set -e

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Always show the "current" (Next / unreleased) docs version in local dev.
# docusaurus.config.ts reads this flag to decide which versions to include.
# Do NOT rely on NODE_ENV for this — it's unreliable across hot-reload paths.
export INCLUDE_CURRENT_VERSION=true

echo "Starting documentation server..."
echo ""
echo "Starting Docusaurus development server..."
echo "Documentation will be available at: http://localhost:3000"
echo "INCLUDE_CURRENT_VERSION=true (Next/unreleased docs visible)"
echo ""
npm run start -w hindsight-docs -- --no-open
