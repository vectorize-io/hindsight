#!/bin/bash

# Start the Hindsight documentation server
# This script starts a local Docusaurus development server for the documentation

set -e

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo "Starting documentation server..."
echo ""
echo "Starting Docusaurus development server..."
echo "Documentation will be available at: http://localhost:3000"
echo ""
npm run start -w hindsight-docs
