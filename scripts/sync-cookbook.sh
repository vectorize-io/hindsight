#!/bin/bash
# Sync cookbook content from hindsight-cookbook repository
# Converts notebooks to markdown and updates the docs

set -e

cd "$(dirname "$0")/.."

echo "Syncing cookbook..."
uv run sync-cookbook

echo ""
echo "Done! Run 'npm run serve' to preview."
