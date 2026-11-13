#!/bin/bash
# Observation tree examples for the Hindsight CLI
# Run: bash examples/api/observation-tree.sh

set -e

HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"
BANK_ID="observation-tree-demo-bank-cli"
OUT_DIR=$(mktemp -d)

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
hindsight bank create "$BANK_ID" --name "Observation Tree Demo"
# Item-level tags (the CLI's retain has only the deprecated document tags), so the scoped
# call below has something to show.
for item in \
  '{"content": "The API runs on Kubernetes behind an nginx ingress.", "tags": ["team:platform"]}' \
  '{"content": "Redis keys expire after an hour.", "tags": ["team:platform"]}' \
  '{"content": "The GPU nodes serve Gemma with vLLM.", "tags": ["team:ml"]}'; do
  curl -s -X POST "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories" \
    -H "Content-Type: application/json" -d "{\"items\": [$item]}" > /dev/null
done
sleep 5

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:get-tree]
# The whole bank: folders, topic files and each folder's index.md (-o json for the full tree).
hindsight memory observation-tree "$BANK_ID"

# Only one scope: the same tag filters as recall.
hindsight memory observation-tree "$BANK_ID" --tags team:platform --tags-match any_strict
# [/docs:get-tree]

# [docs:write-bundle]
# Write the bundle to disk: each file's markdown at its path.
hindsight memory observation-tree "$BANK_ID" --out "$OUT_DIR"
# [/docs:write-bundle]
test -f "$OUT_DIR/index.md"

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
curl -s -X DELETE "$HINDSIGHT_URL/v1/default/banks/$BANK_ID" > /dev/null

echo "observation-tree.sh: All examples passed"
