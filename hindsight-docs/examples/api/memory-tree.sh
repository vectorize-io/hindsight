#!/bin/bash
# Memory tree API examples for Hindsight (there is no CLI command yet, so curl)
# Run: bash examples/api/memory-tree.sh

set -e

HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"
BANK_ID="memory-tree-demo-bank-cli"
OUT_DIR=$(mktemp -d)

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
hindsight bank create "$BANK_ID" --name "Memory Tree Demo"
hindsight memory retain "$BANK_ID" "The API runs on Kubernetes behind an nginx ingress."
hindsight memory retain "$BANK_ID" "Redis keys expire after an hour."
hindsight memory retain "$BANK_ID" "The GPU nodes serve Gemma with vLLM."
sleep 5

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:get-tree]
# The whole bank: folders, topic files and each folder's index.md.
curl -s "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memory-tree" | jq '.observation_count'

# Only one scope: the same tag filters as recall.
curl -s "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memory-tree?tags=team:platform&tags_match=any_strict" \
  | jq '[.. | .files? // empty | .[] | .path]'
# [/docs:get-tree]

# [docs:write-bundle]
# Write the bundle to disk: each file's markdown at its path.
curl -s "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memory-tree" \
  | jq -c '.. | .files? // empty | .[] | {path, content}' \
  | while read -r file; do
      path="$OUT_DIR/$(jq -r .path <<<"$file")"
      mkdir -p "$(dirname "$path")"
      jq -r .content <<<"$file" > "$path"
    done
# [/docs:write-bundle]
test -f "$OUT_DIR/index.md"

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
curl -s -X DELETE "$HINDSIGHT_URL/v1/default/banks/$BANK_ID" > /dev/null

echo "memory-tree.sh: All examples passed"
