#!/bin/bash
# Knowledge Pages API examples for Hindsight CLI
# Run: bash examples/api/knowledge-pages.sh

set -e

HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"
BANK_ID="knowledge-pages-demo-bank-cli"

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
hindsight bank create "$BANK_ID" --name "Knowledge Pages Demo"
hindsight memory retain "$BANK_ID" "The API is deployed to Kubernetes with a rolling update"
hindsight memory retain "$BANK_ID" "Deploys run from the main branch after CI passes"
hindsight memory retain "$BANK_ID" "A failed deploy is rolled back by redeploying the previous tag"
sleep 2

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:create-folder]
# Create a folder (omit --parent-id to create it at the root)
hindsight knowledge-base create-folder "$BANK_ID" "Operations"
# [/docs:create-folder]

FOLDER_ID=$(hindsight knowledge-base tree "$BANK_ID" -o json | jq -r '.roots[0].id')

# [docs:create-page]
# Create a page — content is generated in the background
hindsight knowledge-base create-page "$BANK_ID" \
  "Deploying the API" \
  "How is the API deployed?" \
  --parent-id "$FOLDER_ID" \
  --tags ops,type:runbook
# [/docs:create-page]

# Wait for the page's first build
sleep 20

# [docs:get-tree]
# Show the folder/page tree (no page bodies)
hindsight knowledge-base tree "$BANK_ID"
# [/docs:get-tree]

PAGE_ID=$(hindsight knowledge-base tree "$BANK_ID" -o json | jq -r '.roots[0].children[0].id')

# [docs:get-page]
# Read a page as a markdown document
hindsight knowledge-base get-page "$BANK_ID" "$PAGE_ID"
# [/docs:get-page]

# [docs:search-pages]
# Hybrid search (full-text + vector) over whole pages
hindsight knowledge-base search "$BANK_ID" "how do we deploy" --limit 5
# [/docs:search-pages]

# [docs:update-node]
# Rename a node, move it, and/or update a page's options.
# Changing --source-query rebuilds the page against the new question.
hindsight knowledge-base update "$BANK_ID" "$PAGE_ID" \
  --name "Deploying the API (v2)" \
  --tags ops,type:runbook,reviewed
# [/docs:update-node]

# [docs:export]
# Export the knowledge base as a portable markdown bundle
hindsight knowledge-base export "$BANK_ID"
# [/docs:export]

# [docs:delete-node]
# Delete a folder or page — deleting a folder removes its whole subtree
hindsight knowledge-base delete "$BANK_ID" "$FOLDER_ID" -y
# [/docs:delete-node]

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
curl -s -X DELETE "${HINDSIGHT_URL}/v1/default/banks/${BANK_ID}" > /dev/null

echo "knowledge-pages.sh: All examples passed"
