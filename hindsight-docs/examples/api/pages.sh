#!/bin/bash
# Pages API examples for Hindsight CLI
# Run: bash examples/api/pages.sh

set -e

HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"
BANK_ID="pages-demo-bank-cli"

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
hindsight bank create "$BANK_ID" --name "Pages Demo"
hindsight memory retain "$BANK_ID" "The API is deployed to Kubernetes with a rolling update"
hindsight memory retain "$BANK_ID" "Deploys run from the main branch after CI passes"
hindsight memory retain "$BANK_ID" "A failed deploy is rolled back by redeploying the previous tag"
sleep 2

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:create-folder]
# Create a folder (omit --parent-id to create it at the root)
hindsight pages create-folder "$BANK_ID" "Operations"
# [/docs:create-folder]

FOLDER_ID=$(hindsight pages tree "$BANK_ID" -o json | jq -r '.roots[0].id')

# [docs:create-page]
# Create a page — content is generated in the background
hindsight pages create-page "$BANK_ID" \
  "Deploying the API" \
  "How is the API deployed?" \
  --parent-id "$FOLDER_ID" \
  --tags ops,type:runbook
# [/docs:create-page]

# Wait for the page's first build
sleep 20

# [docs:get-tree]
# Show the folder/page tree (no page bodies)
hindsight pages tree "$BANK_ID"
# [/docs:get-tree]

PAGE_ID=$(hindsight pages tree "$BANK_ID" -o json | jq -r '.roots[0].children[0].id')

# [docs:get-page]
# Read a page as a markdown document
hindsight pages get-page "$BANK_ID" "$PAGE_ID"
# [/docs:get-page]

# [docs:search-pages]
# Hybrid search (full-text + vector) over whole pages
hindsight pages search "$BANK_ID" "how do we deploy" --limit 5
# [/docs:search-pages]

# [docs:filter-by-tags]
# Narrow the tree and search to pages carrying a tag (recall's tags / tags_match / tag_groups)
hindsight pages tree "$BANK_ID" --tags type:runbook --tags-match any_strict
hindsight pages search "$BANK_ID" "how do we deploy" --tags ops --tags-match all_strict

# Compound filters use --tag-groups: here, runbooks that are not drafts
hindsight pages search "$BANK_ID" "how do we deploy" \
    --tag-groups '[{"and":[{"tags":["type:runbook"]},{"not":{"tags":["draft"]}}]}]'
# [/docs:filter-by-tags]

# [docs:update-node]
# Rename a node, move it, and/or update a page's options.
# Changing --source-query rebuilds the page against the new question.
hindsight pages update "$BANK_ID" "$PAGE_ID" \
  --name "Deploying the API (v2)" \
  --tags ops,type:runbook,reviewed
# [/docs:update-node]

# [docs:export]
# Export the bank's pages as a portable markdown bundle
hindsight pages export "$BANK_ID"
# [/docs:export]

# [docs:delete-node]
# Delete a folder or page — deleting a folder removes its whole subtree
hindsight pages delete "$BANK_ID" "$FOLDER_ID" -y
# [/docs:delete-node]

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
curl -s -X DELETE "${HINDSIGHT_URL}/v1/default/banks/${BANK_ID}" > /dev/null

echo "pages.sh: All examples passed"
