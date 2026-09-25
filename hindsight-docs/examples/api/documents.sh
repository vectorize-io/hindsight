#!/bin/bash
# Documents API examples for Hindsight CLI
# Run: bash examples/api/documents.sh

set -e

HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"
BANK_ID="documents-sh-demo-bank"

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
WORKDIR=$(mktemp -d)
cd "$WORKDIR"
mkdir docs
echo "Alice presented the Q4 roadmap to the whole team." > docs/meeting-notes.txt

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:document-retain]
# Retain content with document ID
hindsight memory retain "$BANK_ID" "Meeting notes content..." --doc-id notes-2024-03-15

# Batch retain from files
hindsight memory retain-files "$BANK_ID" docs/
# [/docs:document-retain]

# [docs:document-update]
# Original
hindsight memory retain "$BANK_ID" "Project deadline: March 31" --doc-id project-plan

# Update
hindsight memory retain "$BANK_ID" "Project deadline: April 15 (extended)" --doc-id project-plan
# [/docs:document-update]

# [docs:document-get]
hindsight document get "$BANK_ID" notes-2024-03-15
# [/docs:document-get]

# [docs:document-update-tags]
# Replace tags with new values (comma-separated)
hindsight document update "$BANK_ID" notes-2024-03-15 --tags team-a,team-b

# Remove all tags (make document visible everywhere)
hindsight document update "$BANK_ID" notes-2024-03-15 --tags ""
# [/docs:document-update-tags]

# The empty value must clear the set, not store one "" tag.
TAGS=$(curl -sf "${HINDSIGHT_URL}/v1/default/banks/${BANK_ID}/documents/notes-2024-03-15" | jq -c '.tags')
[ "$TAGS" = "[]" ] || { echo "expected no tags, got $TAGS"; exit 1; }

# [docs:document-list]
# List all documents
hindsight document list "$BANK_ID"

# Filter by ID substring
hindsight document list "$BANK_ID" --query notes
# [/docs:document-list]

# [docs:document-delete]
hindsight document delete "$BANK_ID" notes-2024-03-15
# [/docs:document-delete]

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
rm -rf "$WORKDIR"
curl -s -X DELETE "${HINDSIGHT_URL}/v1/default/banks/${BANK_ID}" > /dev/null

echo "documents.sh: All examples passed"
