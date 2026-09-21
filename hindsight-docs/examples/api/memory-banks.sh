#!/bin/bash
# Memory Banks API examples for Hindsight CLI
# Run: bash examples/api/memory-banks.sh

set -e

HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:create-bank]
hindsight bank create my-bank
# [/docs:create-bank]

# [docs:bank-with-disposition]
hindsight bank create architect-bank \
  --mission "You're a senior software architect - keep track of system designs, technology decisions, and architectural patterns. Prefer simplicity over cutting-edge." \
  --skepticism 4 \
  --literalism 4 \
  --empathy 2
# [/docs:bank-with-disposition]

# [docs:bank-background]
hindsight bank create my-bank \
  --mission "I am a research assistant specializing in machine learning."
# [/docs:bank-background]

# [docs:bank-mission]
hindsight bank create my-bank \
  --mission "You're a senior software architect - keep track of system designs, technology decisions, and architectural patterns."
# [/docs:bank-mission]

# [docs:bank-support-agent]
hindsight bank create support-bank
hindsight bank set-config support-bank \
  --observations-mission "I am a customer support agent. Track customer preferences, recurring issues, and resolution history."
# [/docs:bank-support-agent]

# [docs:update-bank-config]
hindsight bank set-config my-bank \
  --retain-mission "Always include technical decisions, API design choices, and architectural trade-offs. Ignore meeting logistics and social exchanges." \
  --retain-extraction-mode verbose \
  --observations-mission "Observations are stable facts about people and projects. Always include preferences, skills, and recurring patterns. Ignore one-off events." \
  --disposition-skepticism 4 \
  --disposition-literalism 4 \
  --disposition-empathy 2
# [/docs:update-bank-config]

# [docs:get-bank-config]
# Returns resolved config (server defaults merged with bank overrides)
hindsight bank config my-bank

# Show only bank-specific overrides
hindsight bank config my-bank --overrides-only
# [/docs:get-bank-config]

# [docs:reset-bank-config]
# Remove all bank-level overrides, reverting to server defaults
hindsight bank reset-config my-bank -y
# [/docs:reset-bank-config]

# [docs:prompts-preview]
curl --fail-with-body -X POST "$HINDSIGHT_URL/v1/default/banks/my-bank/prompts/preview" \
  -H "Content-Type: application/json" \
  -d '{"operation": "retain"}'
# [/docs:prompts-preview]
echo

# -----------------------------------------------------------------------------
# Transfer setup (not shown in docs): a source bank with two documents.
# Chunks mode stores text verbatim, so no LLM call is needed.
# -----------------------------------------------------------------------------
API_KEY="${HINDSIGHT_API_KEY:-}"
TRANSFER_BANKS="transfer-bank transfer-bank-copy transfer-other-bank transfer-bank-clone"
for bank_id in $TRANSFER_BANKS; do
  curl -s -X DELETE "${HINDSIGHT_URL}/v1/default/banks/${bank_id}" > /dev/null
done
hindsight bank create transfer-bank > /dev/null
hindsight bank set-config transfer-bank --retain-extraction-mode chunks > /dev/null
curl -sf -X POST "$HINDSIGHT_URL/v1/default/banks/transfer-bank/memories" \
  -H "Content-Type: application/json" \
  -d '{"items": [{"content": "Alice leads the payments team.", "document_id": "doc-1"},
                 {"content": "Bob moved to the Berlin office.", "document_id": "doc-2"}]}' > /dev/null
hindsight bank create transfer-other-bank > /dev/null

# Wait for an operation to finish; fail loudly if it does not complete.
wait_op() {
  for _ in $(seq 1 120); do
    status=$(curl -s "$HINDSIGHT_URL/v1/default/banks/$1/operations/$2" | jq -r .status)
    case "$status" in
      completed) return 0 ;;
      failed|cancelled) echo "operation $2 $status" >&2; return 1 ;;
    esac
    sleep 1
  done
  echo "operation $2 timed out" >&2
  return 1
}

# [docs:document-export]
# 1. Submit the export (whole bank; add ?document_id=… to scope it)
OPERATION_ID=$(curl -sf -X POST -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/document-transfer/export" | jq -r .operation_id)
# -> {"operation_id": "…", "status": "pending"}

# 2. Poll until completed
until [ "$(curl -s -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/operations/$OPERATION_ID" | jq -r .status)" = completed ]; do
  sleep 1
done
DOWNLOAD_URL=$(curl -s -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/operations/$OPERATION_ID" | jq -r .result_metadata.download_url)
# -> {"status":"completed","result_metadata":{
#      "download_url":"/v1/default/files/download/banks/transfer-bank/exports/…/transfer.zip",
#      "storage_key":"banks/transfer-bank/exports/…/transfer.zip","byte_size":12345,"filename":"transfer-bank-documents.zip"}}

# 3. Download the archive
curl -sf -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL$DOWNLOAD_URL" -o transfer-bank-documents.zip
# [/docs:document-export]
unzip -tq transfer-bank-documents.zip > /dev/null

# [docs:document-import]
OPERATION_ID=$(curl -sf -H "Authorization: Bearer $API_KEY" -F "file=@transfer-bank-documents.zip" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-other-bank/document-transfer?on_conflict=replace" | jq -r .operation_id)
# -> {"operation_id": "…", "status": "pending"}

curl --fail-with-body -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-other-bank/operations/$OPERATION_ID"
# -> {"status":"completed","result_metadata":{"documents_imported":3,"facts_imported":42,"observations_imported":5,...}}
# [/docs:document-import]
echo
wait_op transfer-other-bank "$OPERATION_ID"

# [docs:transfer-export]
# Whole bank, memories + config, no history
curl --fail-with-body -X POST -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/transfer/export"

# Just the memories
curl --fail-with-body -X POST -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/transfer/export?include_bank_config=false"

# Specific documents (a document subset carries no bank-level sections)
curl --fail-with-body -X POST -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/transfer/export?document_id=doc-1&document_id=doc-2&include_bank_config=false"
# [/docs:transfer-export]
echo

# Build the whole-bank archive the import examples use (not shown in docs):
# export, poll, download.
OPERATION_ID=$(curl -s -X POST "$HINDSIGHT_URL/v1/default/banks/transfer-bank/transfer/export" | jq -r .operation_id)
wait_op transfer-bank "$OPERATION_ID"
DOWNLOAD_URL=$(curl -s "$HINDSIGHT_URL/v1/default/banks/transfer-bank/operations/$OPERATION_ID" | jq -r .result_metadata.download_url)
curl -sf "$HINDSIGHT_URL$DOWNLOAD_URL" -o transfer-bank.zip

# [docs:transfer-import]
# Restore a bank under a new id
curl --fail-with-body -H "Authorization: Bearer $API_KEY" -F "file=@transfer-bank.zip" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/transfer/import?target_bank_id=transfer-bank-copy"

# Merge an archive's documents into an existing bank
curl --fail-with-body -H "Authorization: Bearer $API_KEY" -F "file=@transfer-bank.zip" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-other-bank/transfer/import?mode=merge&document_conflict=replace"
# [/docs:transfer-import]
echo

# Wait for both imports (the restore is recorded against the source bank).
for op in $(curl -s "$HINDSIGHT_URL/v1/default/banks/transfer-bank/operations?limit=50" \
    | jq -r '.operations[] | select(.status=="pending" or .status=="processing") | .id'); do
  wait_op transfer-bank "$op"
done
for op in $(curl -s "$HINDSIGHT_URL/v1/default/banks/transfer-other-bank/operations?limit=50" \
    | jq -r '.operations[] | select(.status=="pending" or .status=="processing") | .id'); do
  wait_op transfer-other-bank "$op"
done
curl -sf "$HINDSIGHT_URL/v1/default/banks/transfer-bank-copy/documents" | jq -e '.total == 2' > /dev/null

# [docs:clone-bank]
curl --fail-with-body -X POST -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/clone?target_bank_id=transfer-bank-clone"
# -> {"operation_id": "…", "status": "pending"}
# [/docs:clone-bank]
echo
for op in $(curl -s "$HINDSIGHT_URL/v1/default/banks/transfer-bank/operations?limit=50" \
    | jq -r '.operations[] | select(.status=="pending" or .status=="processing") | .id'); do
  wait_op transfer-bank "$op"
done
curl -sf "$HINDSIGHT_URL/v1/default/banks/transfer-bank-clone/documents" | jq -e '.total == 2' > /dev/null

rm -f transfer-bank.zip transfer-bank-documents.zip

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
for bank_id in my-bank architect-bank support-bank $TRANSFER_BANKS; do
  curl -s -X DELETE "${HINDSIGHT_URL}/v1/default/banks/${bank_id}" > /dev/null
done

echo "memory-banks.sh: All examples passed"
