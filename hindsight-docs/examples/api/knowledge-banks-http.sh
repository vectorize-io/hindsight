#!/bin/bash
# Knowledge Banks API examples (curl)
# Run: bash examples/api/knowledge-banks-http.sh

set -e

HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The upload example reads its files from the current directory
WORKDIR="$(mktemp -d)"
cd "$WORKDIR"
cp "$SCRIPT_DIR/sample.pdf" x200-manual.pdf
printf '# X200 quick start\n\nFill the tank, then hold the power button for two seconds.\n' > x200-quickstart.md

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:create-knowledge-bank]
curl -X POST "$HINDSIGHT_URL/v1/default/knowledge-banks" \
  -H "Content-Type: application/json" \
  -d '{"bank_id": "product-manuals"}'
# [/docs:create-knowledge-bank]

# [docs:ingest-files]
curl -X POST "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/files/ingest" \
  -F "files=@x200-manual.pdf" \
  -F "files=@x200-quickstart.md" \
  -F 'request={"files": [
        {"document_id": "x200-manual", "tags": ["x200"]},
        {"document_id": "x200-quickstart", "tags": ["x200", "quickstart"]}
      ]}'
# [/docs:ingest-files]

# [docs:ingest-text]
curl -X POST "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "document_id": "returns-policy",
        "text": "# Returns\n\nYou can return an X200 within 30 days...",
        "tags": ["policy"]
      }
    ]
  }'
# [/docs:ingest-text]

# [docs:search]
curl -X POST "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "How long do I have to return an X200?", "max_tokens": 2048}'
# [/docs:search]

# [docs:list-tags]
curl "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/tags?q=x2*"
# [/docs:list-tags]

# [docs:link-memory-bank]
curl -X PATCH "$HINDSIGHT_URL/v1/default/banks/support-agent/config" \
  -H "Content-Type: application/json" \
  -d '{"updates": {"knowledge_banks": ["product-manuals"]}}'
# [/docs:link-memory-bank]

# [docs:bank-config]
curl -X PATCH "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/config" \
  -H "Content-Type: application/json" \
  -d '{"updates": {"text_passage_tokens": 256, "disabled_media_types": ["video"]}}'
# [/docs:bank-config]

# =============================================================================
# Checks (not shown in docs): the calls above must have worked, not just answered
# =============================================================================
# The text ingest returns once the document is searchable, so the search must find it.
curl -sf -X POST "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "return an X200 within 30 days"}' | grep -q '"returns-policy"'
curl -sf "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/tags" | grep -q '"policy"'
curl -sf "$HINDSIGHT_URL/v1/default/banks/support-agent/config" | grep -q '"product-manuals"'
curl -sf "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/config" | grep -q '"text_passage_tokens": *256'
# The file ingest runs in the background: wait for the quick start to be ready.
for _ in $(seq 1 120); do
  curl -sf "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/documents/x200-quickstart" \
    | grep -qE '"status": *"(ready|failed)"' && break
  sleep 1
done
curl -sf "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/documents/x200-quickstart" | grep -qE '"status": *"ready"'

# [docs:documents]
curl "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/documents?status=ready"
curl "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/documents/x200-quickstart"
curl -o manual-copy.pdf "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/documents/x200-manual/file"
# [/docs:documents]
curl -sf "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/documents?status=ready" | grep -q '"x200-quickstart"'

# [docs:settings]
# Every setting as the bank uses it, then only what the bank changes
curl "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/config"

# Drop one override, or all of them
curl -X PATCH "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/config" \
  -H "Content-Type: application/json" \
  -d '{"updates": {"disabled_media_types": null}}'
curl -X DELETE "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/config"
# [/docs:settings]
if curl -sf "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/config" | grep -q '"text_passage_tokens": *256'; then
  echo "override not reset" >&2; exit 1
fi

# [docs:delete-document]
curl -X DELETE "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/documents/returns-policy"
# [/docs:delete-document]

# [docs:delete-knowledge-bank]
curl -X DELETE "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals"
# [/docs:delete-knowledge-bank]

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
rm -rf "$WORKDIR"
curl -s -X DELETE "$HINDSIGHT_URL/v1/default/banks/support-agent" > /dev/null

echo "knowledge-banks-http.sh: All examples passed"
