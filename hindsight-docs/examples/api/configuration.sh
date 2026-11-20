#!/bin/bash
# Configuration examples for Hindsight (bank config over HTTP)
# Run: bash examples/api/configuration.sh

set -e

HINDSIGHT_API_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:plain-retrieval-bank]
curl -X PUT "$HINDSIGHT_API_URL/v1/default/banks/plain-retrieval-bank" \
  -H "Authorization: Bearer $HINDSIGHT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "retain_extraction_mode": "chunks",
    "enable_observations": false,
    "enable_temporal_retrieval": false,
    "enable_graph_retrieval": false,
    "enable_reranking": false
  }'
# [/docs:plain-retrieval-bank]
echo

curl -sf "$HINDSIGHT_API_URL/v1/default/banks/plain-retrieval-bank/config" \
  | jq -e '.overrides.retain_extraction_mode == "chunks" and .overrides.enable_reranking == false' > /dev/null

# [docs:mcp-tools-per-bank]
# Restrict a specific bank to read-only MCP access
curl -X PATCH http://localhost:8888/v1/default/banks/config-demo-bank/config \
  -H "Content-Type: application/json" \
  -d '{"updates": {"mcp_enabled_tools": ["recall"]}}'
# [/docs:mcp-tools-per-bank]
echo

curl -sf "$HINDSIGHT_API_URL/v1/default/banks/config-demo-bank/config" \
  | jq -e '.overrides.mcp_enabled_tools == ["recall"]' > /dev/null

# [docs:mcp-server-instructions-per-bank]
# Tell every MCP client that connects to this bank how to use it
curl -X PATCH http://localhost:8888/v1/default/banks/config-demo-bank/config \
  -H "Content-Type: application/json" \
  -d '{"updates": {"mcp_server_instructions": "Read the document house-rules before anything else. Save new facts with the tag inbox."}}'
# [/docs:mcp-server-instructions-per-bank]
echo

# A client connecting to the bank's endpoint receives the text in its initialize response.
curl -sf -X POST "$HINDSIGHT_API_URL/mcp/config-demo-bank/" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "docs-example", "version": "1"}}}' \
  | sed -n 's/^data: //p; /^{/p' \
  | jq -e '.result.instructions == "Read the document house-rules before anything else. Save new facts with the tag inbox."' > /dev/null

# [docs:bank-config-examples]
# Update retention settings for a bank
curl -X PATCH http://localhost:8888/v1/default/banks/config-demo-bank/config \
  -H "Content-Type: application/json" \
  -d '{
    "updates": {
      "retain_chunk_size": 4000,
      "retain_extraction_mode": "custom",
      "retain_custom_instructions": "Focus on technical details and implementation specifics"
    }
  }'

# Note: retain_extraction_mode must be "custom" to use retain_custom_instructions

# View resolved config (respects permissions)
curl http://localhost:8888/v1/default/banks/config-demo-bank/config

# Reset to defaults
curl -X DELETE http://localhost:8888/v1/default/banks/config-demo-bank/config
# [/docs:bank-config-examples]
echo

curl -sf "$HINDSIGHT_API_URL/v1/default/banks/config-demo-bank/config" \
  | jq -e '.overrides == {}' > /dev/null

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
for bank_id in plain-retrieval-bank config-demo-bank; do
  curl -s -X DELETE "${HINDSIGHT_API_URL}/v1/default/banks/${bank_id}" > /dev/null
done

echo "configuration.sh: All examples passed"
