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
