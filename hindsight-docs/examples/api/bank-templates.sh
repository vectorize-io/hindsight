#!/bin/bash
# Bank Templates API examples for Hindsight CLI
# Run: bash examples/api/bank-templates.sh

set -e

HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:import-template]
curl -X POST "$HINDSIGHT_URL/v1/default/banks/my-bank/import" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1",
    "bank": {
      "retain_mission": "Extract customer issues, resolutions, and sentiment.",
      "enable_observations": true,
      "observations_mission": "Track recurring customer pain points."
    },
    "mental_models": [
      {
        "id": "sentiment-overview",
        "name": "Customer Sentiment Overview",
        "source_query": "What is the overall sentiment trend?",
        "trigger": { "refresh_after_consolidation": true }
      }
    ],
    "directives": [
      {
        "name": "Acknowledge frustration",
        "content": "Always acknowledge frustration before offering solutions.",
        "priority": 10
      }
    ]
  }'
# [/docs:import-template]

# [docs:import-dry-run]
curl -X POST "$HINDSIGHT_URL/v1/default/banks/my-bank/import?dry_run=true" \
  -H "Content-Type: application/json" \
  -d '{"version": "1", "bank": {"retain_mission": "Dry run test."}}'
# [/docs:import-dry-run]

# [docs:export-template]
curl "$HINDSIGHT_URL/v1/default/banks/my-bank/export"
# [/docs:export-template]

# [docs:export-reimport]
# Export from source bank
curl "$HINDSIGHT_URL/v1/default/banks/source-bank/export" > template.json

# Import into a new bank
curl -X POST "$HINDSIGHT_URL/v1/default/banks/new-bank/import" \
  -H "Content-Type: application/json" \
  -d @template.json
# [/docs:export-reimport]

# [docs:get-schema]
curl "$HINDSIGHT_URL/v1/bank-template-schema"
# [/docs:get-schema]
