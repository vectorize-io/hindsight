#!/bin/bash
# Mental Models API examples for Hindsight CLI
# Run: bash examples/api/mental-models.sh

set -e

HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"
BANK_ID="mental-models-demo-bank"

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
hindsight bank create "$BANK_ID" --name "Mental Models Demo"
hindsight memory retain "$BANK_ID" "The team prefers async communication via Slack"
hindsight memory retain "$BANK_ID" "For urgent issues, use the #incidents channel"
hindsight memory retain "$BANK_ID" "Weekly syncs happen every Monday at 10am"
sleep 2

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:create-mental-model]
# Create a mental model (runs reflect in background)
hindsight mental-model create "$BANK_ID" \
  "Team Communication Preferences" \
  "How does the team prefer to communicate?"
# [/docs:create-mental-model]

# [docs:create-mental-model-with-id]
# Create a mental model with a specific custom ID
hindsight mental-model create "$BANK_ID" \
  "Communication Policy" \
  "What are the team's communication guidelines?" \
  --id communication-policy
# [/docs:create-mental-model-with-id]

sleep 5

# [docs:create-mental-model-with-trigger]
# Create a mental model and get its ID for subsequent operations
hindsight mental-model create "$BANK_ID" \
  "Project Status" \
  "What is the current project status?"
# [/docs:create-mental-model-with-trigger]

sleep 5

# [docs:list-mental-models]
# List all mental models in a bank
hindsight mental-model list "$BANK_ID"
# [/docs:list-mental-models]

# Get the first mental model ID for subsequent examples
MENTAL_MODEL_ID=$(hindsight mental-model list "$BANK_ID" -o json | python3 -c "import sys,json; items=json.load(sys.stdin).get('items',[]); print(items[0]['id'] if items else '')" 2>/dev/null || echo "")

if [ -n "$MENTAL_MODEL_ID" ]; then
  # [docs:get-mental-model]
  # Get a specific mental model
  hindsight mental-model get "$BANK_ID" "$MENTAL_MODEL_ID"
  # [/docs:get-mental-model]

  # [docs:refresh-mental-model]
  # Refresh a mental model to update with current knowledge
  hindsight mental-model refresh "$BANK_ID" "$MENTAL_MODEL_ID"
  # [/docs:refresh-mental-model]

  # [docs:clear-mental-model]
  # Clear a mental model's content, then refresh for a full re-synthesis
  curl -s -X POST "${HINDSIGHT_URL}/v1/default/banks/${BANK_ID}/mental-models/${MENTAL_MODEL_ID}/clear"

  # Trigger a fresh full rebuild
  hindsight mental-model refresh "$BANK_ID" "$MENTAL_MODEL_ID"
  # [/docs:clear-mental-model]

  # [docs:update-mental-model]
  # Update a mental model's metadata
  hindsight mental-model update "$BANK_ID" "$MENTAL_MODEL_ID" \
    --name "Updated Team Communication Preferences"
  # [/docs:update-mental-model]

  # [docs:get-mental-model-history]
  # Get the change history of a mental model
  hindsight mental-model history "$BANK_ID" "$MENTAL_MODEL_ID"
  # [/docs:get-mental-model-history]

  # [docs:delete-mental-model]
  # Delete a mental model
  hindsight mental-model delete "$BANK_ID" "$MENTAL_MODEL_ID" -y
  # [/docs:delete-mental-model]
fi

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
curl -s -X DELETE "${HINDSIGHT_URL}/v1/default/banks/${BANK_ID}" > /dev/null

echo "mental-models.sh: All examples passed"
