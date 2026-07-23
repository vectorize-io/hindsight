#!/bin/bash
# Memories API examples for Hindsight CLI — read, list, and curate memory units.
# Run: bash examples/api/memories.sh

set -e

HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"
BANK_ID="memories-demo-bank"

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
hindsight bank create "$BANK_ID" --name "Memories Demo"
hindsight memory retain "$BANK_ID" "The assistant visited Paris in 2023."
hindsight memory retain "$BANK_ID" "The deploy server srv-04 runs PostgreSQL 14."
sleep 3

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:list-memories]
# List memory units in a bank (invalidated rows are included by default)
hindsight memory list "$BANK_ID"

# Filter to only the invalidated facts (e.g. to review duplicates)
curl -s "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/list?state=invalidated"
# [/docs:list-memories]

# Pick a raw fact (world/experience) to curate below.
MEMORY_ID=$(hindsight memory list "$BANK_ID" -o json | python3 -c "import sys,json; items=json.load(sys.stdin).get('items',[]); print(next((u['id'] for u in items if u['fact_type'] in ('world','experience')), ''))")

if [ -n "$MEMORY_ID" ]; then
  # [docs:get-memory]
  # Fetch a single memory unit (metadata, entities, dates, state)
  curl -s "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$MEMORY_ID"
  # [/docs:get-memory]

  # [docs:edit-memory]
  # Correct the fact's text. Re-embeds, drops derived observations/links,
  # re-consolidates, and recomputes the graph automatically.
  curl -s -X PATCH "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$MEMORY_ID" \
    -H "Content-Type: application/json" \
    -d '{"text": "The user visited Paris in 2023.", "reason": "wrong subject"}'
  # [/docs:edit-memory]

  # [docs:edit-memory-fields]
  # Correct dates, fact type, and entities in one call. "" clears a field;
  # entities replaces the set ([] detaches all); omit to leave unchanged.
  curl -s -X PATCH "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$MEMORY_ID" \
    -H "Content-Type: application/json" \
    -d '{"occurred_start": "2023-06-01", "fact_type": "experience", "entities": ["Alice", "Paris"]}'
  # [/docs:edit-memory-fields]

  # [docs:invalidate-memory]
  # Soft-retire a fact: removed from recall/consolidation/graph, links pruned,
  # derived observations recomputed without it — but kept for audit.
  curl -s -X PATCH "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$MEMORY_ID" \
    -H "Content-Type: application/json" \
    -d '{"state": "invalidated", "reason": "server decommissioned 2026-06-01"}'
  # [/docs:invalidate-memory]

  # [docs:restore-memory]
  # Restore a previously invalidated fact.
  curl -s -X PATCH "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$MEMORY_ID" \
    -H "Content-Type: application/json" \
    -d '{"state": "valid"}'
  # [/docs:restore-memory]
fi

# An observation (derived) exposes how it evolved as sources arrived.
OBSERVATION_ID=$(hindsight memory list "$BANK_ID" -o json | python3 -c "import sys,json; items=json.load(sys.stdin).get('items',[]); print(next((u['id'] for u in items if u['fact_type']=='observation'), ''))")
if [ -n "$OBSERVATION_ID" ]; then
  # [docs:observation-history]
  # Get the refresh history of a derived observation
  curl -s "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$OBSERVATION_ID/history"
  # [/docs:observation-history]
fi

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
curl -s -X DELETE "$HINDSIGHT_URL/v1/default/banks/$BANK_ID" > /dev/null

echo "memories.sh: All examples passed"
