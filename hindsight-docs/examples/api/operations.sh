#!/bin/bash
# Operations API examples for Hindsight CLI
# Run: bash examples/api/operations.sh

set -e

BANK_ID="my-bank"

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
# Submit an async retain so we have a real pending operation_id to exercise
# get / cancel below.
OPERATION_ID=$(
  hindsight memory retain "$BANK_ID" "setup-1 for ops examples" --async -o json \
    | jq -r '.operation_id'
)


# =============================================================================
# Doc Examples
# =============================================================================

# [docs:operations-list]
hindsight operation list my-bank
# [/docs:operations-list]


# [docs:operations-get]
hindsight operation get my-bank "$OPERATION_ID"
# [/docs:operations-get]


# [docs:operations-cancel]
hindsight operation cancel my-bank "$OPERATION_ID"
# [/docs:operations-cancel]


# Setup for retry: create another op and cancel it so it's in a retryable state.
OPERATION_ID=$(
  hindsight memory retain "$BANK_ID" "setup-2 for retry example" --async -o json \
    | jq -r '.operation_id'
)
hindsight operation cancel "$BANK_ID" "$OPERATION_ID" >/dev/null


# [docs:operations-retry]
hindsight operation retry my-bank "$OPERATION_ID"
# [/docs:operations-retry]


# [docs:operations-async-retain]
# Submit an async retain and capture the operation_id from the JSON response.
OPERATION_ID=$(
  hindsight memory retain my-bank "Alice joined Google in 2023" --async -o json \
    | jq -r '.operation_id'
)

# Poll until the worker finishes — completed/failed/cancelled are all terminal.
while true; do
  STATUS=$(hindsight operation get my-bank "$OPERATION_ID" -o json | jq -r '.status')
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ] || [ "$STATUS" = "cancelled" ]; then
    echo "finished: $STATUS"
    break
  fi
  sleep 2
done
# [/docs:operations-async-retain]
