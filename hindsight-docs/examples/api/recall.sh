#!/bin/bash
# Recall API examples for Hindsight CLI
# Run: bash examples/api/recall.sh

set -e

HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
hindsight memory retain my-bank "Alice works at Google as a software engineer"
hindsight memory retain my-bank "Alice loves hiking on weekends"

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:recall-basic]
hindsight memory recall my-bank "What does Alice do?"
# [/docs:recall-basic]


# [docs:recall-with-options]
hindsight memory recall my-bank "hiking recommendations" \
  --budget high \
  --max-tokens 8192
# [/docs:recall-with-options]


# [docs:recall-fact-type]
hindsight memory recall my-bank "query" --fact-type world,observation
# [/docs:recall-fact-type]


# [docs:recall-trace]
hindsight memory recall my-bank "query" --trace
# [/docs:recall-trace]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
curl -s -X DELETE "${HINDSIGHT_URL}/v1/default/banks/my-bank" > /dev/null

echo "recall.sh: All examples passed"
