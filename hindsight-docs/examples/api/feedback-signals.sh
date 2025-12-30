#!/bin/bash
# Feedback Signals API Examples
# Track which recalled facts are useful to improve future retrieval.

HINDSIGHT_URL="${HINDSIGHT_URL:-http://localhost:8888}"
BANK_ID="my-agent"

# region submit-signal
# Submit a single feedback signal
curl -X POST "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/signal" \
  -H "Content-Type: application/json" \
  -d '{
    "signals": [
      {
        "fact_id": "abc-123-def-456",
        "signal_type": "used",
        "confidence": 1.0
      }
    ]
  }'
# endregion submit-signal

# region submit-batch
# Submit multiple signals at once
curl -X POST "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/signal" \
  -H "Content-Type: application/json" \
  -d '{
    "signals": [
      {"fact_id": "fact-1", "signal_type": "used", "confidence": 1.0},
      {"fact_id": "fact-2", "signal_type": "ignored", "confidence": 0.8},
      {"fact_id": "fact-3", "signal_type": "helpful", "confidence": 0.95},
      {"fact_id": "fact-4", "signal_type": "not_helpful", "confidence": 0.7}
    ]
  }'
# endregion submit-batch

# region signal-with-context
# Submit signal with query context for pattern tracking
curl -X POST "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/signal" \
  -H "Content-Type: application/json" \
  -d '{
    "signals": [
      {
        "fact_id": "abc-123",
        "signal_type": "helpful",
        "confidence": 0.9,
        "query": "authentication patterns",
        "context": "User found this answer helpful"
      }
    ]
  }'
# endregion signal-with-context

# region get-fact-stats
# Get usefulness statistics for a specific fact
curl "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/facts/abc-123-def-456/stats"

# Example response:
# {
#   "fact_id": "abc-123-def-456",
#   "usefulness_score": 0.72,
#   "signal_count": 5,
#   "signal_breakdown": {"used": 3, "ignored": 1, "helpful": 1},
#   "last_signal_at": "2025-01-15T10:30:00Z",
#   "created_at": "2025-01-10T08:00:00Z"
# }
# endregion get-fact-stats

# region get-bank-stats
# Get aggregate usefulness stats for the entire bank
curl "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/stats/usefulness"

# Example response:
# {
#   "bank_id": "my-agent",
#   "total_facts_with_signals": 150,
#   "average_usefulness": 0.65,
#   "total_signals": 423,
#   "signal_distribution": {"used": 200, "ignored": 150, "helpful": 50, "not_helpful": 23},
#   "top_useful_facts": [...],
#   "least_useful_facts": [...]
# }
# endregion get-bank-stats

# region recall-with-boost
# Recall with usefulness boosting enabled
curl -X POST "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/recall" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I implement authentication?",
    "boost_by_usefulness": true,
    "usefulness_weight": 0.3,
    "min_usefulness": 0.0
  }'
# endregion recall-with-boost

# region recall-filter-useful
# Recall only facts above a usefulness threshold
curl -X POST "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/recall" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "API error handling patterns",
    "boost_by_usefulness": true,
    "usefulness_weight": 0.5,
    "min_usefulness": 0.4
  }'
# endregion recall-filter-useful
