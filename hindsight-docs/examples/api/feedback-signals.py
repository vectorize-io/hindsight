"""
Feedback Signals API Examples
Track which recalled facts are useful to improve future retrieval.
"""

import os
from hindsight import Hindsight

# Initialize client
client = Hindsight(
    base_url=os.getenv("HINDSIGHT_URL", "http://localhost:8888"),
    bank_id="my-agent"
)

# region submit-signal
# Submit a single feedback signal
result = client.signal(
    signals=[
        {
            "fact_id": "abc-123-def-456",
            "signal_type": "used",
            "confidence": 1.0
        }
    ]
)
print(f"Processed: {result.signals_processed}")
# endregion submit-signal

# region submit-batch
# Submit multiple signals at once
result = client.signal(
    signals=[
        {"fact_id": "fact-1", "signal_type": "used", "confidence": 1.0},
        {"fact_id": "fact-2", "signal_type": "ignored", "confidence": 0.8},
        {"fact_id": "fact-3", "signal_type": "helpful", "confidence": 0.95},
        {"fact_id": "fact-4", "signal_type": "not_helpful", "confidence": 0.7},
    ]
)
print(f"Updated facts: {result.updated_facts}")
# endregion submit-batch

# region signal-with-context
# Submit signal with query context for pattern tracking
result = client.signal(
    signals=[
        {
            "fact_id": "abc-123",
            "signal_type": "helpful",
            "confidence": 0.9,
            "query": "authentication patterns",
            "context": "User found this answer helpful for implementing OAuth"
        }
    ]
)
# endregion signal-with-context

# region get-fact-stats
# Get usefulness statistics for a specific fact
stats = client.get_fact_stats(fact_id="abc-123-def-456")

if stats:
    print(f"Usefulness Score: {stats.usefulness_score}")
    print(f"Signal Count: {stats.signal_count}")
    print(f"Breakdown: {stats.signal_breakdown}")
    # Example output:
    # Usefulness Score: 0.72
    # Signal Count: 5
    # Breakdown: {'used': 3, 'ignored': 1, 'helpful': 1}
# endregion get-fact-stats

# region get-bank-stats
# Get aggregate usefulness stats for the entire bank
bank_stats = client.get_bank_usefulness_stats()

print(f"Total facts with signals: {bank_stats.total_facts_with_signals}")
print(f"Average usefulness: {bank_stats.average_usefulness}")
print(f"Signal distribution: {bank_stats.signal_distribution}")
print(f"Top useful facts: {bank_stats.top_useful_facts[:3]}")
# endregion get-bank-stats

# region recall-with-boost
# Recall with usefulness boosting enabled
results = client.recall(
    query="How do I implement authentication?",
    boost_by_usefulness=True,
    usefulness_weight=0.3,  # 30% usefulness, 70% semantic
    min_usefulness=0.0      # No minimum threshold
)

for fact in results:
    print(f"{fact.id}: {fact.text[:50]}...")
# endregion recall-with-boost

# region recall-filter-useful
# Recall only facts above a usefulness threshold
results = client.recall(
    query="API error handling patterns",
    boost_by_usefulness=True,
    usefulness_weight=0.5,
    min_usefulness=0.4  # Only facts with score >= 0.4
)
# endregion recall-filter-useful

# region feedback-loop
# Complete feedback loop example
# 1. Recall memories
recalled = client.recall(query="database connection pooling")
fact_ids = [r.id for r in recalled]

# 2. Agent uses some facts in its response
used_facts = ["fact-1", "fact-3"]  # Facts actually referenced
ignored_facts = [f for f in fact_ids if f not in used_facts]

# 3. Submit feedback
signals = [
    {"fact_id": fid, "signal_type": "used", "confidence": 1.0}
    for fid in used_facts
] + [
    {"fact_id": fid, "signal_type": "ignored", "confidence": 0.5}
    for fid in ignored_facts
]

client.signal(signals=signals)
# Future recalls will boost facts that are frequently used
# endregion feedback-loop
