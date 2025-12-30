/**
 * Feedback Signals API Examples
 * Track which recalled facts are useful to improve future retrieval.
 */

import { HindsightClient } from "@anthropic-ai/hindsight";

const client = new HindsightClient({
  baseUrl: process.env.HINDSIGHT_URL || "http://localhost:8888",
  bankId: "my-agent"
});

// region submit-signal
// Submit a single feedback signal
const result = await client.signal({
  signals: [
    {
      fact_id: "abc-123-def-456",
      signal_type: "used",
      confidence: 1.0
    }
  ]
});
console.log(`Processed: ${result.signals_processed}`);
// endregion submit-signal

// region submit-batch
// Submit multiple signals at once
const batchResult = await client.signal({
  signals: [
    { fact_id: "fact-1", signal_type: "used", confidence: 1.0 },
    { fact_id: "fact-2", signal_type: "ignored", confidence: 0.8 },
    { fact_id: "fact-3", signal_type: "helpful", confidence: 0.95 },
    { fact_id: "fact-4", signal_type: "not_helpful", confidence: 0.7 },
  ]
});
console.log(`Updated facts: ${batchResult.updated_facts}`);
// endregion submit-batch

// region signal-with-context
// Submit signal with query context for pattern tracking
await client.signal({
  signals: [
    {
      fact_id: "abc-123",
      signal_type: "helpful",
      confidence: 0.9,
      query: "authentication patterns",
      context: "User found this answer helpful for implementing OAuth"
    }
  ]
});
// endregion signal-with-context

// region get-fact-stats
// Get usefulness statistics for a specific fact
const stats = await client.getFactStats("abc-123-def-456");

if (stats) {
  console.log(`Usefulness Score: ${stats.usefulness_score}`);
  console.log(`Signal Count: ${stats.signal_count}`);
  console.log(`Breakdown:`, stats.signal_breakdown);
  // Example output:
  // Usefulness Score: 0.72
  // Signal Count: 5
  // Breakdown: { used: 3, ignored: 1, helpful: 1 }
}
// endregion get-fact-stats

// region get-bank-stats
// Get aggregate usefulness stats for the entire bank
const bankStats = await client.getBankUsefulnessStats();

console.log(`Total facts with signals: ${bankStats.total_facts_with_signals}`);
console.log(`Average usefulness: ${bankStats.average_usefulness}`);
console.log(`Signal distribution:`, bankStats.signal_distribution);
console.log(`Top useful facts:`, bankStats.top_useful_facts.slice(0, 3));
// endregion get-bank-stats

// region recall-with-boost
// Recall with usefulness boosting enabled
const results = await client.recall({
  query: "How do I implement authentication?",
  boost_by_usefulness: true,
  usefulness_weight: 0.3,  // 30% usefulness, 70% semantic
  min_usefulness: 0.0      // No minimum threshold
});

for (const fact of results) {
  console.log(`${fact.id}: ${fact.text.slice(0, 50)}...`);
}
// endregion recall-with-boost

// region recall-filter-useful
// Recall only facts above a usefulness threshold
const filtered = await client.recall({
  query: "API error handling patterns",
  boost_by_usefulness: true,
  usefulness_weight: 0.5,
  min_usefulness: 0.4  // Only facts with score >= 0.4
});
// endregion recall-filter-useful

// region feedback-loop
// Complete feedback loop example
// 1. Recall memories
const recalled = await client.recall({ query: "database connection pooling" });
const factIds = recalled.map(r => r.id);

// 2. Agent uses some facts in its response
const usedFacts = ["fact-1", "fact-3"];  // Facts actually referenced
const ignoredFacts = factIds.filter(f => !usedFacts.includes(f));

// 3. Submit feedback
const signals = [
  ...usedFacts.map(fid => ({ fact_id: fid, signal_type: "used", confidence: 1.0 })),
  ...ignoredFacts.map(fid => ({ fact_id: fid, signal_type: "ignored", confidence: 0.5 }))
];

await client.signal({ signals });
// Future recalls will boost facts that are frequently used
// endregion feedback-loop
