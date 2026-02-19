#!/usr/bin/env node
/**
 * Recall API examples for Hindsight (Node.js)
 * Run: node examples/api/recall.mjs
 */
import { HindsightClient } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });

// Seed some data for recall examples
await client.retain('my-bank', 'Alice works at Google as a software engineer');
await client.retain('my-bank', 'Alice loves hiking on weekends');
await client.retain('my-bank', 'Bob is a data scientist who works with Alice');

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:recall-basic]
const response = await client.recall('my-bank', 'What does Alice do?');
for (const r of response.results) {
    console.log(`${r.text} (score: ${r.weight})`);
}
// [/docs:recall-basic]


// [docs:recall-with-options]
const detailedResponse = await client.recall('my-bank', 'What does Alice do?', {
    types: ['world', 'experience'],
    budget: 'high',
    maxTokens: 8000,
    trace: true
});

// Access results
for (const r of detailedResponse.results) {
    console.log(`${r.text} (score: ${r.weight})`);
}
// [/docs:recall-with-options]


// [docs:recall-source-facts]
// Recall observations and include their source facts
const obsResponse = await client.recall('my-bank', 'What patterns have I learned about Alice?', {
    types: ['observation'],
    includeSourceFacts: true,
    maxSourceFactsTokens: 4096,
});

for (const obs of obsResponse.results) {
    console.log(`Observation: ${obs.text}`);
    if (obs.source_fact_ids && obsResponse.source_facts) {
        console.log('  Derived from:');
        for (const factId of obs.source_fact_ids) {
            const fact = obsResponse.source_facts[factId];
            if (fact) console.log(`    - [${fact.type}] ${fact.text}`);
        }
    }
}
// [/docs:recall-source-facts]


// [docs:recall-budget-levels]
// Quick lookup
const quickResults = await client.recall('my-bank', "Alice's email", { budget: 'low' });

// Deep exploration
const deepResults = await client.recall('my-bank', 'How are Alice and Bob connected?', { budget: 'high' });
// [/docs:recall-budget-levels]


// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await fetch(`${HINDSIGHT_URL}/v1/default/banks/my-bank`, { method: 'DELETE' });

console.log('recall.mjs: All examples passed');
