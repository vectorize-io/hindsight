#!/usr/bin/env node
/**
 * Main Methods overview examples for Hindsight (Node.js)
 * Run: node examples/api/main-methods.mjs
 */
import { HindsightClient } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });

// =============================================================================
// Doc Examples - Retain Section
// =============================================================================

// [docs:main-retain]
// Store a single fact
await client.retain('my-bank', 'Alice joined Google in March 2024 as a Senior ML Engineer');

// Store a conversation
const conversation = `
User: What did you work on today?
Assistant: I reviewed the new ML pipeline architecture.
User: How did it look?
Assistant: Promising, but needs better error handling.
`;

await client.retain('my-bank', conversation, {
    context: 'Daily standup conversation'
});

// Batch retain multiple items
await client.retainBatch('my-bank', [
    { content: 'Bob prefers Python for data science' },
    { content: 'Alice recommends using pytest for testing' },
    { content: 'The team uses GitHub for code reviews' }
]);
// [/docs:main-retain]


// =============================================================================
// Doc Examples - Recall Section
// =============================================================================

// [docs:main-recall]
// Basic search
const results = await client.recall('my-bank', 'What does Alice do at Google?');

for (const result of results.results) {
    console.log(`- ${result.text}`);
}

// Search with options
const filteredResults = await client.recall('my-bank', 'What happened last spring?', {
    budget: 'high',
    maxTokens: 8192,
    types: ['world']
});

// Include entity information
const entityResults = await client.recall('my-bank', 'Tell me about Alice', {
    includeEntities: true,
    maxEntityTokens: 500
});

// Check entity details
for (const [entityId, entity] of Object.entries(entityResults.entities || {})) {
    console.log(`Entity: ${entity.canonical_name}`);
    console.log(`Observations: ${entity.observations}`);
}
// [/docs:main-recall]


// =============================================================================
// Doc Examples - Reflect Section
// =============================================================================

// [docs:main-reflect]
// Basic reflect
const response = await client.reflect('my-bank', 'Should we adopt TypeScript for our backend?');

console.log(response.text);
console.log('\nBased on:', (response.based_on || []).length, 'facts');

// Reflect with options
const detailedResponse = await client.reflect('my-bank', "What are Alice's strengths for the team lead role?", {
    budget: 'high'
});

// See which facts influenced the response
for (const fact of detailedResponse.based_on || []) {
    console.log(`- ${fact.text}`);
}
// [/docs:main-reflect]


// =============================================================================
// Doc Examples - List Memories Section
// =============================================================================

// [docs:main-list-memories]
// List all memories in a bank
const memories = await client.listMemories('my-bank', {
    limit: 10
});

for (const memory of memories.items) {
    console.log(`- [${memory.fact_type}] ${memory.text}`);
}

// Filter by type
const worldFacts = await client.listMemories('my-bank', {
    type: 'world',
    limit: 5
});

// Search within memories
const searchResults = await client.listMemories('my-bank', {
    q: 'Alice',
    limit: 10
});
// [/docs:main-list-memories]


// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await fetch(`${HINDSIGHT_URL}/v1/default/banks/my-bank`, { method: 'DELETE' });

console.log('main-methods.mjs: All examples passed');
