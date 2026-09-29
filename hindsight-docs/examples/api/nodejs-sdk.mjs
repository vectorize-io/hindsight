#!/usr/bin/env node
/**
 * TypeScript / JavaScript SDK page examples for Hindsight (docs/sdks/nodejs.mdx).
 * Run: node examples/api/nodejs-sdk.mjs
 *
 * The quickstart and client-init snippets both open with the same import. An ES
 * module can import a name only once, so both sections share the import line
 * (one marker line opens/closes both), and each later section sits in its own
 * { } block so repeated `const` names don't clash.
 */
const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:quickstart] // [docs:client-init]
import { HindsightClient } from '@vectorize-io/hindsight-client';

// [/docs:quickstart] // [/docs:client-init]

// [docs:client-init]
const client = new HindsightClient({
    baseUrl: 'http://localhost:8888',
});
// [/docs:client-init]

{
// [docs:quickstart]
const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

// Retain a memory
await client.retain('nodejs-sdk-bank', 'Alice works at Google');

// Recall memories
const response = await client.recall('nodejs-sdk-bank', 'What does Alice do?');
for (const r of response.results) {
    console.log(r.text);
}

// Reflect - generate response with disposition
const answer = await client.reflect('nodejs-sdk-bank', 'Tell me about Alice');
console.log(answer.text);
// [/docs:quickstart]
}

{
// [docs:get-version]
const version = await client.getVersion();

console.log(version.api_version);

if (!version.features.mcp) {
    throw new Error('This server does not expose the MCP endpoint');
}
// [/docs:get-version]
}

{
// [docs:retain]
// Simple
await client.retain('nodejs-sdk-bank', 'Alice works at Google');

// With options
await client.retain('nodejs-sdk-bank', 'Alice got promoted', {
    timestamp: new Date('2024-01-15'),
    context: 'career update',
    metadata: { source: 'slack' },
    async: false,  // Set true for background processing
});
// [/docs:retain]
}

{
// [docs:retain-batch]
await client.retainBatch('nodejs-sdk-bank', [
    { content: 'Alice works at Google', context: 'career' },
    { content: 'Bob is a data scientist', context: 'career' },
], {
    async: false,
});
// [/docs:retain-batch]
}

{
// [docs:recall]
// Simple - returns RecallResponse
const response = await client.recall('nodejs-sdk-bank', 'What does Alice do?');

for (const r of response.results) {
    console.log(`${r.text} (type: ${r.type})`);
}

// With options
const filtered = await client.recall('nodejs-sdk-bank', 'What does Alice do?', {
    types: ['world', 'observation'],  // Filter by fact type
    maxTokens: 4096,
    budget: 'high',  // 'low', 'mid', or 'high'
});
// [/docs:recall]
}

{
// [docs:reflect]
const answer = await client.reflect('nodejs-sdk-bank', 'What should I know about Alice?', {
    budget: 'low',  // 'low', 'mid', or 'high'
    context: 'preparing for a meeting',
});

console.log(answer.text);       // Generated response
// [/docs:reflect]
}

{
// [docs:create-bank]
await client.createBank('nodejs-sdk-bank', {
    name: 'Assistant',
    mission: "You're a helpful AI assistant - keep track of user preferences and conversation history.",
    disposition: {
        skepticism: 3,   // 1-5: trusting to skeptical
        literalism: 3,   // 1-5: flexible to literal
        empathy: 3,      // 1-5: detached to empathetic
    },
});
// [/docs:create-bank]
}

{
// [docs:list-memories]
const response = await client.listMemories('nodejs-sdk-bank', {
    type: 'world',  // Optional filter
    q: 'Alice',     // Optional text search
    limit: 100,
    offset: 0,
});
console.log(response);
// [/docs:list-memories]
}

// Setup (not shown in docs): a document for the document examples below
await client.retain('nodejs-sdk-bank', 'Alice and Bob met to plan the launch', {
    documentId: 'conversation_001',
});

{
// [docs:get-document]
const doc = await client.getDocument('nodejs-sdk-bank', 'conversation_001');
if (doc) {
    console.log(doc);  // null when document not found
}
// [/docs:get-document]
if (!doc) throw new Error('expected conversation_001 to exist');
}

{
// [docs:list-documents]
const response = await client.listDocuments('nodejs-sdk-bank', {
    limit: 50,
    offset: 0,
});
console.log(response);
// [/docs:list-documents]
}

// [docs:update-document]
await client.updateDocument('nodejs-sdk-bank', 'conversation_001', {
    tags: ['important', 'meeting-notes'],
});
// [/docs:update-document]

// [docs:delete-document]
await client.deleteDocument('nodejs-sdk-bank', 'conversation_001');
// [/docs:delete-document]

// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await fetch(`${HINDSIGHT_URL}/v1/default/banks/nodejs-sdk-bank`, { method: 'DELETE' });

console.log('nodejs-sdk.mjs: All examples passed');
