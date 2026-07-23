#!/usr/bin/env node
/**
 * Memories API examples for Hindsight (Node.js) — read, list, and curate memory units.
 * Run: node examples/api/memories.mjs
 */
import { HindsightClient } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';
const BANK_ID = 'memories-demo-bank';

// PATCH isn't exposed on the high-level client yet, so curation goes through fetch.
const patchMemory = (memoryId, body) =>
    fetch(`${HINDSIGHT_URL}/v1/default/banks/${BANK_ID}/memories/${memoryId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });
await client.createBank(BANK_ID, { name: 'Memories Demo' });
await client.retain(BANK_ID, 'The assistant visited Paris in 2023.');
await client.retain(BANK_ID, 'The deploy server srv-04 runs PostgreSQL 14.');
await new Promise(r => setTimeout(r, 3000));

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:list-memories]
// List memory units in a bank. Invalidated rows are included by default.
const memories = await client.listMemories(BANK_ID);
for (const unit of memories.items) {
    console.log(`- [${unit.fact_type}] ${unit.text}`);
}

// Filter to only the invalidated facts (e.g. to review duplicates).
const invalidated = await client.listMemories(BANK_ID, { state: 'invalidated' });
console.log(`${invalidated.items.length} invalidated fact(s)`);
// [/docs:list-memories]

// Pick a raw fact (world/experience) to curate below.
const fact = memories.items.find(u => u.fact_type === 'world' || u.fact_type === 'experience');
if (!fact) {
    await client.deleteBank(BANK_ID);
    console.log('memories.mjs: All examples passed (no facts extracted yet)');
    process.exit(0);
}
const memoryId = fact.id;

// [docs:get-memory]
// Fetch a single memory unit (metadata, entities, dates, state).
const memory = await (
    await fetch(`${HINDSIGHT_URL}/v1/default/banks/${BANK_ID}/memories/${memoryId}`)
).json();
console.log(`Text: ${memory.text}`);
console.log(`Type: ${memory.type}  Entities: ${memory.entities}`);
// [/docs:get-memory]

// [docs:edit-memory]
// Correct the fact's text. Re-embeds, drops derived observations/links,
// re-consolidates, and recomputes the graph automatically.
await patchMemory(memoryId, { text: 'The user visited Paris in 2023.', reason: 'wrong subject' });
// [/docs:edit-memory]

// [docs:edit-memory-fields]
// Correct dates, fact type, and entities in one call. "" clears a field;
// entities replaces the set ([] detaches all); omit to leave unchanged.
await patchMemory(memoryId, {
    occurred_start: '2023-06-01',
    fact_type: 'experience',
    entities: ['Alice', 'Paris'],
});
// [/docs:edit-memory-fields]

// [docs:invalidate-memory]
// Soft-retire a fact: removed from recall/consolidation/graph, links pruned,
// derived observations recomputed without it — but kept for audit.
await patchMemory(memoryId, { state: 'invalidated', reason: 'server decommissioned 2026-06-01' });
// [/docs:invalidate-memory]

// [docs:restore-memory]
// Restore a previously invalidated fact.
await patchMemory(memoryId, { state: 'valid' });
// [/docs:restore-memory]

// An observation (derived) exposes how it evolved as sources arrived.
const observation = memories.items.find(u => u.fact_type === 'observation');
if (observation) {
    // [docs:observation-history]
    // Get the refresh history of a derived observation.
    const history = await (
        await fetch(`${HINDSIGHT_URL}/v1/default/banks/${BANK_ID}/memories/${observation.id}/history`)
    ).json();
    console.log(`Observation history entries: ${history.length}`);
    // [/docs:observation-history]
}

// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await client.deleteBank(BANK_ID);

console.log('memories.mjs: All examples passed');
