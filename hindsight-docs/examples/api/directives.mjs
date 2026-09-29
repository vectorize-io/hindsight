#!/usr/bin/env node
/**
 * Directives API examples for Hindsight (Node.js)
 * Run: node examples/api/directives.mjs
 */
import { HindsightClient } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';
const BANK_ID = 'directives-example-bank';

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });
await client.createBank(BANK_ID, { name: 'Test Bank' });

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:create-directive]
// Create a directive (hard rule for reflect)
const directive = await client.createDirective(
    BANK_ID,
    'Formal Language',
    'Always respond in formal English, avoiding slang and colloquialisms.'
);

console.log(`Created directive: ${directive.id}`);
// [/docs:create-directive]

const directiveId = directive.id;

// [docs:list-directives]
// List all directives in a bank
const directives = await client.listDirectives(BANK_ID);

for (const d of directives.items) {
    console.log(`- ${d.name}: ${d.content.slice(0, 50)}...`);
}
// [/docs:list-directives]

// [docs:update-directive]
// Update a directive (e.g., disable without deleting)
const updated = await client.updateDirective(BANK_ID, directiveId, {
    isActive: false
});

console.log(`Directive active: ${updated.is_active}`);
// [/docs:update-directive]

// [docs:delete-directive]
// Delete a directive
await client.deleteDirective(BANK_ID, directiveId);
// [/docs:delete-directive]

// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await client.deleteBank(BANK_ID);

console.log('directives.mjs: All examples passed');
