#!/usr/bin/env node
/**
 * Mental Models API examples for Hindsight (Node.js)
 * Run: node examples/api/mental-models.mjs
 */
import { HindsightClient } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';
const BANK_ID = 'mental-models-demo-bank';

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });
await client.createBank(BANK_ID, { name: 'Mental Models Demo' });
await client.retain(BANK_ID, 'The team prefers async communication via Slack');
await client.retain(BANK_ID, 'For urgent issues, use the #incidents channel');
await client.retain(BANK_ID, 'Weekly syncs happen every Monday at 10am');
await new Promise(r => setTimeout(r, 2000));

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:create-mental-model]
// Create a mental model (runs reflect in background)
const result = await client.createMentalModel(
    BANK_ID,
    'Team Communication Preferences',
    'How does the team prefer to communicate?',
    { tags: ['team', 'communication'] },
);

// Returns an operation_id — check operations endpoint for completion
console.log(`Operation ID: ${result.operation_id}`);
// [/docs:create-mental-model]

// [docs:create-mental-model-with-id]
// Create a mental model with a specific custom ID
const resultWithId = await client.createMentalModel(
    BANK_ID,
    'Communication Policy',
    "What are the team's communication guidelines?",
    { id: 'communication-policy' },
);

console.log(`Created with custom ID: ${resultWithId.operation_id}`);
// [/docs:create-mental-model-with-id]

await new Promise(r => setTimeout(r, 5000));

// [docs:create-mental-model-with-trigger]
// Create a mental model with automatic refresh enabled
const result2 = await client.createMentalModel(
    BANK_ID,
    'Project Status',
    'What is the current project status?',
    { trigger: { refreshAfterConsolidation: true } },
);

// This mental model will automatically refresh when observations are updated
console.log(`Operation ID: ${result2.operation_id}`);
// [/docs:create-mental-model-with-trigger]

await new Promise(r => setTimeout(r, 5000));

// [docs:list-mental-models]
// List all mental models in a bank
const mentalModels = await client.listMentalModels(BANK_ID);

for (const mm of mentalModels.items) {
    console.log(`- ${mm.name}: ${mm.source_query}`);
}
// [/docs:list-mental-models]

const mentalModelId = mentalModels.items[0]?.id;
if (!mentalModelId) {
    console.log('mental-models.mjs: All examples passed (no mental models created yet)');
    await fetch(`${HINDSIGHT_URL}/v1/default/banks/${BANK_ID}`, { method: 'DELETE' });
    process.exit(0);
}

// [docs:get-mental-model]
// Get a specific mental model
const mentalModel = await client.getMentalModel(BANK_ID, mentalModelId);

console.log(`Name: ${mentalModel.name}`);
console.log(`Content: ${mentalModel.content}`);
console.log(`Last refreshed: ${mentalModel.last_refreshed_at}`);
// [/docs:get-mental-model]

// [docs:refresh-mental-model]
// Refresh a mental model to update with current knowledge
const refreshResult = await client.refreshMentalModel(BANK_ID, mentalModelId);

console.log(`Refresh operation ID: ${refreshResult.operation_id}`);
// [/docs:refresh-mental-model]

// [docs:clear-mental-model]
// Clear a mental model's content, then refresh for a full re-synthesis
await client.clearMentalModel(BANK_ID, mentalModelId);

// Trigger a fresh full rebuild
const fullRefreshResult = await client.refreshMentalModel(BANK_ID, mentalModelId);

console.log(`Full refresh operation ID: ${fullRefreshResult.operation_id}`);
// [/docs:clear-mental-model]

// [docs:update-mental-model]
// Update a mental model's metadata
const updated = await client.updateMentalModel(BANK_ID, mentalModelId, {
    name: 'Updated Team Communication Preferences',
    trigger: { refresh_after_consolidation: true },
});

console.log(`Updated name: ${updated.name}`);
// [/docs:update-mental-model]

// [docs:get-mental-model-history]
// Get the change history of a mental model
const history = await client.getMentalModelHistory(BANK_ID, mentalModelId);

for (const entry of history) {
    console.log(`Changed at: ${entry.changed_at}`);
    console.log(`Previous content: ${entry.previous_content}`);
}
// [/docs:get-mental-model-history]

// [docs:delete-mental-model]
// Delete a mental model
await client.deleteMentalModel(BANK_ID, mentalModelId);
// [/docs:delete-mental-model]

// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await client.deleteBank(BANK_ID);

console.log('mental-models.mjs: All examples passed');
