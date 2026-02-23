#!/usr/bin/env node
/**
 * Memory Banks API examples for Hindsight (Node.js)
 * Run: node examples/api/memory-banks.mjs
 */
import { HindsightClient } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:create-bank]
await client.createBank('my-bank');
// [/docs:create-bank]


// [docs:bank-with-disposition]
await client.createBank('architect-bank');
await client.updateBankConfig('architect-bank', {
    reflectMission: "You're a senior software architect - keep track of system designs, technology decisions, and architectural patterns.",
    dispositionSkepticism: 4,   // Questions new technologies
    dispositionLiteralism: 4,   // Focuses on concrete specs
    dispositionEmpathy: 2,      // Prioritizes technical facts
});
// [/docs:bank-with-disposition]


// [docs:update-bank-config]
await client.updateBankConfig('my-bank', {
    retainMission: 'Always include technical decisions, API design choices, and architectural trade-offs. Ignore meeting logistics and social exchanges.',
    retainExtractionMode: 'verbose',
    observationsMission: 'Observations are stable facts about people and projects. Always include preferences, skills, and recurring patterns. Ignore one-off events.',
    dispositionSkepticism: 4,
    dispositionLiteralism: 4,
    dispositionEmpathy: 2,
});
// [/docs:update-bank-config]


// [docs:get-bank-config]
// Returns resolved config (server defaults merged with bank overrides) and the raw overrides
const { config, overrides } = await client.getBankConfig('my-bank');
// config    — full resolved configuration
// overrides — only fields overridden at the bank level
// [/docs:get-bank-config]


// [docs:reset-bank-config]
// Remove all bank-level overrides, reverting to server defaults
await client.resetBankConfig('my-bank');
// [/docs:reset-bank-config]


// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await fetch(`${HINDSIGHT_URL}/v1/default/banks/my-bank`, { method: 'DELETE' });
await fetch(`${HINDSIGHT_URL}/v1/default/banks/architect-bank`, { method: 'DELETE' });

console.log('memory-banks.mjs: All examples passed');
