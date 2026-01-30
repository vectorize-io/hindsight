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
await client.createBank('my-bank', {
    name: 'Research Assistant',
    mission: 'I am a research assistant specializing in machine learning',
    disposition: {
        skepticism: 4,
        literalism: 3,
        empathy: 3
    }
});
// [/docs:create-bank]


// [docs:bank-mission]
await client.createBank('financial-advisor', {
    name: 'Financial Advisor',
    mission: `I am a conservative financial advisor with 20 years of experience.
    I prioritize capital preservation over aggressive growth.
    I have seen multiple market crashes and believe in diversification.`
});
// [/docs:bank-mission]


// [docs:bank-background]
// Legacy snippet for v0.3 docs (background renamed to mission in v0.4)
await client.createBank('legacy-bank', {
    name: 'Legacy Example',
    mission: `I'm a personal assistant helping a software engineer. I should track their
    project preferences, coding style, and technology choices.`
});
// [/docs:bank-background]


// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await fetch(`${HINDSIGHT_URL}/v1/default/banks/my-bank`, { method: 'DELETE' });
await fetch(`${HINDSIGHT_URL}/v1/default/banks/financial-advisor`, { method: 'DELETE' });

console.log('memory-banks.mjs: All examples passed');
