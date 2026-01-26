#!/usr/bin/env node
/**
 * Reflect API examples for Hindsight (Node.js)
 * Run: node examples/api/reflect.mjs
 */
import { HindsightClient } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });

// Seed some data for reflect examples
await client.retain('my-bank', 'Alice works at Google as a software engineer');
await client.retain('my-bank', 'Alice has been working there for 5 years');
await client.retain('my-bank', 'Alice recently got promoted to senior engineer');

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:reflect-basic]
await client.reflect('my-bank', 'What should I know about Alice?');
// [/docs:reflect-basic]


// [docs:reflect-with-params]
const response = await client.reflect('my-bank', 'What do you think about remote work?', {
    budget: 'mid',
    context: "We're considering a hybrid work policy"
});
// [/docs:reflect-with-params]


// [docs:reflect-with-context]
// Context helps the LLM understand the current situation
const contextResponse = await client.reflect('my-bank', 'What do you think about the proposal?', {
    context: "We're in a budget review meeting discussing Q4 spending"
});
// [/docs:reflect-with-context]


// [docs:reflect-disposition]
// Create a bank with specific disposition
await client.createBank('cautious-advisor', {
    name: 'Cautious Advisor',
    background: 'I am a risk-aware financial advisor',
    disposition: {
        skepticism: 5,
        literalism: 4,
        empathy: 2
    }
});

// Reflect responses will reflect this disposition
const advisorResponse = await client.reflect('cautious-advisor', 'Should I invest in crypto?');
// [/docs:reflect-disposition]


// [docs:reflect-sources]
const sourcesResponse = await client.reflect('my-bank', 'Tell me about Alice');

console.log('Response:', sourcesResponse.text);
console.log('\nBased on:');
for (const fact of sourcesResponse.based_on || []) {
    console.log(`  - [${fact.type}] ${fact.text}`);
}
// [/docs:reflect-sources]


// [docs:reflect-structured-output]
// Define JSON schema directly
const responseSchema = {
    type: 'object',
    properties: {
        recommendation: { type: 'string' },
        confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
        key_factors: { type: 'array', items: { type: 'string' } },
        risks: { type: 'array', items: { type: 'string' } },
    },
    required: ['recommendation', 'confidence', 'key_factors'],
};

const structuredResponse = await client.reflect('my-bank', 'What do you know about Alice and her career?', {
    responseSchema: responseSchema,
});

// Structured output (if returned)
if (structuredResponse.structuredOutput) {
    console.log('Recommendation:', structuredResponse.structuredOutput.recommendation || 'N/A');
    console.log('Key factors:', structuredResponse.structuredOutput.key_factors || []);
}
// [/docs:reflect-structured-output]


// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await fetch(`${HINDSIGHT_URL}/v1/default/banks/my-bank`, { method: 'DELETE' });
await fetch(`${HINDSIGHT_URL}/v1/default/banks/cautious-advisor`, { method: 'DELETE' });

console.log('reflect.mjs: All examples passed');
