#!/usr/bin/env node
/**
 * Opinions API examples for Hindsight (deprecated - kept for versioned docs).
 * This file is preserved for backward compatibility with v0.3 documentation.
 * Opinions have been replaced by Mental Models in v0.4+.
 */
import { HindsightClient } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });

// [docs:opinion-form]
// Opinions are automatically formed when the bank encounters
// claims, preferences, or judgments in retained content
await client.retain('my-bank',
    "I think Python is excellent for data science because of its libraries"
);

// The bank forms an opinion with confidence based on evidence
// [/docs:opinion-form]


// [docs:opinion-search]
// Search for opinions on a topic
const response = await client.recall('my-bank', 'What do you think about Python?', {
    types: ['opinion']
});

for (const opinion of response.results) {
    console.log(`Opinion: ${opinion.text}`);
    console.log(`Confidence: ${opinion.confidence}`);
}
// [/docs:opinion-search]


// [docs:opinion-disposition]
// Bank disposition affects how opinions are formed
// High skepticism = lower confidence, requires more evidence
// Low skepticism = higher confidence, accepts claims more readily

await client.createBank('skeptical-bank', {
    disposition: { skepticism: 5, literalism: 3, empathy: 2 }
});

// Same content, different confidence due to disposition
await client.retain('skeptical-bank', 'Python is the best language');
// [/docs:opinion-disposition]


// [docs:opinion-in-reflect]
// Opinions influence reflect responses
const reflectResponse = await client.reflect('my-bank',
    'Should I use Python for my data project?'
);

// The response incorporates the bank's opinions with appropriate confidence
console.log(reflectResponse.text);
// [/docs:opinion-in-reflect]


// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await fetch(`${HINDSIGHT_URL}/v1/default/banks/my-bank`, { method: 'DELETE' });
await fetch(`${HINDSIGHT_URL}/v1/default/banks/skeptical-bank`, { method: 'DELETE' });

console.log('opinions.mjs: All examples passed');
