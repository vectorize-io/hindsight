#!/usr/bin/env node
/**
 * Quickstart examples for Hindsight API (Node.js)
 * Run: node examples/api/quickstart.mjs
 */

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:quickstart-full]
import { HindsightClient } from '@vectorize-io/hindsight-client';

const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

// Retain: Store information
await client.retain('my-bank', 'Alice works at Google as a software engineer');

// Recall: Search memories
await client.recall('my-bank', 'What does Alice do?');

// Reflect: Generate response
await client.reflect('my-bank', 'Tell me about Alice');
// [/docs:quickstart-full]


// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await fetch(`${HINDSIGHT_URL}/v1/default/banks/my-bank`, { method: 'DELETE' });

console.log('quickstart.mjs: All examples passed');
