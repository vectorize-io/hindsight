#!/usr/bin/env node
/**
 * Retain API examples for Hindsight (Node.js)
 * Run: node examples/api/retain.mjs
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

// [docs:retain-basic]
await client.retain('my-bank', 'Alice works at Google as a software engineer');
// [/docs:retain-basic]


// [docs:retain-with-context]
await client.retain('my-bank', 'Alice got promoted to senior engineer', {
    context: 'career update',
    timestamp: '2024-03-15T10:00:00Z'
});
// [/docs:retain-with-context]


// [docs:retain-batch]
await client.retainBatch('my-bank', [
    { content: 'Alice works at Google', context: 'career', document_id: 'conversation_001_msg_1' },
    { content: 'Bob is a data scientist at Meta', context: 'career', document_id: 'conversation_001_msg_2' },
    { content: 'Alice and Bob are friends', context: 'relationship', document_id: 'conversation_001_msg_3' }
]);
// [/docs:retain-batch]


// [docs:retain-async]
// Start async ingestion (returns immediately)
await client.retainBatch('my-bank', [
    { content: 'Large batch item 1', document_id: 'large-doc-1' },
    { content: 'Large batch item 2', document_id: 'large-doc-2' },
], {
    async: true
});
// [/docs:retain-async]


// [docs:retain-files]
// Upload files and retain their contents as memories.
// Supports: PDF, DOCX, PPTX, XLSX, images (OCR), audio (transcription), and text formats.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const pdfBytes = readFileSync(join(__dirname, 'sample.pdf'));
const result = await client.retainFiles('my-bank', [
    new File([pdfBytes], 'sample.pdf'),
], { context: 'quarterly report' });
console.log(result.operation_ids);  // Track processing via the operations endpoint
// [/docs:retain-files]


// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await fetch(`${HINDSIGHT_URL}/v1/default/banks/my-bank`, { method: 'DELETE' });

console.log('retain.mjs: All examples passed');
