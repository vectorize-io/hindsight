#!/usr/bin/env node
/**
 * Documents API examples for Hindsight (Node.js)
 * Run: node examples/api/documents.mjs
 */
import { HindsightClient, sdk, createClient, createConfig } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:document-retain]
// Retain with document ID
await client.retain('my-bank', 'Alice presented the Q4 roadmap...', {
    document_id: 'meeting-2024-03-15'
});

// Batch retain for a document with different sections
await client.retainBatch('my-bank', [
    { content: 'Item 1: Product launch delayed to Q2', document_id: 'meeting-2024-03-15-section-1' },
    { content: 'Item 2: New hiring targets announced', document_id: 'meeting-2024-03-15-section-2' },
    { content: 'Item 3: Budget approved for ML team', document_id: 'meeting-2024-03-15-section-3' }
]);
// [/docs:document-retain]


// [docs:document-update]
// Original
await client.retain('my-bank', 'Project deadline: March 31', {
    document_id: 'project-plan'
});

// Update
await client.retain('my-bank', 'Project deadline: April 15 (extended)', {
    document_id: 'project-plan'
});
// [/docs:document-update]


// [docs:document-list]
const apiClient = createClient(createConfig({ baseUrl: 'http://localhost:8888' }));

// List all documents
const { data: allDocs } = await sdk.listDocuments({
    client: apiClient,
    path: { bank_id: 'my-bank' }
});
console.log(`Total documents: ${allDocs.total}`);

// Filter by document ID substring
const { data: reportDocs } = await sdk.listDocuments({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    query: { q: 'report' }
});

// Filter by tags — only docs tagged with "team-a" (untagged excluded)
const { data: taggedDocs } = await sdk.listDocuments({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    query: { tags: ['team-a'], tags_match: 'any_strict' }
});

// Combine ID search and tags
const { data: filtered } = await sdk.listDocuments({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    query: { q: 'meeting', tags: ['team-a', 'team-b'], tags_match: 'all_strict' }
});

// Paginate
const { data: page } = await sdk.listDocuments({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    query: { limit: 20, offset: 40 }
});
console.log(`Page items: ${page.items.length}`);
// [/docs:document-list]


// [docs:document-get]
// Get document to expand context from recall results
const { data: doc, error } = await sdk.getDocument({
    client: apiClient,
    path: { bank_id: 'my-bank', document_id: 'meeting-2024-03-15-section-1' }
});

if (error) {
    throw new Error(`Failed to get document: ${JSON.stringify(error)}`);
}

console.log(`Document: ${doc.id}`);
console.log(`Original text: ${doc.original_text}`);
console.log(`Memory count: ${doc.memory_unit_count}`);
console.log(`Created: ${doc.created_at}`);
// [/docs:document-get]


// [docs:document-update-tags]
// Fix tags on a document retained with the wrong scope
const { data: updateResult, error: updateError } = await sdk.updateDocumentTags({
    client: apiClient,
    path: { bank_id: 'my-bank', document_id: 'meeting-2024-03-15-section-1' },
    body: { tags: ['team-a', 'team-b'] }
});

if (updateError) {
    throw new Error(`Failed to update tags: ${JSON.stringify(updateError)}`);
}

console.log(`Updated: ${updateResult.success}`);

// Remove all tags (make document visible everywhere)
await sdk.updateDocumentTags({
    client: apiClient,
    path: { bank_id: 'my-bank', document_id: 'meeting-2024-03-15-section-1' },
    body: { tags: [] }
});
// [/docs:document-update-tags]


// [docs:document-delete]
// Delete document and all its memories
const { data: deleteResult } = await sdk.deleteDocument({
    client: apiClient,
    path: { bank_id: 'my-bank', document_id: 'meeting-2024-03-15-section-1' }
});

console.log(`Deleted ${deleteResult.memory_units_deleted} memories`);
// [/docs:document-delete]


// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await fetch(`${HINDSIGHT_URL}/v1/default/banks/my-bank`, { method: 'DELETE' });

console.log('documents.mjs: All examples passed');
