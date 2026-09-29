#!/usr/bin/env node
/**
 * Knowledge Pages API examples for Hindsight (Node.js)
 * Run: node examples/api/knowledge-pages.mjs
 */
import { HindsightClient } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';
const BANK_ID = 'knowledge-pages-demo-bank-node';

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });
await client.createBank(BANK_ID, { name: 'Knowledge Pages Demo' });
await client.retain(BANK_ID, 'The API is deployed to Kubernetes with a rolling update');
await client.retain(BANK_ID, 'Deploys run from the main branch after CI passes');
await client.retain(BANK_ID, 'A failed deploy is rolled back by redeploying the previous tag');
await new Promise(r => setTimeout(r, 2000));

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:create-folder]
// Create a folder (omit parentId, or pass null, to create it at the root)
const folder = await client.createKnowledgeFolder(BANK_ID, 'Operations');

console.log(`Folder ID: ${folder.id}`);
// [/docs:create-folder]

// [docs:create-page]
// Create a page — content is generated in the background
const page = await client.createKnowledgePage(
    BANK_ID,
    'Deploying the API',
    'How is the API deployed?',
    { parentId: folder.id, tags: ['ops', 'type:runbook'] },
);

// Poll the operation to know when the first build has finished
console.log(`Page ID: ${page.page_id}, operation: ${page.operation_id}`);
// [/docs:create-page]

// Wait for the page's first build
await new Promise(r => setTimeout(r, 20000));

// [docs:get-tree]
// Fetch the whole knowledge base as a nested folder/page tree (no page bodies)
const tree = await client.getKnowledgeBaseTree(BANK_ID);

for (const root of tree.roots) {
    console.log(`${root.kind}: ${root.name}`);
    for (const child of root.children ?? []) {
        console.log(`  ${child.kind}: ${child.name} (stale: ${child.is_stale})`);
    }
}
// [/docs:get-tree]

// [docs:get-page]
// Read a page as a markdown document
const document = await client.getKnowledgePage(BANK_ID, page.page_id);

console.log(document.type);      // "runbook" — from the type:runbook tag
console.log(document.body);      // the synthesized markdown body
console.log(document.markdown);  // YAML frontmatter + body
// [/docs:get-page]

// [docs:search-pages]
// Hybrid search (full-text + vector) over whole pages
const results = await client.searchKnowledgeBase(BANK_ID, 'how do we deploy', { limit: 5 });

for (const hit of results.results) {
    console.log(`${hit.score.toFixed(3)}  ${hit.name}: ${hit.snippet}`);
}
// [/docs:search-pages]

// [docs:update-node]
// Rename a node, move it, and/or update a page's options.
// Changing sourceQuery rebuilds the page against the new question.
await client.updateKnowledgeNode(BANK_ID, page.page_id, {
    name: 'Deploying the API (v2)',
    tags: ['ops', 'type:runbook', 'reviewed'],
});
// [/docs:update-node]

// [docs:export]
// Export the knowledge base as a portable markdown bundle
const bundle = await client.exportKnowledgeBase(BANK_ID);

for (const file of bundle.files) {
    console.log(file.path);  // index.md, <page-id>.md, <page-id>.log.md
}
// [/docs:export]

// [docs:delete-node]
// Delete a folder or page — deleting a folder removes its whole subtree
await client.deleteKnowledgeNode(BANK_ID, folder.id);
// [/docs:delete-node]

// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await client.deleteBank(BANK_ID);

console.log('knowledge-pages.mjs: All examples passed');
