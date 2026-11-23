#!/usr/bin/env node
/**
 * Knowledge Banks examples for Hindsight (Node.js)
 * Run: node examples/api/knowledge-banks.mjs
 */
import { readFileSync } from 'node:fs';
import { HindsightClient } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';
const SAMPLE_PDF = new URL('./sample.pdf', import.meta.url);

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });
await client.createBank('support-agent', { name: 'Support Agent' });

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:create-knowledge-bank]
await client.createKnowledgeBank('product-manuals');
// [/docs:create-knowledge-bank]

// [docs:ingest-files]
const ingested = await client.ingestKnowledgeFiles(
    'product-manuals',
    [
        new File([readFileSync(SAMPLE_PDF)], 'x200-manual.pdf'),
        new File(['# X200 quick start\n\nHold the power button for two seconds.\n'], 'x200-quickstart.md'),
    ],
    {
        filesOptions: [
            { document_id: 'x200-manual', tags: ['x200'] },
            { document_id: 'x200-quickstart', tags: ['x200', 'quickstart'] },
        ],
    },
);
console.log(ingested.document_ids, ingested.operation_ids);
// [/docs:ingest-files]

// [docs:ingest-text]
await client.ingestKnowledgeTexts('product-manuals', [{
    document_id: 'returns-policy',
    text: '# Returns\n\nYou can return an X200 within 30 days...',
    tags: ['policy'],
}]);
// [/docs:ingest-text]

// [docs:search]
const found = await client.searchKnowledgeBank('product-manuals', {
    query: 'How long do I have to return an X200?',
    maxTokens: 2048,
});
for (const hit of found.results) console.log(hit.document_id, hit.text);
// [/docs:search]

// [docs:list-tags]
const tags = await client.listKnowledgeBankTags('product-manuals', { q: 'x2*' });
for (const item of tags.items) console.log(item.tag, item.count);
// [/docs:list-tags]

// [docs:link-memory-bank]
await client.updateBankConfig('support-agent', { knowledgeBanks: ['product-manuals'] });
// [/docs:link-memory-bank]

// [docs:bank-config]
await client.updateKnowledgeBankConfig('product-manuals', {
    text_passage_tokens: 256,
    disabled_media_types: ['video'],
});
// [/docs:bank-config]

// =============================================================================
// Checks (not shown in docs): the calls above must have worked, not just answered
// =============================================================================
function check(ok, what) {
    if (!ok) throw new Error(`check failed: ${what}`);
}
check(found.results.some((h) => h.document_id === 'returns-policy'), 'search finds the text');
const allTags = await client.listKnowledgeBankTags('product-manuals');
check(allTags.items.some((t) => t.tag === 'policy'), 'policy tag listed');
const kbConfig = await client.getKnowledgeBankConfig('product-manuals');
check(kbConfig.overrides.text_passage_tokens === 256, 'override stored');
check((await client.getBankConfig('support-agent')).config.knowledge_banks.includes('product-manuals'), 'link stored');
// The file ingest runs in the background: wait for the quick start to be ready.
let doc;
for (let i = 0; i < 120; i++) {
    doc = await client.getKnowledgeDocument('product-manuals', 'x200-quickstart');
    if (doc.status === 'ready' || doc.status === 'failed') break;
    await new Promise((r) => setTimeout(r, 1000));
}
check(doc.status === 'ready', `quick start ready (${doc.error})`);

// [docs:documents]
const { documents } = await client.listKnowledgeDocuments('product-manuals', { status: 'ready' });
for (const d of documents) console.log(d.document_id, d.media_type, d.passage_count);
const detail = await client.getKnowledgeDocument('product-manuals', 'x200-quickstart');
console.log(detail.status, detail.passages.map((p) => p.text));
// [/docs:documents]
check(
    (await client.listKnowledgeDocuments('product-manuals')).documents.some((d) => d.document_id === 'returns-policy'),
    'returns policy listed',
);

// [docs:settings]
// Every setting as the bank uses it, then only what the bank changes
const settings = await client.getKnowledgeBankConfig('product-manuals');
console.log(settings.config, settings.overrides);

// Drop one override, or all of them
await client.updateKnowledgeBankConfig('product-manuals', { disabled_media_types: null });
await client.resetKnowledgeBankConfig('product-manuals');
// [/docs:settings]
check(
    (await client.getKnowledgeBankConfig('product-manuals')).overrides.text_passage_tokens == null,
    'overrides reset',
);

// [docs:delete-document]
await client.deleteKnowledgeDocument('product-manuals', 'returns-policy');
// [/docs:delete-document]
check(
    (await client.listKnowledgeDocuments('product-manuals')).documents.every((d) => d.document_id !== 'returns-policy'),
    'returns policy deleted',
);

// [docs:delete-knowledge-bank]
await client.deleteKnowledgeBank('product-manuals');
// [/docs:delete-knowledge-bank]

// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await client.deleteBank('support-agent');

console.log('knowledge-banks.mjs: All examples passed');
