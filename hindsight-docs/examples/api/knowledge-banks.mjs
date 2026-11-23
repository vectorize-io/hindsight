#!/usr/bin/env node
/**
 * Knowledge Banks examples for Hindsight (Node.js)
 * Run: node examples/api/knowledge-banks.mjs
 */
import { readFileSync } from 'node:fs';
import { HindsightClient, sdk, createClient, createConfig } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';
const SAMPLE_PDF = new URL('./sample.pdf', import.meta.url);

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });
await client.createBank('support-agent', { name: 'Support Agent' });
const apiClient = createClient(createConfig({ baseUrl: HINDSIGHT_URL }));

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:create-knowledge-bank]
await sdk.createKnowledgeBank({
    client: apiClient,
    body: { bank_id: 'product-manuals' },
    throwOnError: true,
});
// [/docs:create-knowledge-bank]

// [docs:ingest-files]
const { data: ingested } = await sdk.ingestKnowledgeFiles({
    client: apiClient,
    path: { bank_id: 'product-manuals' },
    body: {
        files: [
            new File([readFileSync(SAMPLE_PDF)], 'x200-manual.pdf'),
            new File(['# X200 quick start\n\nHold the power button for two seconds.\n'], 'x200-quickstart.md'),
        ],
        request: JSON.stringify({ files: [
            { document_id: 'x200-manual', tags: ['x200'] },
            { document_id: 'x200-quickstart', tags: ['x200', 'quickstart'] },
        ] }),
    },
    throwOnError: true,
});
console.log(ingested.document_ids, ingested.operation_ids);
// [/docs:ingest-files]

// [docs:ingest-text]
await sdk.ingestKnowledgeTexts({
    client: apiClient,
    path: { bank_id: 'product-manuals' },
    body: { items: [{
        document_id: 'returns-policy',
        text: '# Returns\n\nYou can return an X200 within 30 days...',
        tags: ['policy'],
    }] },
    throwOnError: true,
});
// [/docs:ingest-text]

// [docs:search]
const { data: found } = await sdk.searchKnowledgeBank({
    client: apiClient,
    path: { bank_id: 'product-manuals' },
    body: { query: 'How long do I have to return an X200?', max_tokens: 2048 },
    throwOnError: true,
});
for (const hit of found.results) console.log(hit.document_id, hit.text);
// [/docs:search]

// [docs:list-tags]
const { data: tags } = await sdk.listKnowledgeBankTags({
    client: apiClient,
    path: { bank_id: 'product-manuals' },
    query: { q: 'x2*' },
    throwOnError: true,
});
for (const item of tags.items) console.log(item.tag, item.count);
// [/docs:list-tags]

// [docs:link-memory-bank]
await client.updateBankConfig('support-agent', { knowledgeBanks: ['product-manuals'] });
// [/docs:link-memory-bank]

// [docs:bank-config]
await sdk.updateKnowledgeBankConfig({
    client: apiClient,
    path: { bank_id: 'product-manuals' },
    body: { updates: { text_passage_tokens: 256, disabled_media_types: ['video'] } },
    throwOnError: true,
});
// [/docs:bank-config]

// =============================================================================
// Checks (not shown in docs): the calls above must have worked, not just answered
// =============================================================================
function check(ok, what) {
    if (!ok) throw new Error(`check failed: ${what}`);
}
check(found.results.some((h) => h.document_id === 'returns-policy'), 'search finds the text');
const { data: allTags } = await sdk.listKnowledgeBankTags({ client: apiClient, path: { bank_id: 'product-manuals' } });
check(allTags.items.some((t) => t.tag === 'policy'), 'policy tag listed');
const { data: kbConfig } = await sdk.getKnowledgeBankConfig({ client: apiClient, path: { bank_id: 'product-manuals' } });
check(kbConfig.overrides.text_passage_tokens === 256, 'override stored');
check((await client.getBankConfig('support-agent')).config.knowledge_banks.includes('product-manuals'), 'link stored');
// The file ingest runs in the background: wait for the quick start to be ready.
let doc;
for (let i = 0; i < 120; i++) {
    ({ data: doc } = await sdk.getKnowledgeDocument({
        client: apiClient,
        path: { bank_id: 'product-manuals', document_id: 'x200-quickstart' },
    }));
    if (doc.status === 'ready' || doc.status === 'failed') break;
    await new Promise((r) => setTimeout(r, 1000));
}
check(doc.status === 'ready', `quick start ready (${doc.error})`);

// [docs:delete-knowledge-bank]
await sdk.deleteKnowledgeBank({
    client: apiClient,
    path: { bank_id: 'product-manuals' },
    throwOnError: true,
});
// [/docs:delete-knowledge-bank]

// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await client.deleteBank('support-agent');

console.log('knowledge-banks.mjs: All examples passed');
