#!/usr/bin/env node
/**
 * Memory Banks API examples for Hindsight (Node.js)
 * Run: node examples/api/memory-banks.mjs
 */
import { HindsightClient, sdk, createClient, createConfig } from '@vectorize-io/hindsight-client';
import { readFile, writeFile, rm } from 'node:fs/promises';

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


// [docs:bank-background]
await client.createBank('my-bank');
await client.updateBankConfig('my-bank', {
    reflectMission: 'I am a research assistant specializing in machine learning.',
});
// [/docs:bank-background]


// [docs:bank-mission]
await client.createBank('my-bank');
await client.updateBankConfig('my-bank', {
    reflectMission: "You're a senior software architect - keep track of system designs, technology decisions, and architectural patterns.",
});
// [/docs:bank-mission]


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
// Prompt preview and bank transfer
// =============================================================================
const apiClient = createClient(createConfig({ baseUrl: HINDSIGHT_URL }));
const TRANSFER_BANKS = ['transfer-js', 'transfer-js-copy', 'transfer-js-other', 'transfer-js-clone'];
for (const b of TRANSFER_BANKS) await fetch(`${HINDSIGHT_URL}/v1/default/banks/${b}`, { method: 'DELETE' });
// Chunks mode stores text verbatim, so no LLM call is needed.
await client.createBank('transfer-js', {});
await client.updateBankConfig('transfer-js', { retainExtractionMode: 'chunks' });
const retainRes = await fetch(`${HINDSIGHT_URL}/v1/default/banks/transfer-js/memories`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items: [
        { content: 'Alice leads the payments team.', document_id: 'doc-1' },
        { content: 'Bob moved to the Berlin office.', document_id: 'doc-2' },
    ] }),
});
if (!retainRes.ok) throw new Error(`retain failed: ${retainRes.status}`);
await client.createBank('transfer-js-other', {});

async function waitFor(bankId, operationId) {
    for (let i = 0; i < 120; i++) {
        const { data } = await sdk.getOperationStatus({
            client: apiClient, path: { bank_id: bankId, operation_id: operationId },
        });
        if (data.status === 'completed') return data;
        if (data.status === 'failed' || data.status === 'cancelled') {
            throw new Error(`operation ${operationId} ${data.status}: ${data.error_message}`);
        }
        await new Promise((r) => setTimeout(r, 1000));
    }
    throw new Error(`operation ${operationId} did not finish`);
}

// [docs:prompts-preview]
const { data: preview } = await sdk.previewPrompt({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    body: { operation: 'retain' },
});
for (const message of preview.messages) console.log(message.role, message.blocks.length);
// [/docs:prompts-preview]
if (!preview.messages.length) throw new Error('empty preview');


// [docs:transfer-export]
// Whole bank, memories + config, no history.
// Submits the export, polls the operation, downloads the ZIP.
const archive = await client.exportBank('transfer-js');

// Just the memories
const memoriesOnly = await client.exportBank('transfer-js', { includeBankConfig: false });

// Specific documents (a document subset carries no bank-level sections).
// The low-level call only submits; poll the returned operation yourself.
const { data: subset } = await sdk.exportBankTransfer({
    client: apiClient,
    path: { bank_id: 'transfer-js' },
    query: { document_id: ['doc-1', 'doc-2'], include_bank_config: false },
});
// [/docs:transfer-export]
if (archive[0] !== 0x50 || memoriesOnly[0] !== 0x50) throw new Error('not a zip');
await waitFor('transfer-js', subset.operation_id);


// [docs:transfer-import]
// Restore a bank under a new id
const restoreId = await client.importBank('transfer-js', new Blob([archive]), {
    targetBankId: 'transfer-js-copy',
});
// The restore is recorded against the bank in the URL — poll it there
const { data: restoreStatus } = await sdk.getOperationStatus({
    client: apiClient,
    path: { bank_id: 'transfer-js', operation_id: restoreId },
});

// Merge an archive's documents into an existing bank
const { data: merge } = await sdk.importBankTransfer({
    client: apiClient,
    path: { bank_id: 'transfer-js-other' },
    query: { mode: 'merge', document_conflict: 'replace' },
    body: { file: new Blob([archive]) },
});
// [/docs:transfer-import]
await waitFor('transfer-js', restoreId);
await waitFor('transfer-js-other', merge.operation_id);
const copied = await client.listDocuments('transfer-js-copy');
if (copied.total !== 2) throw new Error(`restore copied ${copied.total} documents`);


// [docs:clone-bank]
const cloneId = await client.cloneBank('transfer-js', 'transfer-js-clone');
// The operation is recorded against the source bank
const { data: cloneStatus } = await sdk.getOperationStatus({
    client: apiClient,
    path: { bank_id: 'transfer-js', operation_id: cloneId },
});
// [/docs:clone-bank]
await waitFor('transfer-js', cloneId);
const cloned = await client.listDocuments('transfer-js-clone');
if (cloned.total !== 2) throw new Error(`clone copied ${cloned.total} documents`);


// [docs:document-export]
// Submits the export (whole bank; pass { documentIds: [...] } to scope it),
// polls the operation until completed, downloads the archive.
const docArchive = await client.exportDocuments('transfer-js');
await writeFile('transfer-js-documents.zip', docArchive);
// [/docs:document-export]


// [docs:document-import]
const { data: docImport } = await sdk.importDocuments({
    client: apiClient,
    path: { bank_id: 'transfer-js-other' },
    query: { on_conflict: 'replace' },
    body: { file: new Blob([await readFile('transfer-js-documents.zip')]) },
});

const { data: docImportStatus } = await sdk.getOperationStatus({
    client: apiClient,
    path: { bank_id: 'transfer-js-other', operation_id: docImport.operation_id },
});
// docImportStatus.result_metadata -> { documents_imported: 3, facts_imported: 42, observations_imported: 5, ... }
// [/docs:document-import]
await waitFor('transfer-js-other', docImport.operation_id);
await rm('transfer-js-documents.zip');

// [docs:transfer-import-external]
import { crc32 } from 'node:zlib';

// Minimal uncompressed ZIP writer (Node has none built in); a library like jszip works too.
function zip(files) {
    const locals = [], centrals = [];
    let offset = 0;
    for (const [name, text] of Object.entries(files)) {
        const n = Buffer.from(name), d = Buffer.from(text), crc = crc32(d);
        const local = Buffer.alloc(30);
        local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(0x21, 12);
        local.writeUInt32LE(crc, 14); local.writeUInt32LE(d.length, 18); local.writeUInt32LE(d.length, 22);
        local.writeUInt16LE(n.length, 26);
        const central = Buffer.alloc(46);
        central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6);
        central.writeUInt16LE(0x21, 14); central.writeUInt32LE(crc, 16); central.writeUInt32LE(d.length, 20);
        central.writeUInt32LE(d.length, 24); central.writeUInt16LE(n.length, 28); central.writeUInt32LE(offset, 42);
        locals.push(local, n, d);
        centrals.push(central, n);
        offset += 30 + n.length + d.length;
    }
    const dir = Buffer.concat(centrals), end = Buffer.alloc(22);
    const count = Object.keys(files).length;
    end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(count, 8); end.writeUInt16LE(count, 10);
    end.writeUInt32LE(dir.length, 12); end.writeUInt32LE(offset, 16);
    return Buffer.concat([...locals, dir, end]);
}

const doc = {
    id: 'session-2026-09-22',
    original_text: 'Full original session text...',
    chunks: [{ chunk_index: 0, chunk_text: 'Caller-defined source region...' }],
    facts: [{
        text: 'The user prefers lightweight local speech recognition models.',
        fact_type: 'experience',
        chunk_index: 0,
        mentioned_at: '2026-09-22T18:34:00Z',
        entities: ['Parakeet'],
    }],
};
const archiveZip = zip({
    'manifest.json': JSON.stringify({ schema_version: 1, source_bank_id: 'external' }),
    [`documents/${doc.id}.json`]: JSON.stringify(doc),
});

const { data: external } = await sdk.importBankTransfer({
    client: apiClient,
    path: { bank_id: 'transfer-js-other' },
    query: { mode: 'merge', document_conflict: 'replace' },
    body: { file: new Blob([archiveZip]) },
});
// [/docs:transfer-import-external]
await waitFor('transfer-js-other', external.operation_id);
const { data: importedDoc } = await sdk.getDocument({
    client: apiClient,
    path: { bank_id: 'transfer-js-other', document_id: 'session-2026-09-22' },
});
if (importedDoc.memory_unit_count !== 1) throw new Error(`external import stored ${importedDoc.memory_unit_count} facts`);
for (const b of TRANSFER_BANKS) await fetch(`${HINDSIGHT_URL}/v1/default/banks/${b}`, { method: 'DELETE' });


// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await fetch(`${HINDSIGHT_URL}/v1/default/banks/my-bank`, { method: 'DELETE' });
await fetch(`${HINDSIGHT_URL}/v1/default/banks/architect-bank`, { method: 'DELETE' });

console.log('memory-banks.mjs: All examples passed');
