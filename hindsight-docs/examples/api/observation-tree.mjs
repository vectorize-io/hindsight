#!/usr/bin/env node
/**
 * Observation tree API examples for Hindsight (Node.js) — the bank's observations as markdown files.
 * Run: node examples/api/observation-tree.mjs
 */
import { mkdirSync, mkdtempSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { HindsightClient, sdk, createClient, createConfig } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';
const BANK_ID = 'observation-tree-demo-bank-node';

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });
await client.createBank(BANK_ID, { name: 'Observation Tree Demo' });
await client.retain(BANK_ID, 'The API runs on Kubernetes behind an nginx ingress.', { tags: ['team:platform'] });
await client.retain(BANK_ID, 'Redis keys expire after an hour.', { tags: ['team:platform'] });
await client.retain(BANK_ID, 'The GPU nodes serve Gemma with vLLM.', { tags: ['team:ml'] });
await new Promise((r) => setTimeout(r, 5000));
const outDir = mkdtempSync(join(tmpdir(), 'observation-tree-'));

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:get-tree]
// The whole bank: folders, topic files and each folder's index.md.
const apiClient = createClient(createConfig({ baseUrl: HINDSIGHT_URL }));
const { data: tree } = await sdk.getObservationTree({ client: apiClient, path: { bank_id: BANK_ID } });
console.log(`${tree.observation_count} observations`);

const walk = (folder) => {
    for (const f of folder.files) console.log(`/${f.path} (${f.observation_count})`);
    folder.folders.forEach(walk);
};
walk(tree);

// Only one scope: the same tag filters as recall.
const { data: platform } = await sdk.getObservationTree({
    client: apiClient,
    path: { bank_id: BANK_ID },
    query: { tags: ['team:platform'], tags_match: 'any_strict' },
});
walk(platform);
// [/docs:get-tree]

// [docs:write-bundle]
// Write the bundle to disk: each file's markdown at its path.
const write = (folder) => {
    for (const f of folder.files) {
        const path = join(outDir, f.path);
        mkdirSync(dirname(path), { recursive: true });
        writeFileSync(path, f.content, 'utf8');
    }
    folder.folders.forEach(write);
};
write(tree);
// [/docs:write-bundle]
if (!existsSync(join(outDir, 'index.md'))) throw new Error('index.md missing');

// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await client.deleteBank(BANK_ID);

console.log('observation-tree.mjs: All examples passed');
