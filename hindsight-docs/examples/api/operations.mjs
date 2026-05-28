#!/usr/bin/env node
/**
 * Operations API examples for Hindsight (Node.js)
 * Run: node examples/api/operations.mjs
 */
import { HindsightClient, sdk, createClient, createConfig } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });
const apiClient = createClient(createConfig({ baseUrl: HINDSIGHT_URL }));

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:operations-list]
// List recent operations for a bank (default: 20 most recent).
const { data: recent } = await sdk.listOperations({
    client: apiClient,
    path: { bank_id: 'my-bank' },
});
for (const op of recent.operations) {
    console.log(op.id, op.task_type, op.status);
}

// Filter by status and type.
const { data: pendingRecompute } = await sdk.listOperations({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    query: { status: 'pending', type: 'graph_maintenance' },
});

// Hide retain_batch parent rows (show only individual child retain jobs).
const { data: flat } = await sdk.listOperations({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    query: { exclude_parents: true },
});
// [/docs:operations-list]


// [docs:operations-get]
const { data: status } = await sdk.getOperationStatus({
    client: apiClient,
    path: { bank_id: 'my-bank', operation_id: '550e8400-e29b-41d4-a716-446655440000' },
});
console.log(status.status, status.error_message);

// Include the submission payload (can be large for retain batches).
const { data: detailed } = await sdk.getOperationStatus({
    client: apiClient,
    path: { bank_id: 'my-bank', operation_id: '550e8400-e29b-41d4-a716-446655440000' },
    query: { include_payload: true },
});
// [/docs:operations-get]


// [docs:operations-cancel]
// Cancel a pending operation before a worker claims it.
// Returns 409 if the operation is already processing/completed/failed.
await sdk.cancelOperation({
    client: apiClient,
    path: { bank_id: 'my-bank', operation_id: '550e8400-e29b-41d4-a716-446655440000' },
});
// [/docs:operations-cancel]


// [docs:operations-retry]
// Re-queue a failed (or cancelled) operation.
// Returns 409 if the operation isn't in failed/cancelled state.
await sdk.retryOperation({
    client: apiClient,
    path: { bank_id: 'my-bank', operation_id: '550e8400-e29b-41d4-a716-446655440000' },
});
// [/docs:operations-retry]


// [docs:operations-async-retain]
// Submit a large batch asynchronously — the call returns immediately with an
// operation_id you can poll.
const submission = await client.retainBatch('my-bank', [
    { content: 'Alice joined Google in 2023' },
    { content: 'Bob prefers Python over JavaScript' },
], { async: true });
const operationId = submission.operation_id;

while (true) {
    const { data: s } = await sdk.getOperationStatus({
        client: apiClient,
        path: { bank_id: 'my-bank', operation_id: operationId },
    });
    if (['completed', 'failed', 'cancelled'].includes(s.status)) {
        console.log(`finished: ${s.status}`);
        break;
    }
    await new Promise((r) => setTimeout(r, 2000));
}
// [/docs:operations-async-retain]
