
# Operations

Hindsight runs several maintenance and ingestion tasks asynchronously instead of blocking the API call that triggers them. These tasks share a single queue (`async_operations`) and a single worker pool, and the same REST endpoints — list, status, cancel, retry — work across every type.

This page explains each operation type, when it fires, and how to inspect or manage it.

{/* Import raw source files */}

> **💡 Prerequisites**
>
Make sure you've completed the [Quick Start](./quickstart) and understand [how retain works](./retain).
## How operations work

When an API call needs background work, the request handler writes a row to the `async_operations` table with `status=pending` and returns immediately. A worker (running either in-process inside the API by default, or as a dedicated service — see [Services - Worker Service](../services#worker-service)) polls the table, claims pending rows, executes the corresponding handler, and marks the row `completed` or `failed`.

By default, every operation runs in-process: no external queue, no extra process to deploy. The same code paths support scaling out to dedicated worker processes when throughput demands it.

### Lifecycle

| Status | Meaning |
|--------|---------|
| `pending` | The row is queued. Either no worker has picked it up yet, or an extension has parked it via `next_retry_at` in the future (e.g., for backpressure). |
| `processing` | A worker has claimed the row and is actively running the handler. |
| `completed` | The handler returned successfully. |
| `failed` | The handler raised. `error_message` carries the reason; you can re-queue with `POST /…/retry`. |
| `cancelled` | The operation was cancelled via `DELETE /…/operations/{id}` before a worker picked it up. Cancelling a `processing` operation is not supported. |

The worker retries failed operations up to `HINDSIGHT_API_WORKER_MAX_RETRIES` times before settling on `failed`. Deterministic failures (e.g., invalid embedding dimensions, integrity violations) skip retries — they won't succeed by re-running.

## Operation types

Every operation has an `operation_type` in the database and a `task_type` in the payload. They're usually the same.

### `retain`

Submitted by `POST /v1/default/banks/{bank_id}/memories` with `async=true`, or by the multi-item `retain_batch` call. The handler runs the same pipeline as a synchronous retain: fact extraction (LLM), embedding generation, entity resolution, and link creation (temporal, semantic).

Use async retain when you're ingesting thousands of items and don't want the HTTP call to hold for minutes. The `operation_id` in the response lets you poll for completion.

#### Parent op: `retain_batch`

For large submissions, Hindsight automatically splits the input into sub-batches and creates a single `retain_batch` parent operation that tracks the children. The parent's status reflects the aggregate — `pending` until at least one child is running, `processing` while children execute, `completed` once every child has finished, `failed` if any child failed. Each child is itself a `retain` operation linked to the parent, so you can drill in for per-batch error messages.

When you list operations, the parent and its children all appear by default. Pass `exclude_parents=true` to hide the aggregate rows and show only individual `retain` jobs.

### `file_convert_retain`

Submitted by file upload endpoints. The handler runs MIME-specific conversion (PDF → text, DOCX → text, etc.) and then passes the extracted text into the retain pipeline. Failures here are **non-retryable** by default — a corrupted PDF or missing OCR won't improve on rerun, so the operation goes straight to `failed`.

Which parser runs (`markitdown`, `iris`, or `llama_parse`) is selected per deployment via `HINDSIGHT_API_FILE_PARSER`, and clients can override it per request — see [Configuration → File Processing](../configuration#file-processing).

### `consolidation`

Produces **observations** from new world/experience memories. See [Observations](../observations) for what they are and how they're synthesized.

Triggered automatically:

- After every retain that added world/experience facts (gated by per-bank `enable_auto_consolidation` and `enable_observations`).
- After deletes that invalidated existing observations (the source memory disappeared → derived observations are stale → re-run with the surviving co-source memories).
- Manually via `POST /v1/default/banks/{bank_id}/consolidate`. Pass `observation_scopes` to consolidate only memories matching specific tag combinations.

**Bank-deduped**: while one `consolidation` job is pending for a bank, repeat submits return the existing `operation_id` instead of stacking. Once the job starts processing, the next submit becomes the next pending slot.

### `refresh_mental_model`

A mental model has a `source_query` that defines which memories it summarizes. The handler re-runs that query, re-summarizes the result, and updates the model's content in place.

Triggered either manually via `POST /v1/default/banks/{bank_id}/mental-models/{id}/refresh`, or automatically by the auto-refresh schedule for mental models that have one configured.

### `graph_maintenance`

Reconciles derived state that goes stale after a delete. Every invocation runs three passes:

1. **Link top-up.** Drains the `graph_maintenance_queue` (units whose outgoing temporal/semantic links lost a neighbour). For each, if the unit is under its cap (20 temporal, 50 semantic), Hindsight re-runs the same probes retain uses and inserts the missing links. Without this, the retain pipeline's top-K capping would leave surviving units permanently under-capped after every delete — degrading graph-expansion recall.
2. **Orphan entity prune.** Deletes entities in the bank with no remaining `unit_entities` references. FK `ON DELETE CASCADE` on `entity_cooccurrences` then removes any cooccurrence row pointing at a pruned entity.
3. **Stale cooccurrence prune.** Cleans up `entity_cooccurrences` rows where both endpoints still exist but no current memory_unit references both of them — the cooccurrence was real when it was recorded, but every unit that witnessed it has since been deleted.

Bank-deduped at submit time, so concurrent triggers against the same bank coalesce into one drain.

**Triggers:** any delete that removes memory_units — `DELETE /documents/{id}`, `DELETE /memories/{id}`, and re-retaining an existing `document_id` (the upsert path). A full bank wipe (`delete_bank`) is a no-op: there's nothing left in the bank to maintain.

### `webhook_delivery`

After certain operations complete (e.g., consolidation finishing on a bank with a registered webhook), Hindsight enqueues a `webhook_delivery` task. The handler POSTs the payload to the configured URL and retries on transient failures.

## Endpoints

All paths below are scoped by `bank_id`.

### List operations

```bash
GET /v1/default/banks/{bank_id}/operations
```

Query parameters:

| Param | Description |
|-------|-------------|
| `status` | Filter by `pending`, `processing`, `completed`, `failed`, `cancelled`. |
| `type` | Filter by `retain`, `file_convert_retain`, `consolidation`, `refresh_mental_model`, `graph_maintenance`, `webhook_delivery`. |
| `limit` | 1–100, default 20. |
| `offset` | Pagination offset. |
| `exclude_parents` | Exclude parent batch operations from results (large `retain_batch` calls create one parent + N children). |

### Python

```python
# Section 'operations-list' not found in api/operations.py
```

### Node.js

```javascript
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
```

### CLI

```bash
hindsight operation list my-bank
```

### Go

```go
# Section 'operations-list' not found in api/operations.go
```

`items_count` is operation-specific — non-zero only for retain-shaped operations (it counts content items in the submission).

### Get operation status

### Python

```python
# Section 'operations-get' not found in api/operations.py
```

### Node.js

```javascript
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
```

### CLI

```bash
hindsight operation get my-bank "$OPERATION_ID"
```

### Go

```go
# Section 'operations-get' not found in api/operations.go
```

Query parameters:

| Param | Description |
|-------|-------------|
| `include_payload` | Include the raw task payload (the submission params) in the response as `task_payload`. Default `false`; may be large. |

A few response fields are worth calling out:

| Field | Description |
|-------|-------------|
| `updated_at` | When the operation's row last changed — claim, progress heartbeat, or completion. |
| `progress` | Last-known progress snapshot for a running operation, or `null` if none was recorded (completed-instantly or pre-feature rows). |
| `task_payload` | The raw submission params; only populated when `include_payload=true`. |

`progress` is written at coarse phase/batch boundaries (consolidation, batch retain) and lets you tell a healthy long-running job from a frozen one: if `processed` keeps advancing across polls the job is alive; identical numbers with no movement in `at` mean it's stuck. Its shape:

| Field | Description |
|-------|-------------|
| `stage` | Coarse phase the operation last reported (e.g. `processing_batch`). |
| `at` | ISO-8601 timestamp when this snapshot was written. |
| `processed` | Units of work finished so far (sub-batches, memories), when known. |
| `total` | Total units of work for the operation, when known. |
| `detail` | Operation-specific counters (e.g. `observations_created`, `round`, `items_in_sub_batch`). |

### Cancel a pending operation

Returns `409` if the operation is already in `processing`, `completed`, or `failed` state.

### Python

```python
# Section 'operations-cancel' not found in api/operations.py
```

### Node.js

```javascript
// Cancel a pending operation before a worker claims it.
// Returns 409 if the operation is already processing/completed/failed.
await sdk.cancelOperation({
    client: apiClient,
    path: { bank_id: 'my-bank', operation_id: '550e8400-e29b-41d4-a716-446655440000' },
});
```

### CLI

```bash
hindsight operation cancel my-bank "$OPERATION_ID"
```

### Go

```go
# Section 'operations-cancel' not found in api/operations.go
```

### Retry a failed operation

The row's status resets to `pending` and the worker picks it up again. Returns `409` if the operation isn't in `failed` or `cancelled` state.

### Python

```python
# Section 'operations-retry' not found in api/operations.py
```

### Node.js

```javascript
// Re-queue a failed (or cancelled) operation.
// Returns 409 if the operation isn't in failed/cancelled state.
await sdk.retryOperation({
    client: apiClient,
    path: { bank_id: 'my-bank', operation_id: '550e8400-e29b-41d4-a716-446655440000' },
});
```

### CLI

```bash
hindsight operation retry my-bank "$OPERATION_ID"
```

### Go

```go
# Section 'operations-retry' not found in api/operations.go
```

## Async retain example

Submit a batch asynchronously and poll until the operation completes:

### Python

```python
# Section 'operations-async-retain' not found in api/operations.py
```

### Node.js

```javascript
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
```

### CLI

```bash
# Submit an async retain and capture the operation_id from the JSON response.
OPERATION_ID=$(
  hindsight memory retain my-bank "Alice joined Google in 2023" --async -o json \
    | jq -r '.operation_id'
)

# Poll until the worker finishes — completed/failed/cancelled are all terminal.
while true; do
  STATUS=$(hindsight operation get my-bank "$OPERATION_ID" -o json | jq -r '.status')
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ] || [ "$STATUS" = "cancelled" ]; then
    echo "finished: $STATUS"
    break
  fi
  sleep 2
done
```

### Go

```go
# Section 'operations-async-retain' not found in api/operations.go
```

## Worker tuning

Each worker has a single concurrency budget (`HINDSIGHT_API_WORKER_MAX_SLOTS`, default 10) shared across all operation types. Per-type slot reservations (`HINDSIGHT_API_WORKER_<TYPE>_MAX_SLOTS`) carve out guaranteed capacity within that budget; remaining slots form a shared pool any type can use. See [Configuration → Worker Configuration](../configuration#distributed-workers) for the full table.

For most deployments the defaults are fine. Reserve slots for an operation type if you've seen it starved by a flood of another type (e.g., a long file_convert_retain blocking graph_maintenance on a deletion-heavy workload).

## Next Steps

- [**Documents**](./documents) — Track document sources
- [**Memory Banks**](./memory-banks) — Configure bank settings
