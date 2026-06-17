
# Webhooks

Hindsight can notify your application in real-time when memory events occur by sending HTTP POST requests to a URL you configure.

## Delivery and Retries

Webhooks are registered per memory bank and fire automatically when matching events occur. Each delivery attempt is tracked, and failed deliveries are retried with exponential backoff:

| Attempt | Delay after failure |
|---------|---------------------|
| 1 | 5 seconds |
| 2 | 5 minutes |
| 3 | 30 minutes |
| 4 | 2 hours |
| 5 | 5 hours |
| 6 | Permanent failure |

A delivery is considered failed if your endpoint returns a non-2xx status code or does not respond within the configured timeout (default 30 seconds). After 6 failed attempts, the delivery is marked as permanently failed and no further retries are made.

> **ℹ️ At-least-once delivery**
> 
Webhook delivery tasks are queued in the same database transaction as the primary operation (e.g. the retain or consolidation write). This means if the server crashes after committing but before sending, the delivery task survives and will be retried. As a result, **your endpoint may receive the same event more than once** — use the `operation_id` field to deduplicate if needed.
## Event Types

### `consolidation.completed`

Fired after Hindsight finishes consolidating new memories into observations for a bank.

**Payload:**

```json
{
  "event": "consolidation.completed",
  "bank_id": "my-bank",
  "operation_id": "a1b2c3d4e5f6",
  "status": "completed",
  "timestamp": "2026-03-04T12:00:00Z",
  "data": {
    "observations_created": 3,
    "observations_updated": 1,
    "observations_deleted": null,
    "error_message": null
  }
}
```

**`data` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `observations_created` | `integer \| null` | Number of new observations created |
| `observations_updated` | `integer \| null` | Number of existing observations updated |
| `observations_deleted` | `integer \| null` | Number of observations deleted |
| `error_message` | `string \| null` | Set when `status` is `"failed"` |

**`status` values:** `"completed"` or `"failed"`

---

### `retain.completed`

Fired once per document after a retain operation completes (both synchronous and asynchronous). When retaining a batch of N documents, N separate events are fired.

**Payload:**

```json
{
  "event": "retain.completed",
  "bank_id": "my-bank",
  "operation_id": "a1b2c3d4e5f6",
  "status": "completed",
  "timestamp": "2026-03-04T12:00:01Z",
  "data": {
    "document_id": "doc-abc123",
    "tags": ["meeting", "q1-2026"]
  }
}
```

**`data` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `document_id` | `string \| null` | The document ID if one was provided in the retain request |
| `tags` | `string[] \| null` | Document-level tags applied during retain |

**Notes:**
- For async retain (`async: true`), `operation_id` matches the `operation_id` returned by the retain API.
- For sync retain, `operation_id` is a generated identifier for tracing purposes.
- One event is fired per content item in the retain request.

---

### `memory_defense.triggered`

Fired when a bank's [Memory Defense](../memory-defense/index.md) policy acts on a retained item — once per item that is **redacted** or **blocked**. Items that pass cleanly do not fire an event. Requires a Memory Defense policy enabled on the bank and a webhook subscribed to this event type.

**Payload:**

```json
{
  "event": "memory_defense.triggered",
  "bank_id": "my-bank",
  "operation_id": "a1b2c3d4e5f6",
  "status": "redact",
  "timestamp": "2026-03-04T12:00:02Z",
  "data": {
    "action": "redact",
    "detector": "sensitive_data",
    "document_id": "doc-abc123",
    "matched_types": ["github_token", "aws_access_key"],
    "message": "Sensitive data pattern matched: github_token, aws_access_key"
  }
}
```

**`data` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `action` | `string` | Action taken on the item: `"redact"` or `"block"` |
| `detector` | `string \| null` | The detector that matched (`"sensitive_data"`) |
| `document_id` | `string \| null` | The document ID if one was provided in the retain request |
| `matched_types` | `string[] \| null` | Labels of the redaction patterns that fired (e.g. `github_token`, `ssn_us`) |
| `message` | `string \| null` | Human-readable summary of what matched |

**`status` values:** mirrors `data.action` — `"redact"` or `"block"`.

**Notes:**
- A `redact` event means the secret was scrubbed and the redacted memory was still stored. A `block` event means the item was dropped; if every item in the retain request is blocked, the retain call returns `422`.

