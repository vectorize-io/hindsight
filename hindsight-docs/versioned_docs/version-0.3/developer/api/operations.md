---
sidebar_position: 9
---

# Operations

Background tasks that Hindsight executes asynchronously.

:::tip Prerequisites
Make sure you've completed the [Quick Start](./quickstart) and understand [how retain works](./retain).
:::

## How Operations Work

Hindsight processes several types of tasks in the background to maintain memory quality and consistency. These operations run automatically—you don't need to trigger them manually.

By default, all background operations are executed in-process within the API service.

:::note Kafka Integration
Support for external streaming platforms like Kafka for scale-out processing is planned but **not available out of the box** in the current release.
:::

## Operation Types

| Operation | Trigger | Description |
|-----------|---------|-------------|
| **batch_retain** | `retain_batch` with `async=True` | Processes large content batches in the background |
| **form_opinion** | After each `reflect` call | Extracts and stores new opinions formed during reflection |
| **reinforce_opinion** | After `retain` | Updates opinion confidence based on new supporting evidence |
| **regenerate_observations** | Bank profile update | Regenerates entity observations when disposition changes |

## Async Retain Example

When retaining large batches of memories, use `async=true` to process in the background. The response includes an `operation_id` that you can use to poll for completion.

### 1. Submit async retain request

```bash
curl -X POST "http://localhost:8000/v1/default/banks/my-bank/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"content": "Alice joined Google in 2023"},
      {"content": "Bob prefers Python over JavaScript"}
    ],
    "async": true
  }'
```

Response:
```json
{
  "success": true,
  "bank_id": "my-bank",
  "items_count": 2,
  "async": true,
  "operation_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 2. Poll for operation status

```bash
curl "http://localhost:8000/v1/default/banks/my-bank/operations"
```

Response:
```json
{
  "bank_id": "my-bank",
  "operations": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "task_type": "retain",
      "items_count": 2,
      "document_id": null,
      "created_at": "2024-01-15T10:30:00Z",
      "status": "completed",
      "error_message": null
    }
  ]
}
```

### Operation Status Values

| Status | Description |
|--------|-------------|
| `pending` | Operation is queued and waiting to be processed |
| `completed` | Operation finished successfully |
| `failed` | Operation failed (check `error_message` for details) |

## Next Steps

- [**Documents**](./documents) — Track document sources
- [**Entities**](./entities) — Monitor entity tracking
- [**Memory Banks**](./memory-banks) — Configure bank settings
