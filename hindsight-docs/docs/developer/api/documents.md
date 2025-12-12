---
sidebar_position: 8
---

# Documents

Track and manage document sources in your memory bank. Documents provide traceability — knowing where memories came from.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::tip Prerequisites
Make sure you've completed the [Quick Start](./quickstart) and understand [how retain works](./retain).
:::

## What Are Documents?

Documents are containers for retained content. They help you:

- **Track sources** — Know which PDF, conversation, or file a memory came from
- **Update content** — Re-retain a document to update its facts
- **Delete in bulk** — Remove all memories from a document at once
- **Organize memories** — Group related facts by source

## Chunks

When you retain content, Hindsight splits it into chunks before extracting facts. These chunks are stored alongside the extracted memories, preserving the original text segments.

**Why chunks matter:**
- **Context preservation** — Chunks contain the raw text that generated facts, useful when you need the exact wording
- **Richer recall** — Including chunks in recall provides surrounding context for matched facts

:::tip Include Chunks in Recall
Use `include_chunks=True` in your recall calls to get the original text chunks alongside fact results. See [Recall](./recall) for details.
:::

## Retain with Document ID

Associate retained content with a document:

<Tabs>
<TabItem value="python" label="Python">

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

# Retain with document ID
client.retain(
    bank_id="my-bank",
    content="Alice presented the Q4 roadmap...",
    document_id="meeting-2024-03-15"
)

# Batch retain for a document
client.retain_batch(
    bank_id="my-bank",
    items=[
        {"content": "Item 1: Product launch delayed to Q2"},
        {"content": "Item 2: New hiring targets announced"},
        {"content": "Item 3: Budget approved for ML team"}
    ],
    document_id="meeting-2024-03-15"
)

# From file
with open("notes.txt") as f:
    client.retain(
        bank_id="my-bank",
        content=f.read(),
        document_id="notes-2024-03-15"
    )
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
import { HindsightClient } from '@vectorize-io/hindsight-client';

const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

// Retain with document ID
await client.retain('my-bank', 'Alice presented the Q4 roadmap...', {
    document_id: 'meeting-2024-03-15'
});

// Batch retain
await client.retainBatch('my-bank', [
    { content: 'Item 1: Product launch delayed to Q2' },
    { content: 'Item 2: New hiring targets announced' },
    { content: 'Item 3: Budget approved for ML team' }
], { documentId: 'meeting-2024-03-15' });
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
# Retain file with document ID
hindsight retain my-bank --file notes.txt --document-id notes-2024-03-15

# Batch retain directory
hindsight retain my-bank --files docs/*.md --document-id project-docs
```

</TabItem>
</Tabs>

## Update Documents

Re-retaining with the same document_id **replaces** the old content:

<Tabs>
<TabItem value="python" label="Python">

```python
# Original
client.retain(
    bank_id="my-bank",
    content="Project deadline: March 31",
    document_id="project-plan"
)

# Update (deletes old facts, creates new ones)
client.retain(
    bank_id="my-bank",
    content="Project deadline: April 15 (extended)",
    document_id="project-plan"
)
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
// Original
await client.retain('my-bank', 'Project deadline: March 31', {
    document_id: 'project-plan'
});

// Update
await client.retain('my-bank', 'Project deadline: April 15 (extended)', {
    document_id: 'project-plan'
});
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
# Original
hindsight retain my-bank "Project deadline: March 31" --document-id project-plan

# Update
hindsight retain my-bank "Project deadline: April 15 (extended)" --document-id project-plan
```

</TabItem>
</Tabs>

## Get Document

Retrieve a document's original text and metadata. This is useful for expanding document context after a recall operation returns memories with document references.

<Tabs>
<TabItem value="python" label="Python">

```python
from hindsight_client_api import ApiClient, Configuration
from hindsight_client_api.api import DefaultApi

config = Configuration(host="http://localhost:8888")
api_client = ApiClient(config)
api = DefaultApi(api_client)

# Get document to expand context from recall results
doc = api.get_document(
    bank_id="my-bank",
    document_id="meeting-2024-03-15"
)

print(f"Document: {doc.id}")
print(f"Original text: {doc.original_text}")
print(f"Memory count: {doc.memory_unit_count}")
print(f"Created: {doc.created_at}")
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
import { sdk, createClient, createConfig } from '@vectorize-io/hindsight-client';

const apiClient = createClient(createConfig({ baseUrl: 'http://localhost:8888' }));

// Get document to expand context from recall results
const { data: doc } = await sdk.getDocument({
    client: apiClient,
    path: { bank_id: 'my-bank', document_id: 'meeting-2024-03-15' }
});

console.log(`Document: ${doc.id}`);
console.log(`Original text: ${doc.original_text}`);
console.log(`Memory count: ${doc.memory_unit_count}`);
console.log(`Created: ${doc.created_at}`);
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
hindsight documents get my-bank meeting-2024-03-15
```

</TabItem>
</Tabs>

## Document Response Format

```json
{
  "id": "meeting-2024-03-15",
  "bank_id": "my-bank",
  "original_text": "Alice presented the Q4 roadmap...",
  "content_hash": "abc123def456",
  "memory_unit_count": 12,
  "created_at": "2024-03-15T14:00:00Z",
  "updated_at": "2024-03-15T14:00:00Z"
}
```

## Next Steps

- [**Entities**](./entities) — Track people, places, and concepts
- [**Operations**](./operations) — Monitor background tasks
- [**Memory Banks**](./memory-banks) — Configure bank settings
