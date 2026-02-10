
# Documents

Track and manage document sources in your memory bank. Documents provide traceability — knowing where memories came from.

{/* Import raw source files */}

> **💡 Prerequisites**
> 
Make sure you've completed the [Quick Start](./quickstart) and understand [how retain works](./retain).
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

> **💡 Include Chunks in Recall**
> 
Use `include_chunks=True` in your recall calls to get the original text chunks alongside fact results. See [Recall](./recall) for details.
## Retain with Document ID

Associate retained content with a document:

### Python

```python
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
```

### Node.js

```javascript
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

### CLI

```bash
# Retain content with document ID
hindsight memory retain my-bank "Meeting notes content..." --doc-id notes-2024-03-15

# Batch retain from files
hindsight memory retain-files my-bank docs/
```

## Update Documents

Re-retaining with the same document_id **replaces** the old content:

### Python

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

### Node.js

```javascript
// Original
await client.retain('my-bank', 'Project deadline: March 31', {
    document_id: 'project-plan'
});

// Update
await client.retain('my-bank', 'Project deadline: April 15 (extended)', {
    document_id: 'project-plan'
});
```

### CLI

```bash
# Original
hindsight memory retain my-bank "Project deadline: March 31" --doc-id project-plan

# Update
hindsight memory retain my-bank "Project deadline: April 15 (extended)" --doc-id project-plan
```

## Get Document

Retrieve a document's original text and metadata. This is useful for expanding document context after a recall operation returns memories with document references.

### Python

```python
import asyncio
from hindsight_client_api import ApiClient, Configuration
from hindsight_client_api.api import DocumentsApi

async def get_document_example():
    config = Configuration(host="http://localhost:8888")
    api_client = ApiClient(config)
    api = DocumentsApi(api_client)

    # Get document to expand context from recall results
    doc = await api.get_document(
        bank_id="my-bank",
        document_id="meeting-2024-03-15"
    )

    print(f"Document: {doc.id}")
    print(f"Original text: {doc.original_text}")
    print(f"Memory count: {doc.memory_unit_count}")
    print(f"Created: {doc.created_at}")

asyncio.run(get_document_example())
```

### Node.js

```javascript
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

### CLI

```bash
hindsight document get my-bank meeting-2024-03-15
```

## Delete Document

Remove a document and all its associated memories:

### Python

```python
from hindsight_client_api import ApiClient, Configuration
from hindsight_client_api.api import DocumentsApi

async def delete_document_example():
    config = Configuration(host="http://localhost:8888")
    api_client = ApiClient(config)
    api = DocumentsApi(api_client)

    # Delete document and all its memories
    result = await api.delete_document(
        bank_id="my-bank",
        document_id="meeting-2024-03-15"
    )

    print(f"Deleted {result.memory_units_deleted} memories")

asyncio.run(delete_document_example())
```

### Node.js

```javascript
// Delete document and all its memories
const { data: deleteResult } = await sdk.deleteDocument({
    client: apiClient,
    path: { bank_id: 'my-bank', document_id: 'meeting-2024-03-15' }
});

console.log(`Deleted ${deleteResult.memory_units_deleted} memories`);
```

### CLI

```bash
hindsight document delete my-bank meeting-2024-03-15
```

:::warning
Deleting a document permanently removes all memories extracted from it. This action cannot be undone.
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

- [**Operations**](./operations) — Monitor background tasks
- [**Memory Banks**](./memory-banks) — Configure bank settings
