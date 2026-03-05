
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

# Batch retain for a document with different sections
client.retain_batch(
    bank_id="my-bank",
    items=[
        {"content": "Item 1: Product launch delayed to Q2", "document_id": "meeting-2024-03-15-section-1"},
        {"content": "Item 2: New hiring targets announced", "document_id": "meeting-2024-03-15-section-2"},
        {"content": "Item 3: Budget approved for ML team", "document_id": "meeting-2024-03-15-section-3"}
    ]
)
```

### Node.js

```javascript
// Retain with document ID
await client.retain('my-bank', 'Alice presented the Q4 roadmap...', {
    document_id: 'meeting-2024-03-15'
});

// Batch retain for a document with different sections
await client.retainBatch('my-bank', [
    { content: 'Item 1: Product launch delayed to Q2', document_id: 'meeting-2024-03-15-section-1' },
    { content: 'Item 2: New hiring targets announced', document_id: 'meeting-2024-03-15-section-2' },
    { content: 'Item 3: Budget approved for ML team', document_id: 'meeting-2024-03-15-section-3' }
]);
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
// Get document to expand context from recall results
const { data: doc, error } = await sdk.getDocument({
    client: apiClient,
    path: { bank_id: 'my-bank', document_id: 'meeting-2024-03-15-section-1' }
});

if (error) {
    throw new Error(`Failed to get document: ${JSON.stringify(error)}`);
}

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
    path: { bank_id: 'my-bank', document_id: 'meeting-2024-03-15-section-1' }
});

console.log(`Deleted ${deleteResult.memory_units_deleted} memories`);
```

### CLI

```bash
hindsight document delete my-bank meeting-2024-03-15
```

:::warning
Deleting a document permanently removes all memories extracted from it. This action cannot be undone.
## List Documents

List documents in a bank with optional filtering by ID and tags.

### Python

```python
from hindsight_client_api import ApiClient, Configuration
from hindsight_client_api.api import DocumentsApi

async def list_documents_example():
    config = Configuration(host="http://localhost:8888")
    api_client = ApiClient(config)
    api = DocumentsApi(api_client)

    # List all documents
    result = await api.list_documents(bank_id="my-bank")
    print(f"Total documents: {result.total}")

    # Filter by document ID substring
    result = await api.list_documents(bank_id="my-bank", q="report")

    # Filter by tags — only docs tagged with "team-a" (untagged excluded)
    result = await api.list_documents(
        bank_id="my-bank",
        tags=["team-a"],
        tags_match="any_strict",
    )

    # Combine ID search and tags
    result = await api.list_documents(
        bank_id="my-bank",
        q="meeting",
        tags=["team-a", "team-b"],
        tags_match="all_strict",  # must have both tags
    )

    # Paginate
    result = await api.list_documents(bank_id="my-bank", limit=20, offset=40)
    print(f"Page items: {len(result.items)}")

import asyncio
asyncio.run(list_documents_example())
```

### Node.js

```javascript
const apiClient = createClient(createConfig({ baseUrl: 'http://localhost:8888' }));

// List all documents
const { data: allDocs } = await sdk.listDocuments({
    client: apiClient,
    path: { bank_id: 'my-bank' }
});
console.log(`Total documents: ${allDocs.total}`);

// Filter by document ID substring
const { data: reportDocs } = await sdk.listDocuments({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    query: { q: 'report' }
});

// Filter by tags — only docs tagged with "team-a" (untagged excluded)
const { data: taggedDocs } = await sdk.listDocuments({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    query: { tags: ['team-a'], tags_match: 'any_strict' }
});

// Combine ID search and tags
const { data: filtered } = await sdk.listDocuments({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    query: { q: 'meeting', tags: ['team-a', 'team-b'], tags_match: 'all_strict' }
});

// Paginate
const { data: page } = await sdk.listDocuments({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    query: { limit: 20, offset: 40 }
});
console.log(`Page items: ${page.items.length}`);
```

### CLI

```bash
# List all documents
hindsight document list my-bank

# Filter by ID substring
hindsight document list my-bank --q report

# Filter by tags
hindsight document list my-bank --tags team-a --tags team-b
```

### Filtering Options

| Parameter | Description |
|---|---|
| `q` | Case-insensitive substring match on document ID. `report` matches `report-2024`, `annual-report`, etc. |
| `tags` | Filter by document tags. Accepts multiple values. |
| `tags_match` | How to match tags (default: `any_strict`). See below. |
| `limit` / `offset` | Pagination. Default limit is 100. |

**`tags_match` modes:**

| Mode | Behaviour |
|---|---|
| `any_strict` *(default)* | Document must have **at least one** of the specified tags. Untagged docs excluded. |
| `any` | Same as `any_strict` but also includes untagged documents. |
| `all_strict` | Document must have **all** specified tags. Untagged docs excluded. |
| `all` | Same as `all_strict` but also includes untagged documents. |

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
