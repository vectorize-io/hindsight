
# Ingest Data

Store documents, conversations, and raw content into Hindsight to automatically extract and create memories.

When you **retain** content, Hindsight doesn't just store the raw text—it intelligently analyzes the content to extract meaningful facts, identify entities, and build a connected knowledge graph. This process transforms unstructured information into structured, queryable memories.

{/* Import raw source files */}

:::info How Retain Works
Learn about fact extraction, entity resolution, and graph construction in the [Retain Architecture](/developer/retain) guide.
> **💡 Prerequisites**
> 
Make sure you've completed the [Quick Start](./quickstart) to install the client and start the server.
## Store a Document

A single retain call accepts one or more **items**. Each item is a piece of raw content — a conversation, a document, a note — that Hindsight will analyze and decompose into one or many memories. The content itself is never stored verbatim; what gets stored are the structured facts the LLM extracts from it.

### Python

```python
client.retain(
    bank_id="my-bank",
    content="Alice works at Google as a software engineer"
)
```

### Node.js

```javascript
await client.retain('my-bank', 'Alice works at Google as a software engineer');
```

### CLI

```bash
hindsight memory retain my-bank "Alice works at Google as a software engineer"
```

### Retaining a Conversation

A full conversation should be retained as a single item. The LLM can parse any format — plain text, JSON, Markdown, or any structured representation — as long as it clearly conveys who said what and when. The example below uses a simple `Name (timestamp): text` format.

### Python

```python
# Retain an entire conversation as a single document.
# Format each message as "Name (timestamp): text" so the LLM can attribute
# facts to the right person and resolve temporal references across the thread.
conversation = "\n".join([
    "Alice (2024-03-15T09:00:00Z): Hi Bob! Did you end up going to the doctor last week?",
    "Bob (2024-03-15T09:01:00Z): Yes, finally. Turns out I have a mild peanut allergy.",
    "Alice (2024-03-15T09:02:00Z): Oh no! Are you okay?",
    "Bob (2024-03-15T09:03:00Z): Yeah, nothing serious. Just need to carry an antihistamine.",
    "Alice (2024-03-15T09:04:00Z): Good to know. We'll avoid peanuts at the team lunch.",
])

client.retain(
    bank_id="my-bank",
    content=conversation,
    context="team chat",
    timestamp="2024-03-15T09:04:00Z",
    document_id="chat-2024-03-15-alice-bob",
)
```

### Node.js

```javascript
// Retain an entire conversation as a single document.
// Format each message as "Name (timestamp): text" so the LLM can attribute
// facts to the right person and resolve temporal references across the thread.
const conversation = [
    'Alice (2024-03-15T09:00:00Z): Hi Bob! Did you end up going to the doctor last week?',
    'Bob (2024-03-15T09:01:00Z): Yes, finally. Turns out I have a mild peanut allergy.',
    'Alice (2024-03-15T09:02:00Z): Oh no! Are you okay?',
    'Bob (2024-03-15T09:03:00Z): Yeah, nothing serious. Just need to carry an antihistamine.',
    'Alice (2024-03-15T09:04:00Z): Good to know. We\'ll avoid peanuts at the team lunch.',
].join('\n');

await client.retain('my-bank', conversation, {
    context: 'team chat',
    timestamp: '2024-03-15T09:04:00Z',
    documentId: 'chat-2024-03-15-alice-bob',
});
```

When the conversation grows — a new message arrives — just retain again with the full updated content and the same `document_id`. Hindsight will delete the previous version and reprocess from scratch, so memories always reflect the latest state of the conversation.

---

## Parameters

### content

The raw text to store. This is the only required field. Hindsight chunks the content, sends each chunk to the LLM for fact extraction, and stores the resulting structured facts — not the original text. A single `content` value can produce many memories depending on how much information it contains.

### timestamp

When the event described in the content actually occurred. Accepts any ISO 8601 string (e.g., `"2024-01-15T10:30:00Z"`). If omitted, defaults to the current time at ingestion.

The timestamp is injected verbatim into the LLM fact-extraction prompt so the model can resolve relative temporal references in the content — for example, if the content says "last Monday", the model uses the provided timestamp as the anchor to pin down the actual date. It also enables temporal recall queries like "What happened last spring?" to work correctly.

### context

A short label describing the source or situation — for example `"team meeting"`, `"slack"`, or `"support ticket"`. It is injected directly into the LLM prompt, so it actively shapes how facts are extracted. The same sentence can mean something very different depending on context: "the project was terminated" in a `"performance review"` context versus a `"product roadmap"` context produces different memories.

Providing context consistently is one of the highest-leverage things you can do to improve memory quality.

### Python

```python
client.retain(
    bank_id="my-bank",
    content="Alice got promoted to senior engineer",
    context="career update",
    timestamp="2024-03-15T10:00:00Z"
)
```

### Node.js

```javascript
await client.retain('my-bank', 'Alice got promoted to senior engineer', {
    context: 'career update',
    timestamp: '2024-03-15T10:00:00Z'
});
```

### CLI

```bash
hindsight memory retain my-bank "Alice got promoted" \
    --context "career update"
```

### metadata

Arbitrary key-value string pairs attached to every fact extracted from this item. For example: `{"source": "slack", "channel": "engineering", "thread_id": "T123"}`. The LLM never sees this field — it is passed through as-is and stored on each memory unit. During recall, every returned memory includes its metadata, which lets you do client-side filtering or static enrichment without extra lookups — for example, linking a memory back to its source URL, thread ID, or any application-specific identifier.

### document_id

A caller-supplied string that groups one or more items under a logical document. This field is the key to making retain **idempotent**.

When you provide a `document_id`, Hindsight upserts the document: if a document with that ID already exists in the bank, it and all its associated memories are deleted before the new content is processed and inserted. This means you can safely re-run retain on updated content — for example, a chat thread that grew since last time — without accumulating duplicate memories.

If you omit `document_id`, Hindsight assigns a random UUID per request, so re-ingesting the same content will create duplicate memories.

### entities

A list of entities you want to guarantee are recognized, merged with any entities the LLM extracts automatically. Each entry has a `text` field (the entity name) and an optional `type` (e.g., `"PERSON"`, `"ORG"`, `"CONCEPT"` — defaults to `"CONCEPT"` if omitted).

Use this when you know certain entities are important but the LLM might miss them or refer to them inconsistently across different parts of the content. Providing entities explicitly ensures they are always linked in the knowledge graph.

### tags and document_tags

Tags control **visibility scoping** — which memories are visible during recall. A memory is only returned if its tags intersect with the tags filter provided in the recall request. This makes tags useful when a single memory bank serves multiple users or sessions and each should only see their own memories.

Use consistent naming patterns to keep tag filtering predictable. Common conventions: `user:<id>` for per-user scoping, `session:<id>` for session isolation, `room:<id>` for chat rooms, `topic:<name>` for category filtering. The bank also exposes a list-tags endpoint that returns all tags with their memory counts, useful for UI autocomplete or wildcard expansion.

See [Recall API](./recall#filter-by-tags) for filtering by tags during retrieval.

### Response

The synchronous retain response includes:

- `success` — whether the operation completed without errors
- `bank_id` — the memory bank that received the content
- `items_count` — number of items processed
- `async` — whether processing ran asynchronously
- `usage` — token usage for the LLM calls (`input_tokens`, `output_tokens`, `total_tokens`), only present for synchronous operations

---

## Batch Ingestion

Multiple items can be submitted in a single request. Batch ingestion is the recommended approach — it reduces network overhead and lets Hindsight optimize extraction across related content.

### Python

```python
client.retain_batch(
    bank_id="my-bank",
    items=[
        {"content": "Alice works at Google", "context": "career", "document_id": "conversation_001_msg_1"},
        {"content": "Bob is a data scientist at Meta", "context": "career", "document_id": "conversation_001_msg_2"},
        {"content": "Alice and Bob are friends", "context": "relationship", "document_id": "conversation_001_msg_3"}
    ]
)
```

### Node.js

```javascript
await client.retainBatch('my-bank', [
    { content: 'Alice works at Google', context: 'career', document_id: 'conversation_001_msg_1' },
    { content: 'Bob is a data scientist at Meta', context: 'career', document_id: 'conversation_001_msg_2' },
    { content: 'Alice and Bob are friends', context: 'relationship', document_id: 'conversation_001_msg_3' }
]);
```

---

## Files

Upload files directly — Hindsight converts them to text and extracts memories automatically. File processing always runs asynchronously and returns operation IDs for tracking.

**Supported formats:** PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, images (JPG, PNG, GIF, etc. — OCR), audio (MP3, WAV, FLAC, etc. — transcription), HTML, and plain text formats (TXT, MD, CSV, JSON, YAML, etc.)

### CLI

```bash
# Upload a single file (PDF, DOCX, PPTX, XLSX, images, audio, and more)
hindsight memory retain-files my-bank report.pdf

# Upload a directory of files
hindsight memory retain-files my-bank ./documents/

# Upload and wait for processing to complete (polls until done)
hindsight memory retain-files my-bank report.pdf

# Queue files for background processing (returns immediately)
hindsight memory retain-files my-bank ./documents/ --async
```

### HTTP

```bash
# Via HTTP API (multipart/form-data)
curl -X POST "${HINDSIGHT_URL}/v1/default/banks/my-bank/files/retain" \
    -F "files=@report.pdf;type=application/octet-stream" \
    -F "request={\"files_metadata\": [{\"context\": \"quarterly report\"}]}"
```

### Python

```python
# Upload files and retain their contents as memories.
# Supports: PDF, DOCX, PPTX, XLSX, images (OCR), audio (transcription), and text formats.
# Pass file paths — the client reads and uploads them automatically.
result = client.retain_files(
    bank_id="my-bank",
    files=[EXAMPLES_DIR / "sample.pdf"],
    context="quarterly report",
)
print(result.operation_ids)  # Track processing via the operations endpoint
```

### Node.js

```javascript
// Upload files and retain their contents as memories.
// Supports: PDF, DOCX, PPTX, XLSX, images (OCR), audio (transcription), and text formats.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const pdfBytes = readFileSync(join(__dirname, 'sample.pdf'));
const result = await client.retainFiles('my-bank', [
    new File([pdfBytes], 'sample.pdf'),
], { context: 'quarterly report' });
console.log(result.operation_ids);  // Track processing via the operations endpoint
```

The file retain endpoint always returns asynchronously. The response contains `operation_ids` — one per uploaded file — which you can poll via `GET /v1/default/banks/{bank_id}/operations` to track progress.

Upload up to 10 files per request (max 100 MB total). Each file becomes a separate document with optional per-file metadata:

### Python

```python
# Upload multiple files with per-file metadata (up to 10 files per batch).
# Each file gets its own context, document_id, and tags.
result = client.retain_files(
    bank_id="my-bank",
    files=[
        EXAMPLES_DIR / "sample.pdf",
        EXAMPLES_DIR / "sample.pdf",  # Replace with a second file path
    ],
    files_metadata=[
        {"context": "quarterly report", "document_id": "q1-report", "tags": ["project:alpha"]},
        {"context": "meeting notes", "document_id": "q1-notes", "tags": ["project:alpha"]},
    ],
)
print(result.operation_ids)  # One operation ID per file
```

:::info File Storage
Uploaded files are stored server-side (PostgreSQL by default, or S3/GCS/Azure for production). Configure storage via `HINDSIGHT_API_FILE_STORAGE_TYPE`. See [Configuration](../configuration#file-processing) for details.
---

## Async Ingestion

For large batches, use async ingestion to avoid blocking your application:

### Python

```python
# Start async ingestion (returns immediately)
result = client.retain_batch(
    bank_id="my-bank",
    items=[
        {"content": "Large batch item 1", "document_id": "large-doc-1"},
        {"content": "Large batch item 2", "document_id": "large-doc-2"},
    ],
    retain_async=True
)

# Check if it was processed asynchronously
print(result.var_async)  # True
```

### Node.js

```javascript
// Start async ingestion (returns immediately)
await client.retainBatch('my-bank', [
    { content: 'Large batch item 1', document_id: 'large-doc-1' },
    { content: 'Large batch item 2', document_id: 'large-doc-2' },
], {
    async: true
});
```

When `async: true`, the call returns immediately with an `operation_id`. Processing runs in the background via the worker service. No `usage` metrics are returned for async operations.

### Cut Costs 50% with Provider Batch APIs

When using async retain, enable the provider Batch API to reduce LLM fact-extraction costs by 50%. OpenAI and Groq both offer this discount in exchange for a processing window of up to 24 hours — a trade-off that's typically invisible when retain already runs in the background.

```bash
export HINDSIGHT_API_RETAIN_BATCH_ENABLED=true
```

Hindsight submits fact extraction calls as a batch job to the provider, polls for completion, and processes results automatically. No changes to your API calls are needed.

:::note
Batch API cost savings require `async=true` in your retain request and a compatible provider (OpenAI or Groq).
