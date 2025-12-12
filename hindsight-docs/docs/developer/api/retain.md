---
sidebar_position: 2
---

# Ingest Data

Store documents, conversations, and raw content into Hindsight to automatically extract and create memories.

When you **retain** content, Hindsight doesn't just store the raw text—it intelligently analyzes the content to extract meaningful facts, identify entities, and build a connected knowledge graph. This process transforms unstructured information into structured, queryable memories.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::info How Retain Works
Learn about fact extraction, entity resolution, and graph construction in the [Retain Architecture](/developer/retain) guide.
:::

:::tip Prerequisites
Make sure you've completed the [Quick Start](./quickstart) to install the client and start the server.
:::

## Store a Single Memory

<Tabs>
<TabItem value="python" label="Python">

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

client.retain(
    bank_id="my-bank",
    content="Alice works at Google as a software engineer"
)
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
import { HindsightClient } from '@vectorize-io/hindsight-client';

const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

await client.retain('my-bank', 'Alice works at Google as a software engineer');
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
hindsight memory put my-bank "Alice works at Google as a software engineer"
```

</TabItem>
</Tabs>

## The Importance of Context

The `context` parameter is crucial for guiding how Hindsight extracts memories from your content. Think of it as providing a lens through which the system interprets the information.

**Why context matters:**
- **Steers memory extraction**: Context tells the memory bank what type of information to focus on and how to interpret ambiguous content
- **Improves relevance**: Memories extracted with proper context are more accurately categorized and easier to retrieve
- **Disambiguates meaning**: The same sentence can have different implications depending on context (e.g., "the project was terminated" means different things in a career vs. product context)

## Store with Context and Date

Always provide context and event dates for optimal memory extraction:

<Tabs>
<TabItem value="python" label="Python">

```python
client.retain(
    bank_id="my-bank",
    content="Alice got promoted to senior engineer",
    context="career update",
    timestamp="2024-03-15T10:00:00Z"
)
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
await client.retain('my-bank', 'Alice got promoted to senior engineer', {
    context: 'career update',
    timestamp: '2024-03-15T10:00:00Z'
});
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
hindsight memory put my-bank "Alice got promoted" \
    --context "career update" \
    --event-date "2024-03-15"
```

</TabItem>
</Tabs>

The `timestamp` defaults to the current time if not specified. Providing explicit timestamps enables temporal queries like "What happened last spring?"

## Batch Ingestion

Store multiple items in a single request. **Batch ingestion is the recommended approach** as it significantly improves performance by reducing network overhead and allowing Hindsight to optimize the memory extraction process across related content.

<Tabs>
<TabItem value="python" label="Python">

```python
client.retain_batch(
    bank_id="my-bank",
    items=[
        {"content": "Alice works at Google", "context": "career"},
        {"content": "Bob is a data scientist at Meta", "context": "career"},
        {"content": "Alice and Bob are friends", "context": "relationship"}
    ],
    document_id="conversation_001"
)
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
await client.retainBatch('my-bank', [
    { content: 'Alice works at Google', context: 'career' },
    { content: 'Bob is a data scientist at Meta', context: 'career' },
    { content: 'Alice and Bob are friends', context: 'relationship' }
], { documentId: 'conversation_001' });
```

</TabItem>
</Tabs>

The `document_id` groups related memories for later management.

## Store from Files

<Tabs>
<TabItem value="cli" label="CLI">

```bash
# Single file
hindsight memory put-files my-bank document.txt

# Multiple files
hindsight memory put-files my-bank doc1.txt doc2.md notes.txt

# With document ID
hindsight memory put-files my-bank report.pdf --document-id "q4-report"
```

</TabItem>
</Tabs>


## Async Ingestion

For large batches, use async ingestion to avoid blocking:

<Tabs>
<TabItem value="python" label="Python">

```python
# Start async ingestion (returns immediately)
result = client.retain_batch(
    bank_id="my-bank",
    items=[...large batch...],
    document_id="large-doc",
    retain_async=True
)

# Check if it was processed asynchronously
print(result.var_async)  # True
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
// Start async ingestion (returns immediately)
const result = await client.retainBatch('my-bank', largeItems, {
    documentId: 'large-doc',
    async: true
});

console.log(result.async);  // true
```

</TabItem>
</Tabs>

