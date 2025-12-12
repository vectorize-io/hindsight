---
sidebar_position: 1
---

# Python Client

Official Python client for the Hindsight API.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Installation

<Tabs>
<TabItem value="all-in-one" label="All-in-One (Recommended)">

The `hindsight-all` package includes embedded PostgreSQL, HTTP API server, and client:

```bash
pip install hindsight-all
```

</TabItem>
<TabItem value="client-only" label="Client Only">

If you already have a Hindsight server running:

```bash
pip install hindsight-client
```

</TabItem>
</Tabs>

## Quick Start

<Tabs>
<TabItem value="all-in-one" label="All-in-One">

```python
import os
from hindsight import HindsightServer, HindsightClient

with HindsightServer(
    llm_provider="openai",
    llm_model="gpt-4.1-mini",
    llm_api_key=os.environ["OPENAI_API_KEY"]
) as server:
    client = HindsightClient(base_url=server.url)

    # Retain a memory
    client.retain(bank_id="my-bank", content="Alice works at Google")

    # Recall memories
    results = client.recall(bank_id="my-bank", query="What does Alice do?")
    for r in results:
        print(r.text)

    # Reflect - generate response with disposition
    answer = client.reflect(bank_id="my-bank", query="Tell me about Alice")
    print(answer.text)
```

</TabItem>
<TabItem value="client-only" label="Client Only">

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

# Retain a memory
client.retain(bank_id="my-bank", content="Alice works at Google")

# Recall memories
results = client.recall(bank_id="my-bank", query="What does Alice do?")
for r in results:
    print(r.text)

# Reflect - generate response with disposition
answer = client.reflect(bank_id="my-bank", query="Tell me about Alice")
print(answer.text)
```

</TabItem>
</Tabs>

## Client Initialization

```python
from hindsight_client import Hindsight

client = Hindsight(
    base_url="http://localhost:8888",  # Hindsight API URL
    timeout=30.0,                       # Request timeout in seconds
)
```

## Core Operations

### Retain (Store Memory)

```python
# Simple
client.retain(
    bank_id="my-bank",
    content="Alice works at Google as a software engineer",
)

# With options
from datetime import datetime

client.retain(
    bank_id="my-bank",
    content="Alice got promoted",
    context="career update",
    timestamp=datetime(2024, 1, 15),
    document_id="conversation_001",
    metadata={"source": "slack"},
)
```

### Retain Batch

```python
client.retain_batch(
    bank_id="my-bank",
    items=[
        {"content": "Alice works at Google", "context": "career"},
        {"content": "Bob is a data scientist", "context": "career"},
    ],
    document_id="conversation_001",
    retain_async=False,  # Set True for background processing
)
```

### Recall (Search)

```python
# Simple - returns list of RecallResult
results = client.recall(
    bank_id="my-bank",
    query="What does Alice do?",
)

for r in results.results:
    print(f"{r.text} (type: {r.type})")

# With options
results = client.recall(
    bank_id="my-bank",
    query="What does Alice do?",
    types=["world", "opinion"],  # Filter by fact type
    max_tokens=4096,
    budget="high",  # low, mid, or high
)
```

### Recall with Full Response

```python
# Returns RecallResponse with entities and chunks
response = client.recall(
    bank_id="my-bank",
    query="What does Alice do?",
    types=["world", "experience"],
    budget="mid",
    max_tokens=4096,
    include_entities=True,
    max_entity_tokens=500
)

print(f"Found {len(response.results)} memories")
for r in response.results:
    print(f"  - {r.text}")

# Access entities
if response.entities:
    for entity in response.entities:
        print(f"Entity: {entity.name}")
```

### Reflect (Generate Response)

```python
answer = client.reflect(
    bank_id="my-bank",
    query="What should I know about Alice?",
    budget="low",  # low, mid, or high
    context="preparing for a meeting",
)

print(answer.text)  # Generated response
```

## Bank Management

### Create Bank

```python
client.create_bank(
    bank_id="my-bank",
    name="Assistant",
    background="I am a helpful AI assistant",
    disposition={
        "skepticism": 3,    # 1-5: trusting to skeptical
        "literalism": 3,    # 1-5: flexible to literal
        "empathy": 3,       # 1-5: detached to empathetic
    },
)
```

### List Memories

```python
client.list_memories(
    bank_id="my-bank",
    type="world",  # Optional: filter by type
    search_query="Alice",  # Optional: text search
    limit=100,
    offset=0,
)
```

## Async Support

All methods have async versions prefixed with `a`:

```python
import asyncio
from hindsight_client import Hindsight

async def main():
    client = Hindsight(base_url="http://localhost:8888")

    # Async retain
    await client.aretain(bank_id="my-bank", content="Hello world")

    # Async recall
    results = await client.arecall(bank_id="my-bank", query="Hello")
    for r in results:
        print(r.text)

    # Async reflect
    answer = await client.areflect(bank_id="my-bank", query="What did I say?")
    print(answer.text)

    client.close()

asyncio.run(main())
```

## Context Manager

```python
from hindsight_client import Hindsight

with Hindsight(base_url="http://localhost:8888") as client:
    client.retain(bank_id="my-bank", content="Hello")
    results = client.recall(bank_id="my-bank", query="Hello")
# Client automatically closed
```
