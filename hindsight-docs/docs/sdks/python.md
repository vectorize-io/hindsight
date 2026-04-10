---
sidebar_position: 1
---

# Python Client

Official HTTP client for the Hindsight API. Use this when you have a Hindsight server already running — locally, in Docker, or as a managed service — and you want a typed Python client to talk to it.

If you want to **embed and run a Hindsight server in your Python process** (no external server required), see [Embedded Python (hindsight-all)](./hindsight-all.md) instead.

## Installation

```bash
pip install hindsight-client
```

## Quick Start

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

# Retain a memory
client.retain(bank_id="my-bank", content="Alice works at Google")

# Recall memories
results = client.recall(bank_id="my-bank", query="What does Alice do?")
for r in results.results:
    print(r.text)

# Reflect - generate a contextual answer
answer = client.reflect(bank_id="my-bank", query="Tell me about Alice")
print(answer.text)
```

## Client Initialization

```python
from hindsight_client import Hindsight

client = Hindsight(
    base_url="http://localhost:8888",  # Hindsight API URL
    timeout=30.0,                       # Request timeout in seconds
    # api_key="your-api-key",          # Optional bearer token
)

# Core operations
client.retain(bank_id="test", content="Hello world")
results = client.recall(bank_id="test", query="Hello")

# Organized API namespaces
client.banks.create(bank_id="test", name="Test Bank")
models = client.mental_models.list(bank_id="test")
directives = client.directives.list(bank_id="test")
memories = client.memories.list(bank_id="test")
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
    types=["world", "observation"],  # Filter by fact type
    max_tokens=4096,
    budget="high",  # low, mid, or high
)
```

### Recall with Chunks

```python
# Returns RecallResponse with source chunks
response = client.recall(
    bank_id="my-bank",
    query="What does Alice do?",
    types=["world", "experience"],
    budget="mid",
    max_tokens=4096,
    include_chunks=True,
    max_chunk_tokens=500
)

print(f"Found {len(response.results)} memories")
for r in response.results:
    print(f"  - {r.text}")
    if r.chunks:
        print(f"    Source: {r.chunks[0].text[:100]}...")
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
    mission="You're a helpful AI assistant - keep track of user preferences and conversation history.",
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
