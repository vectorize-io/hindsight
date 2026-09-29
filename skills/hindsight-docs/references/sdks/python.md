

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
client.retain(bank_id="python-sdk-bank", content="Alice works at Google")

# Recall memories
results = client.recall(bank_id="python-sdk-bank", query="What does Alice do?")
for r in results.results:
    print(r.text)

# Reflect - generate a contextual answer
answer = client.reflect(bank_id="python-sdk-bank", query="Tell me about Alice")
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
client.retain(bank_id="python-sdk-test", content="Hello world")
results = client.recall(bank_id="python-sdk-test", query="Hello")

# Bank, mental model, directive and memory helpers
client.create_bank(bank_id="python-sdk-test", name="Test Bank")
models = client.list_mental_models(bank_id="python-sdk-test")
directives = client.list_directives(bank_id="python-sdk-test")
memories = client.list_memories(bank_id="python-sdk-test")
```

For operations the helper methods don't cover, the client also exposes the full generated API as async namespaces (`client.banks`, `client.memory`, `client.documents`, `client.mental_models`, `client.directives`, ...). Their methods must be awaited.

## Core Operations

### Version and Feature Checks

```python
version = client.get_version()

print(version.api_version)

if not version.features.mcp:
    raise RuntimeError("This server does not expose the MCP endpoint")
```

The async client method is available as `await client.aget_version()`.

### Retain (Store Memory)

```python
# Simple
client.retain(
    bank_id="python-sdk-bank",
    content="Alice works at Google as a software engineer",
)

# With options
from datetime import datetime

client.retain(
    bank_id="python-sdk-bank",
    content="Alice got promoted",
    context="career update",
    timestamp=datetime(2024, 1, 15),
    document_id="conversation_001",
    metadata={"source": "slack"},
    retain_async=False,  # Set True for background processing
)
```

### Retain Batch

```python
client.retain_batch(
    bank_id="python-sdk-bank",
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
# Simple - returns a RecallResponse
results = client.recall(
    bank_id="python-sdk-bank",
    query="What does Alice do?",
)

for r in results.results:
    print(f"{r.text} (type: {r.type})")

# With options
results = client.recall(
    bank_id="python-sdk-bank",
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
    bank_id="python-sdk-bank",
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
    chunk = (response.chunks or {}).get(r.chunk_id)
    if chunk:
        print(f"    Source: {chunk.text[:100]}...")
```

### Reflect (Generate Response)

```python
answer = client.reflect(
    bank_id="python-sdk-bank",
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
    bank_id="python-sdk-bank",
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
    bank_id="python-sdk-bank",
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
    await client.aretain(bank_id="python-sdk-bank", content="Hello world")

    # Async recall
    results = await client.arecall(bank_id="python-sdk-bank", query="Hello")
    for r in results.results:
        print(r.text)

    # Async reflect
    answer = await client.areflect(bank_id="python-sdk-bank", query="What did I say?")
    print(answer.text)

    await client.aclose()

asyncio.run(main())
```

## Context Manager

```python
from hindsight_client import Hindsight

with Hindsight(base_url="http://localhost:8888") as client:
    client.retain(bank_id="python-sdk-bank", content="Hello")
    results = client.recall(bank_id="python-sdk-bank", query="Hello")
# Client automatically closed
```
