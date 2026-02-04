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
    llm_model="gpt-4o-mini",
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

## Embedded Client (Easiest Option)

`HindsightEmbedded` provides the simplest way to use Hindsight in Python. It automatically manages a background server for you - no manual setup required:

```python
from hindsight import HindsightEmbedded
import os

# Server starts automatically on first use
client = HindsightEmbedded(
    profile="myapp",                        # Profile for data isolation
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    llm_api_key=os.environ["OPENAI_API_KEY"],
)

# Use immediately - no manual server management needed
client.retain(bank_id="my-bank", content="Alice works at Google")
results = client.recall(bank_id="my-bank", query="What does Alice do?")

# Server continues running (auto-stops after idle timeout)
# Or explicitly stop it:
client.close(stop_daemon=True)
```

**What's a Profile?**

A profile is an isolated Hindsight environment. Each profile gets its own PostgreSQL database (stored in `~/.pg0/instances/hindsight-embed-{profile}/`) and its own API server. Use different profiles to separate environments (dev/prod), applications, or users.

**When to Use HindsightEmbedded**

Use `HindsightEmbedded` when you want the server to start automatically and manage itself. Use `HindsightServer` when you need explicit control over server lifecycle (e.g., testing where you want immediate startup/shutdown).

**Advanced Operations**

`HindsightEmbedded` provides organized API namespaces for advanced operations. Each method call automatically ensures the daemon is running:

```python
from hindsight import HindsightEmbedded
import os

embedded = HindsightEmbedded(
    profile="myapp",
    llm_provider="openai",
    llm_api_key=os.environ["OPENAI_API_KEY"],
)

# Core operations (automatically proxied)
embedded.retain(bank_id="test", content="Hello")
results = embedded.recall(bank_id="test", query="Hello")

# Bank management
embedded.banks.create(bank_id="test", name="Test Bank", mission="Help users")
embedded.banks.set_mission(bank_id="test", mission="Updated mission")
embedded.banks.delete(bank_id="test")

# Mental models
embedded.mental_models.create(
    bank_id="test",
    name="User Preferences",
    content="User prefers dark mode"
)
models = embedded.mental_models.list(bank_id="test")
embedded.mental_models.update(bank_id="test", mental_model_id="...", content="New content")

# Directives
embedded.directives.create(
    bank_id="test",
    name="Response Style",
    content="Be concise and friendly"
)
directives = embedded.directives.list(bank_id="test")

# List memories
memories = embedded.memories.list(bank_id="test", type="world", limit=50)
```

**Why Use API Namespaces?**

API namespaces (`banks`, `mental_models`, `directives`, `memories`) ensure the daemon is running before each call. This handles daemon crashes gracefully:

```python
# ✅ GOOD - Uses API namespace (daemon restarts handled)
embedded.banks.create(bank_id="test", name="Test")

# ❌ BAD - Direct client access (daemon crashes NOT handled)
client = embedded.client
client.create_bank(bank_id="test", name="Test")  # Fails if daemon crashed
```

## Client Initialization

```python
from hindsight import HindsightClient

client = HindsightClient(
    base_url="http://localhost:8888",  # Hindsight API URL
    timeout=30.0,                       # Request timeout in seconds
)

# Core operations
client.retain(bank_id="test", content="Hello world")
results = client.recall(bank_id="test", query="Hello")

# Organized API access (same as HindsightEmbedded)
client.banks.create(bank_id="test", name="Test Bank")
models = client.mental_models.list(bank_id="test")
directives = client.directives.list(bank_id="test")
memories = client.memories.list(bank_id="test")
```

Both `HindsightClient` and `HindsightEmbedded` provide the same organized API namespaces (`banks`, `mental_models`, `directives`, `memories`) for consistent developer experience.

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
