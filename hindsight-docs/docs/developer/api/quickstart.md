---
sidebar_position: 0
---

# Quick Start

Get up and running with Hindsight in 60 seconds.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Installation

<Tabs>
<TabItem value="all-in-one" label="All-in-One (Recommended)">

The `hindsight-all` package includes everything you need: embedded PostgreSQL, HTTP API server, and Python client.

```bash
pip install hindsight-all
```

</TabItem>
<TabItem value="client-only" label="Client Only">

If you already have a Hindsight server running, install just the client:

```bash
pip install hindsight-client
```

</TabItem>
</Tabs>

## Basic Usage

<Tabs>
<TabItem value="all-in-one" label="All-in-One">

```python
import os
from hindsight import HindsightServer, HindsightClient

# Start embedded server (PostgreSQL + HTTP API)
with HindsightServer(
    llm_provider="openai",
    llm_model="gpt-4.1-mini",
    llm_api_key=os.environ["OPENAI_API_KEY"]
) as server:
    client = HindsightClient(base_url=server.url)

    # Store memories
    client.put(agent_id="my-agent", content="Alice works at Google")
    client.put(agent_id="my-agent", content="Bob prefers Python over JavaScript")

    # Search memories
    results = client.search(agent_id="my-agent", query="What does Alice do?")
    for r in results:
        print(r["text"])

    # Generate response with personality
    response = client.think(agent_id="my-agent", query="Tell me about Alice")
    print(response["text"])
```

</TabItem>
<TabItem value="client-only" label="Client Only">

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

# Store memories
client.put(agent_id="my-agent", content="Alice works at Google")
client.put(agent_id="my-agent", content="Bob prefers Python over JavaScript")

# Search memories
results = client.search(agent_id="my-agent", query="What does Alice do?")
for r in results:
    print(r["text"])

# Generate response with personality
response = client.think(agent_id="my-agent", query="Tell me about Alice")
print(response["text"])
```

</TabItem>
</Tabs>

## What's Happening

1. **Store** — Content is processed, facts are extracted, and entities are linked in a knowledge graph
2. **Search** — Four search strategies (semantic, keyword, graph, temporal) run in parallel and results are fused
3. **Think** — Retrieved memories are used to generate a personality-aware response

## Server Options

When using `HindsightServer`, you can configure:

```python
from hindsight import HindsightServer

server = HindsightServer(
    # Database
    db_url="pg0",                    # "pg0" for embedded PostgreSQL, or a connection URL

    # LLM Configuration
    llm_provider="openai",           # "openai", "groq", or "ollama"
    llm_api_key="your-api-key",
    llm_model="gpt-4.1-mini",
    llm_base_url=None,               # Custom endpoint (for ollama or proxies)

    # Server
    host="127.0.0.1",
    port=None,                       # Auto-select free port if None
    mcp_enabled=False,               # Enable MCP API
)

server.start()
print(f"Server running at {server.url}")

# ... use server ...

server.stop()
```

## Environment Variables

For production, use environment variables:

```bash
export OPENAI_API_KEY=sk-...
# or
export GROQ_API_KEY=gsk_...
```

```python
import os
from hindsight import HindsightServer, HindsightClient

with HindsightServer(
    llm_provider="openai",
    llm_api_key=os.environ["OPENAI_API_KEY"],
    llm_model="gpt-4.1-mini"
) as server:
    client = HindsightClient(base_url=server.url)
    # ...
```

## Next Steps

- [Ingest Data](./ingest) — Store memories, conversations, and documents
- [Search Facts](./search) — Multi-strategy retrieval
- [Think](./think) — Personality-aware response generation
- [Agent Identity](./agent-identity) — Configure agent personality and background
