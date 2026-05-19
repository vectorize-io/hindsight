
# Quick Start

Get up and running with Hindsight in 60 seconds.

{/* Import raw source files */}

## Clients

<ClientsGrid />

## Start the API Server

### pip (API only)

```bash
pip install hindsight-api
export OPENAI_API_KEY=sk-xxx
export HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY

hindsight-api
```

API available at [http://localhost:8888](http://localhost:8888/docs)

### Docker (Full Experience)

```bash

export OPENAI_API_KEY=sk-xxx

docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

- **API**: http://localhost:8888
- **Control Plane** (Web UI): http://localhost:9999

> **💡 Set a stable `HINDSIGHT_API_WORKER_ID` in production**
> 
The worker uses the container hostname as its identity, which Docker sets to the container ID by default. That value changes on every restart, so any task that was being processed when the container went down stays parked under the old ID with no way for the new container to recognize it as its own.

Set `HINDSIGHT_API_WORKER_ID` to a stable value (e.g., `-e HINDSIGHT_API_WORKER_ID=hindsight-prod`) so the worker keeps the same identity across restarts. This is recommended even for single-container deployments. For diagnosis and recovery commands, see [Admin CLI - Recovering stuck operations](../admin-cli.md#recovering-stuck-or-zombie-operations).
> **💡 LLM Provider**
> 
Hindsight requires an LLM with structured output support. Recommended: **Groq** with `gpt-oss-20b` for fast, cost-effective inference.
See [LLM Providers](../models.md#llm) for more details.
---

## Use the Client

### Python

```bash
pip install hindsight-client
```

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

# Retain: Store information
client.retain(bank_id="my-bank", content="Alice works at Google as a software engineer")

# Recall: Search memories
client.recall(bank_id="my-bank", query="What does Alice do?")

# Reflect: Generate disposition-aware response
client.reflect(bank_id="my-bank", query="Tell me about Alice")
```

### Node.js

```bash
npm install @vectorize-io/hindsight-client
```

```javascript
import { HindsightClient } from '@vectorize-io/hindsight-client';

const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

// Retain: Store information
await client.retain('my-bank', 'Alice works at Google as a software engineer');

// Recall: Search memories
await client.recall('my-bank', 'What does Alice do?');

// Reflect: Generate response
await client.reflect('my-bank', 'Tell me about Alice');
```

### CLI

```bash
curl -fsSL https://hindsight.vectorize.io/get-cli | bash
```

```bash
# Retain: Store information
hindsight memory retain my-bank "Alice works at Google as a software engineer"

# Recall: Search memories
hindsight memory recall my-bank "What does Alice do?"

# Reflect: Generate response
hindsight memory reflect my-bank "Tell me about Alice"
```

### Go

```bash
go get github.com/vectorize-io/hindsight/hindsight-clients/go
```

```go
# Section 'quickstart-full' not found in api/quickstart.go
```

---

## What's Happening

| Operation | What it does |
|-----------|--------------|
| **Retain** | Content is processed, facts are extracted, entities are identified and linked in a knowledge graph |
| **Recall** | Four search strategies (semantic, keyword, graph, temporal) run in parallel to find relevant memories |
| **Reflect** | Retrieved memories are used to generate a disposition-aware response |

---

## Integrations

Browse all supported integrations in the Integrations Hub.

## Next Steps

- [**Retain**](./retain) — Advanced options for storing memories
- [**Recall**](./recall) — Search and retrieval strategies
- [**Reflect**](./reflect) — Disposition-aware reasoning
- [**Memory Banks**](./memory-banks) — Configure disposition and mission
- [**Server Deployment**](../installation.md) — Docker Compose, Helm, and production setup
