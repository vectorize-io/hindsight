# Installation

Hindsight can be deployed in several ways depending on your infrastructure and requirements.

:::tip Don't want to manage infrastructure?
**[Hindsight Cloud](https://ui.hindsight.vectorize.io/signup)** is a fully managed service that handles all infrastructure, scaling, and maintenance — [sign up here](https://ui.hindsight.vectorize.io/signup).
:::

## Supported Platforms

Hindsight runs on **Linux**, **macOS**, and **Windows**:

| Platform | Docker | Bare Metal (pip) | Embedded DB (pg0) | Notes |
|----------|--------|------------------|--------------------|-------|
| **Linux** (x86_64, ARM64) | ✅ | ✅ | ✅ | Fully supported, recommended for production |
| **macOS** (Apple Silicon, Intel) | ✅ | ✅ | ✅ | Fully supported |
| **Windows** (x86_64) | ✅ | ✅ | ✅ | Fully supported — see [Windows setup](#windows) for external PostgreSQL option |

All platforms support the embedded database (pg0) for development. On Windows, you can also use an external PostgreSQL installation — see the [Windows](#windows) section for a step-by-step guide.

---

## Prerequisites

### PostgreSQL

Hindsight requires PostgreSQL 14+ with a vector extension for similarity search. The supported extensions are:

- **pgvector** (default)
- **pgvectorscale**
- **vchord**

Configure which one to use with `HINDSIGHT_API_VECTOR_EXTENSION`. See [Configuration](./configuration) for details.

**By default**, Hindsight uses **pg0** — an embedded PostgreSQL that runs locally on your machine. This is convenient for development but **not recommended for production**.

**For production**, use an external PostgreSQL with one of the supported vector extensions:
- **Supabase** — Managed PostgreSQL with pgvector built-in
- **Neon** — Serverless PostgreSQL with pgvector
- **Azure Database for PostgreSQL** — With pgvector and pgvectorscale support
- **AWS RDS** / **Cloud SQL** — With pgvector extension enabled
- **Self-hosted** — PostgreSQL 14+ with your preferred vector extension

### LLM Provider

You need an LLM API key for fact extraction, entity resolution, and answer generation. See [Models](./models) for supported providers, model recommendations, and configuration.

---

## Docker

**Best for**: Quick start, development, small deployments

Run everything in one container with embedded PostgreSQL:

```bash
export OPENAI_API_KEY=sk-xxx

docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

- **API Server**: http://localhost:8888
- **Control Plane** (Web UI): http://localhost:9999

### Docker Image Variants

| Variant | Size (AMD64) | Size (ARM64) | When to use |
|---------|--------------|--------------|-------------|
| **Full** (`latest`) | ~9 GB | ~3.7 GB | Default. Works out of the box with no external services except the LLM. |
| **Slim** (`slim`) | ~500 MB | ~500 MB | Use when you already rely on external services for embeddings and reranking (OpenAI, Cohere, TEI). Significantly smaller image, faster deploys. Requires [external providers](./configuration#embeddings). |

The slim image corresponds to the [`hindsight-api-slim`](#bare-metal-pip) pip package. See [Configuration](./configuration#embeddings) for external provider options.

### Available Tags

```bash
# Standalone (API + Control Plane)
ghcr.io/vectorize-io/hindsight:latest        # Full, latest release
ghcr.io/vectorize-io/hindsight:latest-slim          # Slim, latest release
ghcr.io/vectorize-io/hindsight:0.4.9         # Full, specific version
ghcr.io/vectorize-io/hindsight:0.4.9-slim    # Slim, specific version

# API only
ghcr.io/vectorize-io/hindsight-api:latest
ghcr.io/vectorize-io/hindsight-api:latest-slim

# Control Plane only
ghcr.io/vectorize-io/hindsight-control-plane:latest
```

---

## Helm / Kubernetes

**Best for**: Production deployments, auto-scaling, cloud environments

```bash
# Install with built-in PostgreSQL
helm install hindsight oci://ghcr.io/vectorize-io/charts/hindsight \
  --set api.llm.provider=groq \
  --set api.llm.apiKey=gsk_xxxxxxxxxxxx \
  --set postgresql.enabled=true

# Or use external PostgreSQL
helm install hindsight oci://ghcr.io/vectorize-io/charts/hindsight \
  --set api.llm.provider=groq \
  --set api.llm.apiKey=gsk_xxxxxxxxxxxx \
  --set postgresql.enabled=false \
  --set api.database.url=postgresql://user:pass@postgres.example.com:5432/hindsight

# Install a specific version
helm install hindsight oci://ghcr.io/vectorize-io/charts/hindsight --version 0.1.3

# Upgrade to latest
helm upgrade hindsight oci://ghcr.io/vectorize-io/charts/hindsight
```

**Requirements**:
- Kubernetes cluster (GKE, EKS, AKS, or self-hosted)
- Helm 3.8+

### Distributed Workers

For high-throughput deployments, enable dedicated worker pods to scale task processing independently:

```bash
helm install hindsight oci://ghcr.io/vectorize-io/charts/hindsight \
  --set worker.enabled=true \
  --set worker.replicaCount=3
```

See [Services - Worker Service](./services#worker-service) for configuration details and architecture.

See the [Helm chart values.yaml](https://github.com/vectorize-io/hindsight/tree/main/helm/hindsight/values.yaml) for all chart options.

---

## Bare Metal (pip)

**Best for**: Running Hindsight as a standalone service on a host machine.

### Install

```bash
pip install hindsight-api        # Full — works out of the box
pip install hindsight-api-slim   # Slim — requires external services for embeddings, reranking, and the database
```

When using `hindsight-api-slim`, you must configure external providers for all model operations. See [Configuration](./configuration#embeddings) for details.

### Run with Embedded Database

For development and testing, Hindsight can run with an embedded PostgreSQL (pg0):

```bash
export HINDSIGHT_API_LLM_PROVIDER=groq
export HINDSIGHT_API_LLM_API_KEY=gsk_xxxxxxxxxxxx

hindsight-api
```

This creates a database in `~/.hindsight/data/` and starts the API on http://localhost:8888.

### Run with External PostgreSQL

For production, connect to your own PostgreSQL instance:

```bash
export HINDSIGHT_API_DATABASE_URL=postgresql://user:pass@localhost:5432/hindsight
export HINDSIGHT_API_LLM_PROVIDER=groq
export HINDSIGHT_API_LLM_API_KEY=gsk_xxxxxxxxxxxx

hindsight-api
```

**Note**: The database must exist and have pgvector enabled (`CREATE EXTENSION vector;`).

### CLI Options

```bash
hindsight-api --port 9000          # Custom port (default: 8888)
hindsight-api --host 127.0.0.1     # Bind to localhost only
hindsight-api --workers 4          # Multiple worker processes
hindsight-api --log-level debug    # Verbose logging
```

### Control Plane

The Control Plane (Web UI) can be run standalone using npx:

```bash
npx @vectorize-io/hindsight-control-plane --api-url http://localhost:8888
```

This connects to your running API server and provides a visual interface for managing memory banks, exploring entities, and testing queries.

#### Options

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `-p, --port` | `PORT` | 9999 | Port to listen on |
| `-H, --hostname` | `HOSTNAME` | 0.0.0.0 | Hostname to bind to |
| `-a, --api-url` | `HINDSIGHT_CP_DATAPLANE_API_URL` | http://localhost:8888 | Hindsight API URL |

#### Examples

```bash
# Run on custom port
npx @vectorize-io/hindsight-control-plane --port 9999 --api-url http://localhost:8888

# Using environment variables
export HINDSIGHT_CP_DATAPLANE_API_URL=http://api.example.com
npx @vectorize-io/hindsight-control-plane

# Production deployment
PORT=80 HINDSIGHT_CP_DATAPLANE_API_URL=https://api.hindsight.io npx @vectorize-io/hindsight-control-plane
```

---

## Windows

**Best for**: Running Hindsight natively on Windows without Docker

Hindsight works on Windows with the embedded database (pg0) out of the box — just install and run:

```powershell
pip install hindsight-api

set HINDSIGHT_API_LLM_PROVIDER=openai
set HINDSIGHT_API_LLM_API_KEY=sk-xxx
set HINDSIGHT_API_LLM_MODEL=gpt-4o-mini

hindsight-api
```

### Using External PostgreSQL (optional)

If you prefer to use your own PostgreSQL instance instead of the embedded database:

```powershell
# Install PostgreSQL
winget install PostgreSQL.PostgreSQL.17

# Build pgvector (requires Visual Studio Build Tools)
git clone https://github.com/pgvector/pgvector.git
cd pgvector

# Open "x64 Native Tools Command Prompt for VS" and run:
set PGROOT=C:\Program Files\PostgreSQL\17
nmake /F Makefile.win
nmake /F Makefile.win install

# Create the database and enable the vector extension
psql -U postgres -c "CREATE DATABASE hindsight;"
psql -U postgres -d hindsight -c "CREATE EXTENSION vector;"
```

Then run Hindsight pointing to your database:

```powershell
pip install hindsight-api

set HINDSIGHT_API_DATABASE_URL=postgresql://postgres@localhost:5432/hindsight
set HINDSIGHT_API_LLM_PROVIDER=openai
set HINDSIGHT_API_LLM_API_KEY=sk-xxx
set HINDSIGHT_API_LLM_MODEL=gpt-4o-mini

hindsight-api
```

- **API Server**: http://localhost:8888

:::tip
You can also use the slim package (`pip install hindsight-api-slim`) if you configure external providers for embeddings and reranking. See [Configuration](./configuration#embeddings) for details.
:::

---

## Embedded in a Python Application

**Best for**: Using Hindsight programmatically from Python without running a separate server process.

```bash
pip install hindsight-all        # Full — works out of the box
pip install hindsight-all-slim   # Slim — requires external services for embeddings, reranking, and the database
```

`hindsight-all` supports two modes of embedding:

**In-process** (`HindsightServer`): the server runs in a background thread inside your application. Best when you want the tightest integration and are already managing your own process lifecycle.

```python
from hindsight import HindsightServer, HindsightClient

with HindsightServer(llm_provider="openai", llm_api_key="sk-xxx") as server:
    client = HindsightClient(base_url=server.url)
    client.retain(bank_id="alice", content="Alice prefers concise answers.")
    results = client.recall(bank_id="alice", query="How should I respond to Alice?")
```

**Managed subprocess** (`HindsightEmbedded`): the server runs as a background daemon process, shared across multiple Python processes or sessions. The daemon starts on first use and shuts down automatically after an idle timeout.

```python
from hindsight import HindsightEmbedded

client = HindsightEmbedded(llm_provider="openai", llm_api_key="sk-xxx")
client.retain(bank_id="alice", content="Alice prefers concise answers.")
results = client.recall(bank_id="alice", query="How should I respond to Alice?")
```

See the [Python SDK](../sdks/python.md) for the full API reference.

---

## Next Steps

- [Configuration](./configuration.md) — Environment variables and settings
- [Models](./models.mdx) — ML models and providers
- [Monitoring](./monitoring.md) — Metrics and observability
