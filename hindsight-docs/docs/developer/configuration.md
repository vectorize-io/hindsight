# Configuration

Complete reference for configuring Hindsight services through environment variables.

Hindsight has two services, each with its own configuration prefix:

| Service | Prefix | Description |
|---------|--------|-------------|
| **API Service** | `HINDSIGHT_API_*` | Core memory engine |
| **Control Plane** | `HINDSIGHT_CP_*` | Web UI |

---

## API Service

The API service handles all memory operations (retain, recall, reflect).

### Database

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_DATABASE_URL` | PostgreSQL connection string | `pg0` (embedded) |
| `HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP` | Run database migrations on API startup | `true` |

If not provided, the server uses embedded `pg0` — convenient for development but not recommended for production.

### Database Connection Pool

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_DB_POOL_MIN_SIZE` | Minimum connections in the pool | `5` |
| `HINDSIGHT_API_DB_POOL_MAX_SIZE` | Maximum connections in the pool | `100` |
| `HINDSIGHT_API_DB_COMMAND_TIMEOUT` | PostgreSQL command timeout in seconds | `60` |
| `HINDSIGHT_API_DB_ACQUIRE_TIMEOUT` | Connection acquisition timeout in seconds | `30` |

For high-concurrency workloads, increase `DB_POOL_MAX_SIZE`. Each concurrent recall/think operation can use 2-4 connections.

To run migrations manually (e.g., before starting the API), use the admin CLI:

```bash
hindsight-admin run-db-migration
# Or for a specific schema:
hindsight-admin run-db-migration --schema tenant_acme
```

### LLM Provider

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_LLM_PROVIDER` | Provider: `openai`, `anthropic`, `gemini`, `groq`, `ollama`, `lmstudio` | `openai` |
| `HINDSIGHT_API_LLM_API_KEY` | API key for LLM provider | - |
| `HINDSIGHT_API_LLM_MODEL` | Model name | `gpt-5-mini` |
| `HINDSIGHT_API_LLM_BASE_URL` | Custom LLM endpoint | Provider default |
| `HINDSIGHT_API_LLM_MAX_CONCURRENT` | Max concurrent LLM requests | `32` |
| `HINDSIGHT_API_LLM_TIMEOUT` | LLM request timeout in seconds | `120` |
| `HINDSIGHT_API_LLM_GROQ_SERVICE_TIER` | Groq service tier: `on_demand`, `flex`, `auto` | `auto` |

**Provider Examples**

```bash
# Groq (recommended for fast inference)
export HINDSIGHT_API_LLM_PROVIDER=groq
export HINDSIGHT_API_LLM_API_KEY=gsk_xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=openai/gpt-oss-20b
# For free tier users: override to on_demand if you get service_tier errors
# export HINDSIGHT_API_LLM_GROQ_SERVICE_TIER=on_demand

# OpenAI
export HINDSIGHT_API_LLM_PROVIDER=openai
export HINDSIGHT_API_LLM_API_KEY=sk-xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=gpt-4o

# Gemini
export HINDSIGHT_API_LLM_PROVIDER=gemini
export HINDSIGHT_API_LLM_API_KEY=xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=gemini-2.0-flash

# Anthropic
export HINDSIGHT_API_LLM_PROVIDER=anthropic
export HINDSIGHT_API_LLM_API_KEY=sk-ant-xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=claude-sonnet-4-20250514

# Ollama (local, no API key)
export HINDSIGHT_API_LLM_PROVIDER=ollama
export HINDSIGHT_API_LLM_BASE_URL=http://localhost:11434/v1
export HINDSIGHT_API_LLM_MODEL=llama3

# LM Studio (local, no API key)
export HINDSIGHT_API_LLM_PROVIDER=lmstudio
export HINDSIGHT_API_LLM_BASE_URL=http://localhost:1234/v1
export HINDSIGHT_API_LLM_MODEL=your-local-model

# OpenAI-compatible endpoint
export HINDSIGHT_API_LLM_PROVIDER=openai
export HINDSIGHT_API_LLM_BASE_URL=https://your-endpoint.com/v1
export HINDSIGHT_API_LLM_API_KEY=your-api-key
export HINDSIGHT_API_LLM_MODEL=your-model-name
```

### Per-Operation LLM Configuration

Different memory operations have different requirements. **Retain** (fact extraction) benefits from models with strong structured output capabilities, while **Reflect** (reasoning/response generation) can use lighter, faster models. Configure separate LLM models for each operation to optimize for cost and performance.

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_RETAIN_LLM_PROVIDER` | LLM provider for retain operations | Falls back to `HINDSIGHT_API_LLM_PROVIDER` |
| `HINDSIGHT_API_RETAIN_LLM_API_KEY` | API key for retain LLM | Falls back to `HINDSIGHT_API_LLM_API_KEY` |
| `HINDSIGHT_API_RETAIN_LLM_MODEL` | Model for retain operations | Falls back to `HINDSIGHT_API_LLM_MODEL` |
| `HINDSIGHT_API_RETAIN_LLM_BASE_URL` | Base URL for retain LLM | Falls back to `HINDSIGHT_API_LLM_BASE_URL` |
| `HINDSIGHT_API_REFLECT_LLM_PROVIDER` | LLM provider for reflect operations | Falls back to `HINDSIGHT_API_LLM_PROVIDER` |
| `HINDSIGHT_API_REFLECT_LLM_API_KEY` | API key for reflect LLM | Falls back to `HINDSIGHT_API_LLM_API_KEY` |
| `HINDSIGHT_API_REFLECT_LLM_MODEL` | Model for reflect operations | Falls back to `HINDSIGHT_API_LLM_MODEL` |
| `HINDSIGHT_API_REFLECT_LLM_BASE_URL` | Base URL for reflect LLM | Falls back to `HINDSIGHT_API_LLM_BASE_URL` |

:::tip When to Use Per-Operation Config
- **Retain**: Use models with strong structured output (e.g., GPT-4o, Claude) for accurate fact extraction
- **Reflect**: Use faster/cheaper models (e.g., GPT-4o-mini, Groq) for reasoning and response generation
- **Recall**: Does not use LLM (pure retrieval), so no configuration needed
:::

**Example: Separate Models for Retain and Reflect**

```bash
# Default LLM (used as fallback)
export HINDSIGHT_API_LLM_PROVIDER=openai
export HINDSIGHT_API_LLM_API_KEY=sk-xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=gpt-4o

# Use GPT-4o for retain (strong structured output)
export HINDSIGHT_API_RETAIN_LLM_MODEL=gpt-4o

# Use faster/cheaper model for reflect
export HINDSIGHT_API_REFLECT_LLM_PROVIDER=groq
export HINDSIGHT_API_REFLECT_LLM_API_KEY=gsk_xxxxxxxxxxxx
export HINDSIGHT_API_REFLECT_LLM_MODEL=llama-3.3-70b-versatile
```

### Embeddings

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_EMBEDDINGS_PROVIDER` | Provider: `local`, `tei`, `openai`, or `cohere` | `local` |
| `HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL` | Model for local provider | `BAAI/bge-small-en-v1.5` |
| `HINDSIGHT_API_EMBEDDINGS_TEI_URL` | TEI server URL | - |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY` | OpenAI API key (falls back to `HINDSIGHT_API_LLM_API_KEY`) | - |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `HINDSIGHT_API_COHERE_API_KEY` | Cohere API key (shared for embeddings and reranker) | - |
| `HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL` | Cohere embedding model | `embed-english-v3.0` |

```bash
# Local (default) - uses SentenceTransformers
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
export HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-small-en-v1.5

# OpenAI - cloud-based embeddings
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
export HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=sk-xxxxxxxxxxxx  # or reuses HINDSIGHT_API_LLM_API_KEY
export HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=text-embedding-3-small  # 1536 dimensions

# TEI - HuggingFace Text Embeddings Inference (recommended for production)
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=tei
export HINDSIGHT_API_EMBEDDINGS_TEI_URL=http://localhost:8080

# Cohere - cloud-based embeddings
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=cohere
export HINDSIGHT_API_COHERE_API_KEY=your-api-key
export HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL=embed-english-v3.0  # 1024 dimensions
```

#### Embedding Dimensions

Hindsight automatically detects the embedding dimension from the model at startup and adjusts the database schema accordingly. The default model (`BAAI/bge-small-en-v1.5`) produces 384-dimensional vectors, while OpenAI models produce 1536 or 3072 dimensions.

:::warning Dimension Changes
Once memories are stored, you cannot change the embedding dimension without losing data. If you need to switch to a model with different dimensions:

1. **Empty database**: The schema is adjusted automatically on startup
2. **Existing data**: Either delete all memories first, or use a model with matching dimensions

Supported OpenAI embedding dimensions:
- `text-embedding-3-small`: 1536 dimensions
- `text-embedding-3-large`: 3072 dimensions
- `text-embedding-ada-002`: 1536 dimensions (legacy)
:::

### Reranker

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_RERANKER_PROVIDER` | Provider: `local`, `tei`, or `cohere` | `local` |
| `HINDSIGHT_API_RERANKER_LOCAL_MODEL` | Model for local provider | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `HINDSIGHT_API_RERANKER_LOCAL_MAX_CONCURRENT` | Max concurrent local reranking (prevents CPU thrashing under load) | `4` |
| `HINDSIGHT_API_RERANKER_TEI_URL` | TEI server URL | - |
| `HINDSIGHT_API_RERANKER_TEI_BATCH_SIZE` | Batch size for TEI reranking | `128` |
| `HINDSIGHT_API_RERANKER_TEI_MAX_CONCURRENT` | Max concurrent TEI reranking requests | `8` |
| `HINDSIGHT_API_RERANKER_COHERE_MODEL` | Cohere rerank model | `rerank-english-v3.0` |

```bash
# Local (default) - uses SentenceTransformers CrossEncoder
export HINDSIGHT_API_RERANKER_PROVIDER=local
export HINDSIGHT_API_RERANKER_LOCAL_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# TEI - for high-performance inference
export HINDSIGHT_API_RERANKER_PROVIDER=tei
export HINDSIGHT_API_RERANKER_TEI_URL=http://localhost:8081

# Cohere - cloud-based reranking
export HINDSIGHT_API_RERANKER_PROVIDER=cohere
export HINDSIGHT_API_COHERE_API_KEY=your-api-key  # shared with embeddings
export HINDSIGHT_API_RERANKER_COHERE_MODEL=rerank-english-v3.0
```

### Authentication

By default, Hindsight runs without authentication. For production deployments, enable API key authentication using the built-in tenant extension:

```bash
# Enable the built-in API key authentication
export HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
export HINDSIGHT_API_TENANT_API_KEY=your-secret-api-key
```

When enabled, all requests must include the API key in the `Authorization` header:

```bash
curl -H "Authorization: Bearer your-secret-api-key" \
  http://localhost:8888/v1/default/banks
```

Requests without a valid API key receive a `401 Unauthorized` response.

:::tip Custom Authentication
For advanced authentication (JWT, OAuth, multi-tenant schemas), implement a custom `TenantExtension`. See the [Extensions documentation](./extensions.md) for details.
:::

### Server

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_HOST` | Bind address | `0.0.0.0` |
| `HINDSIGHT_API_PORT` | Server port | `8888` |
| `HINDSIGHT_API_WORKERS` | Number of uvicorn worker processes | `1` |
| `HINDSIGHT_API_LOG_LEVEL` | Log level: `debug`, `info`, `warning`, `error` | `info` |
| `HINDSIGHT_API_MCP_ENABLED` | Enable MCP server at `/mcp/{bank_id}/` | `true` |

### Retrieval

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_GRAPH_RETRIEVER` | Graph retrieval algorithm: `link_expansion`, `mpfp`, or `bfs` | `link_expansion` |
| `HINDSIGHT_API_RECALL_MAX_CONCURRENT` | Max concurrent recall operations per worker (backpressure) | `32` |
| `HINDSIGHT_API_RERANKER_MAX_CANDIDATES` | Max candidates to rerank per recall (RRF pre-filters the rest) | `300` |

#### Graph Retrieval Algorithms

- **`link_expansion`** (default): Fast, simple graph expansion from semantic seeds via entity co-occurrence and causal links. Target latency under 100ms. Recommended for most use cases.
- **`mpfp`**: Multi-Path Fact Propagation - iterative graph traversal with activation spreading. More thorough but slower.
- **`bfs`**: Breadth-first search from seed facts. Simple but less effective for large graphs.

### Entity Observations

Controls when the system generates entity observations (summaries about entities mentioned in retained content).

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_OBSERVATION_MIN_FACTS` | Minimum facts about an entity before generating observations | `5` |
| `HINDSIGHT_API_OBSERVATION_TOP_ENTITIES` | Max entities to process per retain batch | `5` |

### Retain

Controls the retain (memory ingestion) pipeline.

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS` | Max completion tokens for fact extraction LLM calls | `64000` |
| `HINDSIGHT_API_RETAIN_CHUNK_SIZE` | Max characters per chunk for fact extraction. Larger chunks extract fewer LLM calls but may lose context. | `3000` |
| `HINDSIGHT_API_RETAIN_EXTRACTION_MODE` | Fact extraction mode: `concise` (selective, fewer high-quality facts) or `verbose` (detailed, more facts) | `concise` |
| `HINDSIGHT_API_RETAIN_EXTRACT_CAUSAL_LINKS` | Extract causal relationships between facts | `true` |
| `HINDSIGHT_API_RETAIN_OBSERVATIONS_ASYNC` | Run entity observation generation asynchronously (after retain completes) | `false` |

#### Extraction Modes

The extraction mode controls how aggressively facts are extracted from content:

- **`concise`** (default): Selective extraction that focuses on significant, long-term valuable facts. Filters out greetings, filler, and trivial information. Produces fewer but higher-quality facts with better performance.

- **`verbose`**: Detailed extraction that captures every piece of information with maximum verbosity. Produces more facts with extensive detail but slower performance and higher token usage.

### Local MCP Server

Configuration for the local MCP server (`hindsight-local-mcp` command).

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_MCP_LOCAL_BANK_ID` | Memory bank ID for local MCP | `mcp` |
| `HINDSIGHT_API_MCP_INSTRUCTIONS` | Additional instructions appended to retain/recall tool descriptions | - |

```bash
# Example: instruct MCP to also store assistant actions
export HINDSIGHT_API_MCP_INSTRUCTIONS="Also store every action you take, including tool calls and decisions made."
```

### Background Tasks

Controls background task processing for async operations like opinion formation and entity observations.

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_TASK_BACKEND` | Task backend implementation: `memory` (in-process queue) or `noop` (discard tasks, useful for tests) | `memory` |
| `HINDSIGHT_API_TASK_BACKEND_MEMORY_BATCH_SIZE` | Max tasks to process in one batch (memory backend only) | `10` |
| `HINDSIGHT_API_TASK_BACKEND_MEMORY_BATCH_INTERVAL` | Interval between batch processing in seconds (memory backend only) | `1.0` |

### Performance Optimization

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_SKIP_LLM_VERIFICATION` | Skip LLM connection check on startup | `false` |
| `HINDSIGHT_API_LAZY_RERANKER` | Lazy-load reranker model (faster startup) | `false` |

### Programmatic Configuration

You can also configure the API programmatically using `MemoryEngine.from_env()`:

```python
from hindsight_api import MemoryEngine

memory = MemoryEngine.from_env()
await memory.initialize()
```

---

## Control Plane

The Control Plane is the web UI for managing memory banks.

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_CP_DATAPLANE_API_URL` | URL of the API service | `http://localhost:8888` |

```bash
# Point Control Plane to a remote API service
export HINDSIGHT_CP_DATAPLANE_API_URL=http://api.example.com:8888
```

---

## Example .env File

```bash
# API Service
HINDSIGHT_API_DATABASE_URL=postgresql://hindsight:hindsight_dev@localhost:5432/hindsight
HINDSIGHT_API_LLM_PROVIDER=groq
HINDSIGHT_API_LLM_API_KEY=gsk_xxxxxxxxxxxx

# Authentication (optional, recommended for production)
# HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
# HINDSIGHT_API_TENANT_API_KEY=your-secret-api-key

# Control Plane
HINDSIGHT_CP_DATAPLANE_API_URL=http://localhost:8888
```

---

For configuration issues not covered here, please [open an issue](https://github.com/vectorize-io/hindsight/issues) on GitHub.
