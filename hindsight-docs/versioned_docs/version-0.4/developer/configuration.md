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
| `HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER` | LLM provider for observation consolidation | Falls back to `HINDSIGHT_API_LLM_PROVIDER` |
| `HINDSIGHT_API_CONSOLIDATION_LLM_API_KEY` | API key for consolidation LLM | Falls back to `HINDSIGHT_API_LLM_API_KEY` |
| `HINDSIGHT_API_CONSOLIDATION_LLM_MODEL` | Model for consolidation operations | Falls back to `HINDSIGHT_API_LLM_MODEL` |
| `HINDSIGHT_API_CONSOLIDATION_LLM_BASE_URL` | Base URL for consolidation LLM | Falls back to `HINDSIGHT_API_LLM_BASE_URL` |

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
| `HINDSIGHT_API_EMBEDDINGS_PROVIDER` | Provider: `local`, `tei`, `openai`, `cohere`, or `litellm` | `local` |
| `HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL` | Model for local provider | `BAAI/bge-small-en-v1.5` |
| `HINDSIGHT_API_EMBEDDINGS_TEI_URL` | TEI server URL | - |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY` | OpenAI API key (falls back to `HINDSIGHT_API_LLM_API_KEY`) | - |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL` | Custom base URL for OpenAI-compatible API (e.g., Azure OpenAI) | - |
| `HINDSIGHT_API_COHERE_API_KEY` | Cohere API key (shared for embeddings and reranker) | - |
| `HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL` | Cohere embedding model | `embed-english-v3.0` |
| `HINDSIGHT_API_EMBEDDINGS_COHERE_BASE_URL` | Custom base URL for Cohere-compatible API (e.g., Azure-hosted) | - |
| `HINDSIGHT_API_LITELLM_API_BASE` | LiteLLM proxy base URL (shared for embeddings and reranker) | `http://localhost:4000` |
| `HINDSIGHT_API_LITELLM_API_KEY` | LiteLLM proxy API key (optional, depends on proxy config) | - |
| `HINDSIGHT_API_EMBEDDINGS_LITELLM_MODEL` | LiteLLM embedding model (use provider prefix, e.g., `cohere/embed-english-v3.0`) | `text-embedding-3-small` |

```bash
# Local (default) - uses SentenceTransformers
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
export HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-small-en-v1.5

# OpenAI - cloud-based embeddings
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
export HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=sk-xxxxxxxxxxxx  # or reuses HINDSIGHT_API_LLM_API_KEY
export HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=text-embedding-3-small  # 1536 dimensions

# Azure OpenAI - embeddings via Azure endpoint
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
export HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=your-azure-api-key
export HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=text-embedding-3-small
export HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment

# TEI - HuggingFace Text Embeddings Inference (recommended for production)
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=tei
export HINDSIGHT_API_EMBEDDINGS_TEI_URL=http://localhost:8080

# Cohere - cloud-based embeddings
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=cohere
export HINDSIGHT_API_COHERE_API_KEY=your-api-key
export HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL=embed-english-v3.0  # 1024 dimensions

# Azure-hosted Cohere - embeddings via custom endpoint
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=cohere
export HINDSIGHT_API_COHERE_API_KEY=your-azure-api-key
export HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL=embed-english-v3.0
export HINDSIGHT_API_EMBEDDINGS_COHERE_BASE_URL=https://your-azure-cohere-endpoint.com

# LiteLLM proxy - unified gateway for multiple providers
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=litellm
export HINDSIGHT_API_LITELLM_API_BASE=http://localhost:4000
export HINDSIGHT_API_LITELLM_API_KEY=your-litellm-key  # optional
export HINDSIGHT_API_EMBEDDINGS_LITELLM_MODEL=text-embedding-3-small  # or cohere/embed-english-v3.0
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
| `HINDSIGHT_API_RERANKER_PROVIDER` | Provider: `local`, `tei`, `cohere`, `flashrank`, `litellm`, or `rrf` | `local` |
| `HINDSIGHT_API_RERANKER_LOCAL_MODEL` | Model for local provider | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `HINDSIGHT_API_RERANKER_LOCAL_MAX_CONCURRENT` | Max concurrent local reranking (prevents CPU thrashing under load) | `4` |
| `HINDSIGHT_API_RERANKER_TEI_URL` | TEI server URL | - |
| `HINDSIGHT_API_RERANKER_TEI_BATCH_SIZE` | Batch size for TEI reranking | `128` |
| `HINDSIGHT_API_RERANKER_TEI_MAX_CONCURRENT` | Max concurrent TEI reranking requests | `8` |
| `HINDSIGHT_API_RERANKER_COHERE_MODEL` | Cohere rerank model | `rerank-english-v3.0` |
| `HINDSIGHT_API_RERANKER_COHERE_BASE_URL` | Custom base URL for Cohere-compatible API (e.g., Azure-hosted) | - |
| `HINDSIGHT_API_RERANKER_LITELLM_MODEL` | LiteLLM rerank model (use provider prefix, e.g., `cohere/rerank-english-v3.0`) | `cohere/rerank-english-v3.0` |
| `HINDSIGHT_API_RERANKER_FLASHRANK_MODEL` | FlashRank model for fast CPU-based reranking | `ms-marco-MiniLM-L-12-v2` |
| `HINDSIGHT_API_RERANKER_FLASHRANK_CACHE_DIR` | Cache directory for FlashRank models | System default |

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

# Azure-hosted Cohere - reranking via custom endpoint
export HINDSIGHT_API_RERANKER_PROVIDER=cohere
export HINDSIGHT_API_COHERE_API_KEY=your-azure-api-key
export HINDSIGHT_API_RERANKER_COHERE_MODEL=rerank-english-v3.0
export HINDSIGHT_API_RERANKER_COHERE_BASE_URL=https://your-azure-cohere-endpoint.com

# LiteLLM proxy - unified gateway for multiple reranking providers
export HINDSIGHT_API_RERANKER_PROVIDER=litellm
export HINDSIGHT_API_LITELLM_API_BASE=http://localhost:4000
export HINDSIGHT_API_LITELLM_API_KEY=your-litellm-key  # optional
export HINDSIGHT_API_RERANKER_LITELLM_MODEL=cohere/rerank-english-v3.0  # or voyage/rerank-2, together_ai/...
```

LiteLLM supports multiple reranking providers via the `/rerank` endpoint:
- Cohere (`cohere/rerank-english-v3.0`, `cohere/rerank-multilingual-v3.0`)
- Together AI (`together_ai/...`)
- Voyage AI (`voyage/rerank-2`)
- Jina AI (`jina_ai/...`)
- AWS Bedrock (`bedrock/...`)

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
| `HINDSIGHT_API_LOG_FORMAT` | Log format: `text` or `json` (structured logging for cloud platforms) | `text` |
| `HINDSIGHT_API_MCP_ENABLED` | Enable MCP server at `/mcp/{bank_id}/` | `true` |

### Retrieval

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_GRAPH_RETRIEVER` | Graph retrieval algorithm: `link_expansion`, `mpfp`, or `bfs` | `link_expansion` |
| `HINDSIGHT_API_RECALL_MAX_CONCURRENT` | Max concurrent recall operations per worker (backpressure) | `32` |
| `HINDSIGHT_API_RECALL_CONNECTION_BUDGET` | Max concurrent DB connections per recall operation | `4` |
| `HINDSIGHT_API_RERANKER_MAX_CANDIDATES` | Max candidates to rerank per recall (RRF pre-filters the rest) | `300` |
| `HINDSIGHT_API_MPFP_TOP_K_NEIGHBORS` | Fan-out limit per node in MPFP graph traversal | `20` |
| `HINDSIGHT_API_MENTAL_MODEL_REFRESH_CONCURRENCY` | Max concurrent mental model refreshes | `8` |

#### Graph Retrieval Algorithms

- **`link_expansion`** (default): Fast, simple graph expansion from semantic seeds via entity co-occurrence and causal links. Target latency under 100ms. Recommended for most use cases.
- **`mpfp`**: Multi-Path Fact Propagation - iterative graph traversal with activation spreading. More thorough but slower.
- **`bfs`**: Breadth-first search from seed facts. Simple but less effective for large graphs.

### Retain

Controls the retain (memory ingestion) pipeline.

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS` | Max completion tokens for fact extraction LLM calls | `64000` |
| `HINDSIGHT_API_RETAIN_CHUNK_SIZE` | Max characters per chunk for fact extraction. Larger chunks extract fewer LLM calls but may lose context. | `3000` |
| `HINDSIGHT_API_RETAIN_EXTRACTION_MODE` | Fact extraction mode: `concise`, `verbose`, or `custom` | `concise` |
| `HINDSIGHT_API_RETAIN_CUSTOM_INSTRUCTIONS` | Custom extraction guidelines (only used when mode is `custom`) | - |
| `HINDSIGHT_API_RETAIN_EXTRACT_CAUSAL_LINKS` | Extract causal relationships between facts | `true` |

#### Extraction Modes

The extraction mode controls how aggressively facts are extracted from content:

- **`concise`** (default): Selective extraction that focuses on significant, long-term valuable facts. Filters out greetings, filler, and trivial information. Produces fewer but higher-quality facts with better performance.

- **`verbose`**: Detailed extraction that captures every piece of information with maximum verbosity. Produces more facts with extensive detail but slower performance and higher token usage.

- **`custom`**: Inject your own extraction guidelines while keeping the structural parts of the prompt (output format, coreference resolution, temporal handling, etc.) intact. Useful for A/B testing different extraction strategies or domain-specific customization.

**Example: Custom Extraction Mode**

```bash
# Set mode to custom
export HINDSIGHT_API_RETAIN_EXTRACTION_MODE=custom

# Define custom guidelines (multi-line is fine)
export HINDSIGHT_API_RETAIN_CUSTOM_INSTRUCTIONS="ONLY extract facts that are:
✅ Technical decisions and their rationale
✅ Architecture patterns and design choices
✅ Performance metrics and benchmarks
✅ Code reviews and feedback

DO NOT extract:
❌ Generic greetings or pleasantries
❌ Process chatter (\"let me check\", \"one moment\")
❌ Repeated information already captured

CONSOLIDATE related technical discussions into ONE fact when possible.

Ask yourself: 'Would this technical context be useful in 6 months?' If no, skip it."
```

### Observations (Experimental)

Observations are consolidated knowledge synthesized from facts.

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_ENABLE_OBSERVATIONS` | Enable observation consolidation | `true` |
| `HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE` | Memories to load per batch (internal optimization) | `50` |
| `HINDSIGHT_API_RETAIN_OBSERVATIONS_ASYNC` | Run observation generation asynchronously (after retain completes) | `false` |

### Reflect

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_REFLECT_MAX_ITERATIONS` | Max tool call iterations before forcing a response | `10` |

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

### Distributed Workers

Configuration for background task processing. By default, the API processes tasks internally. For high-throughput deployments, run dedicated workers. See [Services - Worker Service](./services#worker-service) for details.

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_WORKER_ENABLED` | Enable internal worker in API process | `true` |
| `HINDSIGHT_API_WORKER_ID` | Unique worker identifier | hostname |
| `HINDSIGHT_API_WORKER_POLL_INTERVAL_MS` | Database polling interval in milliseconds | `500` |
| `HINDSIGHT_API_WORKER_BATCH_SIZE` | Tasks to claim per poll cycle | `10` |
| `HINDSIGHT_API_WORKER_MAX_RETRIES` | Max retries before marking task failed | `3` |
| `HINDSIGHT_API_WORKER_HTTP_PORT` | HTTP port for worker metrics/health (worker CLI only) | `8889` |

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
