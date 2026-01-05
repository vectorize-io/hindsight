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

If not provided, the server uses embedded `pg0` — convenient for development but not recommended for production.

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

### Embeddings

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_EMBEDDINGS_PROVIDER` | Provider: `local`, `tei`, or `openai` | `local` |
| `HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL` | Model for local provider | `BAAI/bge-small-en-v1.5` |
| `HINDSIGHT_API_EMBEDDINGS_TEI_URL` | TEI server URL | - |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY` | OpenAI API key (falls back to `HINDSIGHT_API_LLM_API_KEY`) | - |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL` | OpenAI embedding model | `text-embedding-3-small` |

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
| `HINDSIGHT_API_RERANKER_PROVIDER` | Provider: `local` or `tei` | `local` |
| `HINDSIGHT_API_RERANKER_LOCAL_MODEL` | Model for local provider | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `HINDSIGHT_API_RERANKER_TEI_URL` | TEI server URL | - |

```bash
# Local (default) - uses SentenceTransformers CrossEncoder
export HINDSIGHT_API_RERANKER_PROVIDER=local
export HINDSIGHT_API_RERANKER_LOCAL_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# TEI - for high-performance inference
export HINDSIGHT_API_RERANKER_PROVIDER=tei
export HINDSIGHT_API_RERANKER_TEI_URL=http://localhost:8081
```

### Server

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_HOST` | Bind address | `0.0.0.0` |
| `HINDSIGHT_API_PORT` | Server port | `8888` |
| `HINDSIGHT_API_LOG_LEVEL` | Log level: `debug`, `info`, `warning`, `error` | `info` |
| `HINDSIGHT_API_MCP_ENABLED` | Enable MCP server at `/mcp/{bank_id}/` | `true` |

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

### Retrieval

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_GRAPH_RETRIEVER` | Graph retrieval algorithm: `bfs` or `mpfp` | `bfs` |

### Entity Observations

Controls when the system generates entity observations (summaries about entities mentioned in retained content).

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_OBSERVATION_MIN_FACTS` | Minimum facts about an entity before generating observations | `5` |
| `HINDSIGHT_API_OBSERVATION_TOP_ENTITIES` | Max entities to process per retain batch | `5` |

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
