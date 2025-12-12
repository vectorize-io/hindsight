# Configuration

Complete reference for configuring Hindsight server through environment variables and configuration files.

## Environment Variables

Hindsight is configured entirely through environment variables, making it easy to deploy across different environments and container orchestration platforms.

All environment variable names and defaults are defined in `hindsight_api.config`. You can use `MemoryEngine.from_env()` to create a MemoryEngine instance configured from environment variables:

```python
from hindsight_api import MemoryEngine

# Create from environment variables
memory = MemoryEngine.from_env()
await memory.initialize()
```

### LLM Provider Configuration

Configure the LLM provider used for fact extraction, entity resolution, and reasoning operations.

#### Common LLM Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HINDSIGHT_API_LLM_PROVIDER` | LLM provider: `groq`, `openai`, `gemini`, `ollama` | `groq` | Yes |
| `HINDSIGHT_API_LLM_API_KEY` | API key for LLM provider | - | Yes (except ollama) |
| `HINDSIGHT_API_LLM_MODEL` | Model name | Provider-specific | No |
| `HINDSIGHT_API_LLM_BASE_URL` | Custom LLM endpoint | Provider default | No |

#### Provider-Specific Examples

**Groq (Recommended for Fast Inference)**

```bash
export HINDSIGHT_API_LLM_PROVIDER=groq
export HINDSIGHT_API_LLM_API_KEY=gsk_xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=openai/gpt-oss-20b
```

**OpenAI**

```bash
export HINDSIGHT_API_LLM_PROVIDER=openai
export HINDSIGHT_API_LLM_API_KEY=sk-xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=gpt-4o
```

**Gemini**

```bash
export HINDSIGHT_API_LLM_PROVIDER=gemini
export HINDSIGHT_API_LLM_API_KEY=xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=gemini-2.0-flash
```

**Ollama (Local, No API Key)**

```bash
export HINDSIGHT_API_LLM_PROVIDER=ollama
export HINDSIGHT_API_LLM_BASE_URL=http://localhost:11434/v1
export HINDSIGHT_API_LLM_MODEL=llama3.1
```

**OpenAI-Compatible Endpoints**

```bash
export HINDSIGHT_API_LLM_PROVIDER=openai
export HINDSIGHT_API_LLM_BASE_URL=https://your-endpoint.com/v1
export HINDSIGHT_API_LLM_API_KEY=your-api-key
export HINDSIGHT_API_LLM_MODEL=your-model-name
```

### Database Configuration

Configure the PostgreSQL database connection and behavior.

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HINDSIGHT_API_DATABASE_URL` | PostgreSQL connection string | - | Yes* |

**\*Note**: If `DATABASE_URL` is not provided, the server will use embedded `pg0` (embedded PostGRE).

### MCP Server Configuration

Configure the Model Context Protocol (MCP) server for AI assistant integrations.

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HINDSIGHT_API_MCP_ENABLED` | Enable MCP server | `true` | No |

```bash
# Enable MCP server (default)
export HINDSIGHT_API_MCP_ENABLED=true

# Disable MCP server
export HINDSIGHT_API_MCP_ENABLED=false
```

### Embeddings Configuration

Configure the embeddings provider for semantic search. By default, uses local SentenceTransformers models.

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HINDSIGHT_API_EMBEDDINGS_PROVIDER` | Provider: `local` or `tei` | `local` | No |
| `HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL` | Model name for local provider | `BAAI/bge-small-en-v1.5` | No |
| `HINDSIGHT_API_EMBEDDINGS_TEI_URL` | TEI server URL | - | Yes (if provider is `tei`) |

**Local Provider (Default)**

Uses SentenceTransformers to run embedding models locally. Good for development and smaller deployments.

```bash
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
export HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-small-en-v1.5
```

**TEI Provider (HuggingFace Text Embeddings Inference)**

Uses a remote [TEI server](https://github.com/huggingface/text-embeddings-inference) for high-performance inference. Recommended for production deployments.

```bash
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=tei
export HINDSIGHT_API_EMBEDDINGS_TEI_URL=http://localhost:8080
```

:::warning
All embedding models must produce 384-dimensional vectors to match the database schema.
:::

### Reranker Configuration

Configure the cross-encoder reranker for improving search result relevance. By default, uses local SentenceTransformers models.

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HINDSIGHT_API_RERANKER_PROVIDER` | Provider: `local` or `tei` | `local` | No |
| `HINDSIGHT_API_RERANKER_LOCAL_MODEL` | Model name for local provider | `cross-encoder/ms-marco-MiniLM-L-6-v2` | No |
| `HINDSIGHT_API_RERANKER_TEI_URL` | TEI server URL | - | Yes (if provider is `tei`) |

**Local Provider (Default)**

Uses SentenceTransformers CrossEncoder to run reranking locally.

```bash
export HINDSIGHT_API_RERANKER_PROVIDER=local
export HINDSIGHT_API_RERANKER_LOCAL_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

**TEI Provider (HuggingFace Text Embeddings Inference)**

Uses a remote [TEI server](https://github.com/huggingface/text-embeddings-inference) with a reranker model.

```bash
export HINDSIGHT_API_RERANKER_PROVIDER=tei
export HINDSIGHT_API_RERANKER_TEI_URL=http://localhost:8081
```

:::tip
When using TEI, you can run separate servers for embeddings and reranking, or use a single server if it supports both operations with your chosen model.
:::

## Configuration Files

### .env File

The Hindsight API will look for a `.env` file:

```bash
# .env

# Database
HINDSIGHT_API_DATABASE_URL=postgresql://hindsight:hindsight_dev@localhost:5432/hindsight

# LLM
HINDSIGHT_API_LLM_PROVIDER=groq
HINDSIGHT_API_LLM_API_KEY=gsk_xxxxxxxxxxxx

# Embeddings (optional, defaults to local)
# HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
# HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-small-en-v1.5

# Reranker (optional, defaults to local)
# HINDSIGHT_API_RERANKER_PROVIDER=local
# HINDSIGHT_API_RERANKER_LOCAL_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

For configuration issues not covered here, please [open an issue](https://github.com/your-repo/hindsight/issues) on GitHub.
