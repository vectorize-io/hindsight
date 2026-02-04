# Models

Hindsight uses several machine learning models for different tasks.

## Overview

| Model Type | Purpose | Default | Configurable |
|------------|---------|---------|--------------|
| **LLM** | Fact extraction, reasoning, generation | Provider-specific | Yes |
| **Embedding** | Vector representations for semantic search | `BAAI/bge-small-en-v1.5` | Yes |
| **Cross-Encoder** | Reranking search results | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Yes |

All local models (embedding, cross-encoder) are automatically downloaded from HuggingFace on first run.

---

## LLM

Used for fact extraction, entity resolution, mental model consolidation, and answer synthesis.

**Supported providers:** OpenAI, Anthropic, Gemini, Groq, Ollama, LM Studio, and **any OpenAI-compatible API**

:::tip OpenAI-Compatible Providers
Hindsight works with any provider that exposes an OpenAI-compatible API (e.g., Azure OpenAI). Simply set `HINDSIGHT_API_LLM_PROVIDER=openai` and configure `HINDSIGHT_API_LLM_BASE_URL` to point to your provider's endpoint.

See [Configuration](./configuration#llm-provider) for setup examples.
:::

### Tested Models

The following models have been tested and verified to work correctly with Hindsight:

| Provider | Model |
|----------|-------|
| **OpenAI** | `gpt-5.2` |
| **OpenAI** | `gpt-5` |
| **OpenAI** | `gpt-5-mini` |
| **OpenAI** | `gpt-5-nano` |
| **OpenAI** | `gpt-4.1-mini` |
| **OpenAI** | `gpt-4.1-nano` |
| **OpenAI** | `gpt-4o-mini` |
| **Anthropic** | `claude-sonnet-4-20250514` |
| **Anthropic** | `claude-3-5-sonnet-20241022` |
| **Gemini** | `gemini-3-pro-preview` |
| **Gemini** | `gemini-2.5-flash` |
| **Gemini** | `gemini-2.5-flash-lite` |
| **Groq** | `openai/gpt-oss-120b` |
| **Groq** | `openai/gpt-oss-20b` |

### Provider Default Models

Each provider has a recommended default model that's used when `HINDSIGHT_API_LLM_MODEL` is not explicitly set. This makes configuration simpler - just specify the provider and get a sensible default:

| Provider | Default Model |
|----------|--------------|
| `openai` | `o3-mini` |
| `anthropic` | `claude-haiku-4-5-20251001` |
| `gemini` | `gemini-2.5-flash` |
| `groq` | `openai/gpt-oss-120b` |
| `ollama` | `gemma3:12b` |
| `lmstudio` | `local-model` |
| `vertexai` | `gemini-2.0-flash-001` |
| `openai-codex` | `gpt-5.2-codex` |
| `claude-code` | `claude-sonnet-4-5-20250929` |

**Example:** Setting just the provider uses its default model:
```bash
# Uses claude-haiku-4-5-20251001 automatically
export HINDSIGHT_API_LLM_PROVIDER=anthropic
export HINDSIGHT_API_LLM_API_KEY=sk-ant-xxxxxxxxxxxx
```

You can override the default by explicitly setting `HINDSIGHT_API_LLM_MODEL`:
```bash
# Override to use Sonnet instead
export HINDSIGHT_API_LLM_PROVIDER=anthropic
export HINDSIGHT_API_LLM_API_KEY=sk-ant-xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=claude-sonnet-4-5-20250929
```

This also applies to per-operation overrides:
```bash
# Global: OpenAI o3-mini (default)
export HINDSIGHT_API_LLM_PROVIDER=openai

# Retain: Anthropic claude-haiku-4-5-20251001 (default)
export HINDSIGHT_API_RETAIN_LLM_PROVIDER=anthropic
```

### Using Other Models

Other LLM models not listed above may work with Hindsight, but they must support **at least 65,000 output tokens** to ensure reliable fact extraction. If you need support for a specific model that doesn't meet this requirement, please [open an issue](https://github.com/hindsight-ai/hindsight/issues) to request an exception.

### Configuration

```bash
# Groq (recommended)
export HINDSIGHT_API_LLM_PROVIDER=groq
export HINDSIGHT_API_LLM_API_KEY=gsk_xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=openai/gpt-oss-20b

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

# Ollama (local)
export HINDSIGHT_API_LLM_PROVIDER=ollama
export HINDSIGHT_API_LLM_BASE_URL=http://localhost:11434/v1
export HINDSIGHT_API_LLM_MODEL=llama3

# LM Studio (local)
export HINDSIGHT_API_LLM_PROVIDER=lmstudio
export HINDSIGHT_API_LLM_BASE_URL=http://localhost:1234/v1
export HINDSIGHT_API_LLM_MODEL=your-local-model
```

**Note:** The LLM is the primary bottleneck for retain operations. See [Performance](./performance) for optimization strategies.

---

### OpenAI Codex Setup (ChatGPT Plus/Pro)

Use your ChatGPT Plus or Pro subscription for Hindsight without separate OpenAI Platform API costs.

**Prerequisites:**
- Active ChatGPT Plus or Pro subscription
- Node.js/npm installed (for Codex CLI)

**Setup Steps:**

1. **Install Codex CLI:**
   ```bash
   npm install -g @openai/codex
   ```

2. **Login with ChatGPT credentials:**
   ```bash
   codex auth login
   ```
   This opens a browser window to authenticate with your ChatGPT account and saves OAuth tokens to `~/.codex/auth.json`.

3. **Verify authentication:**
   ```bash
   ls ~/.codex/auth.json  # Should show the auth file exists
   ```

4. **Configure Hindsight:**
   ```bash
   export HINDSIGHT_API_LLM_PROVIDER=openai-codex
   # export HINDSIGHT_API_LLM_MODEL=gpt-5.1-codex  # defaults to gpt-5.2-codex
   # No API key needed - reads from ~/.codex/auth.json automatically
   ```

5. **Start Hindsight:**
   ```bash
   hindsight-api
   ```

You can use any model supported by OpenAI Codex CLI

**Important Notes:**
- OAuth tokens are stored in `~/.codex/auth.json`
- Tokens refresh automatically when needed
- Usage is billed to your ChatGPT subscription (not separate API costs)
- For personal development use only (see ChatGPT Terms of Service)

**Troubleshooting:**

If authentication fails:
```bash
# Re-login to refresh tokens
codex auth login
```

---

### Claude Code Setup (Claude Pro/Max)

Use your Claude Pro or Max subscription for Hindsight without separate Anthropic API costs.

**Prerequisites:**
- Active Claude Pro or Max subscription
- Claude Code CLI installed

**Setup Steps:**

1. **Install Claude Code CLI:**
   ```bash
   npm install -g @anthropics/claude-code
   # Or via Homebrew
   brew install anthropics/claude-code/claude-code
   ```

2. **Login with Claude credentials:**
   ```bash
   claude auth login
   ```
   This opens a browser window to authenticate with your Claude account. Authentication is automatically managed by the Claude Agent SDK.

3. **Verify authentication:**
   ```bash
   claude --version
   # Should show version without errors
   ```

4. **Configure Hindsight:**
   ```bash
   export HINDSIGHT_API_LLM_PROVIDER=claude-code
   # No API key needed - uses claude auth login credentials
   ```

5. **Start Hindsight:**
   ```bash
   hindsight-api
   ```

You can use any model supported by Claude Code CLI.

**Important Notes:**
- Authentication handled by Claude Agent SDK (uses bundled CLI)
- Credentials managed securely by Claude Code
- Usage billed to your Claude subscription (not separate API costs)
- For personal development use only (see Claude Terms of Service)

---

## Embedding Model

Converts text into dense vector representations for semantic similarity search.

**Default:** `BAAI/bge-small-en-v1.5` (384 dimensions, ~130MB)

### Supported Providers

| Provider | Description | Best For |
|----------|-------------|----------|
| `local` | SentenceTransformers (default) | Development, low latency |
| `openai` | OpenAI embeddings API | Production, high quality |
| `cohere` | Cohere embeddings API | Production, multilingual |
| `tei` | HuggingFace Text Embeddings Inference | Production, self-hosted |
| `litellm` | LiteLLM proxy (unified gateway) | Multi-provider setups |

### Local Models

| Model | Dimensions | Use Case |
|-------|------------|----------|
| `BAAI/bge-small-en-v1.5` | 384 | Default, fast, good quality |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual (50+ languages) |

### OpenAI Models

| Model | Dimensions | Use Case |
|-------|------------|----------|
| `text-embedding-3-small` | 1536 | Default OpenAI, cost-effective |
| `text-embedding-3-large` | 3072 | Higher quality, more expensive |
| `text-embedding-ada-002` | 1536 | Legacy model |

### Cohere Models

| Model | Dimensions | Use Case |
|-------|------------|----------|
| `embed-english-v3.0` | 1024 | English text |
| `embed-multilingual-v3.0` | 1024 | 100+ languages |

:::warning Embedding Dimensions
Hindsight automatically detects the embedding dimension at startup and adjusts the database schema. Once memories are stored, you cannot change dimensions without losing data.
:::

**Configuration Examples:**

```bash
# Local provider (default)
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
export HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-small-en-v1.5

# OpenAI
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
export HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=sk-xxxxxxxxxxxx
export HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=text-embedding-3-small

# Cohere
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=cohere
export HINDSIGHT_API_COHERE_API_KEY=your-api-key
export HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL=embed-english-v3.0

# TEI (self-hosted)
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=tei
export HINDSIGHT_API_EMBEDDINGS_TEI_URL=http://localhost:8080

# LiteLLM proxy
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=litellm
export HINDSIGHT_API_LITELLM_API_BASE=http://localhost:4000
export HINDSIGHT_API_EMBEDDINGS_LITELLM_MODEL=text-embedding-3-small
```

See [Configuration](./configuration#embeddings) for all options including Azure OpenAI and custom endpoints.

---

## Cross-Encoder (Reranker)

Reranks initial search results to improve precision.

**Default:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~85MB)

### Supported Providers

| Provider | Description | Best For |
|----------|-------------|----------|
| `local` | SentenceTransformers CrossEncoder (default) | Development, low latency |
| `cohere` | Cohere rerank API | Production, high quality |
| `tei` | HuggingFace Text Embeddings Inference | Production, self-hosted |
| `flashrank` | FlashRank (lightweight, fast) | Resource-constrained environments |
| `litellm` | LiteLLM proxy (unified gateway) | Multi-provider setups |
| `rrf` | RRF-only (no neural reranking) | Testing, minimal resources |

### Local Models

| Model | Use Case |
|-------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Default, fast |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Higher accuracy |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Multilingual |

### Cohere Models

| Model | Use Case |
|-------|----------|
| `rerank-english-v3.0` | English text |
| `rerank-multilingual-v3.0` | 100+ languages |

### LiteLLM Supported Providers

LiteLLM supports multiple reranking providers via the `/rerank` endpoint:

| Provider | Model Example |
|----------|---------------|
| Cohere | `cohere/rerank-english-v3.0` |
| Together AI | `together_ai/...` |
| Voyage AI | `voyage/rerank-2` |
| Jina AI | `jina_ai/...` |
| AWS Bedrock | `bedrock/...` |

**Configuration Examples:**

```bash
# Local provider (default)
export HINDSIGHT_API_RERANKER_PROVIDER=local
export HINDSIGHT_API_RERANKER_LOCAL_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Cohere
export HINDSIGHT_API_RERANKER_PROVIDER=cohere
export HINDSIGHT_API_COHERE_API_KEY=your-api-key
export HINDSIGHT_API_RERANKER_COHERE_MODEL=rerank-english-v3.0

# TEI (self-hosted)
export HINDSIGHT_API_RERANKER_PROVIDER=tei
export HINDSIGHT_API_RERANKER_TEI_URL=http://localhost:8081

# FlashRank (lightweight)
export HINDSIGHT_API_RERANKER_PROVIDER=flashrank

# LiteLLM proxy
export HINDSIGHT_API_RERANKER_PROVIDER=litellm
export HINDSIGHT_API_LITELLM_API_BASE=http://localhost:4000
export HINDSIGHT_API_RERANKER_LITELLM_MODEL=cohere/rerank-english-v3.0

# RRF-only (no neural reranking)
export HINDSIGHT_API_RERANKER_PROVIDER=rrf
```

See [Configuration](./configuration#reranker) for all options including Azure-hosted endpoints and batch settings.
