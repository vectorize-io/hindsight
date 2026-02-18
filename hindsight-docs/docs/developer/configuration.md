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
| `HINDSIGHT_API_DATABASE_SCHEMA` | PostgreSQL schema name for tables | `public` |
| `HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP` | Run database migrations on API startup | `true` |

If not provided, the server uses embedded `pg0` — convenient for development but not recommended for production.

The `DATABASE_SCHEMA` setting allows you to use a custom PostgreSQL schema instead of the default `public` schema. This is useful for:
- Multi-database setups where you want Hindsight tables in a dedicated schema
- Hosting platforms (e.g., Supabase) where `public` schema is reserved or shared
- Organizational preferences for schema naming conventions

```bash
# Example: Using a custom schema
export HINDSIGHT_API_DATABASE_URL=postgresql://user:pass@host:5432/dbname
export HINDSIGHT_API_DATABASE_SCHEMA=hindsight
```

Migrations will automatically create the schema if it doesn't exist and create all tables in the configured schema.

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

### Vector Extension

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_VECTOR_EXTENSION` | Vector index algorithm: `pgvector`, `vchord`, or `pgvectorscale` | `pgvector` |

Hindsight supports three PostgreSQL vector extensions:

#### **pgvector** (HNSW - default)
- In-memory index using Hierarchical Navigable Small World algorithm
- Works well for most embeddings and dataset sizes
- Fast for small-medium datasets (&lt;10M vectors)
- Higher memory usage for large datasets
- Most widely deployed and supported

#### **pgvectorscale** (DiskANN - recommended for scale) ⭐
- Disk-based index using StreamingDiskANN algorithm
- **28x lower p95 latency** and **16x higher throughput** vs dedicated vector DBs
- **60-75% cost reduction** at scale (SSDs cheaper than RAM)
- Superior filtering performance with streaming retrieval model
- Optimized for large datasets (10M+ vectors)
- Supports both **pgvectorscale** (open source) and **pg_diskann** (Azure)
- **Installation:**
  - Open source/self-hosted: `CREATE EXTENSION vector; CREATE EXTENSION vectorscale CASCADE;`
  - Azure PostgreSQL: `CREATE EXTENSION vector; CREATE EXTENSION pg_diskann CASCADE;`

#### **vchord** (vchordrq)
- Alternative high-performance vector index
- Optimized for high-dimensional embeddings (3000+ dimensions)
- Includes integrated BM25 search capabilities
- Requires `vchord` extension

**When to use pgvectorscale (DiskANN):**
- Large datasets (10M+ vectors) ⭐
- Complex filtering requirements
- Cost-sensitive deployments
- Production workloads requiring high throughput
- When disk I/O is not a bottleneck

**When to use pgvector (HNSW):**
- Small-medium datasets (&lt;10M vectors)
- Maximum query speed when all data fits in memory
- Simple nearest-neighbor queries without filters
- Standard PostgreSQL deployment preference

**When to use vchord:**
- High-dimensional embeddings (3000+ dimensions)
- Want integrated BM25 search
- Already using vchord for text search

**Switching extensions:**

If you need to switch from one extension to another:
1. Set `HINDSIGHT_API_VECTOR_EXTENSION` to your desired extension (`pgvector`, `vchord`, or `pgvectorscale`)
2. If your database has existing data, you'll get an error with migration instructions
3. For empty databases, indexes will be automatically recreated on startup

**Learn more:**
- [HNSW vs. DiskANN comparison](https://www.tigerdata.com/learn/hnsw-vs-diskann)
- [pgvectorscale GitHub](https://github.com/timescale/pgvectorscale)

### Text Search Extension

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_TEXT_SEARCH_EXTENSION` | Text search backend: `native`, `vchord`, or `pg_textsearch` | `native` |

Hindsight supports three text search backends for BM25 keyword retrieval:
- **native**: PostgreSQL's built-in full-text search (`tsvector` + GIN indexes)
- **vchord**: VectorChord BM25 (`bm25vector` + BM25 indexes) - requires `vchord_bm25` extension
- **pg_textsearch**: Timescale BM25 (text columns + BM25 indexes) - requires `pg_textsearch` extension

**When to use native:**
- Standard PostgreSQL deployment (no extra extensions)
- Simpler setup and wider compatibility
- Works well for most use cases

**When to use vchord:**
- Already using vchord for vector search (good integration)
- Want better BM25 ranking performance
- Need advanced tokenization (uses `llmlingua2` tokenizer)

**When to use pg_textsearch:**
- Want industry-standard BM25 ranking with better relevance than native PostgreSQL
- Need efficient top-K queries with Block-Max WAND optimization
- Prefer lower memory footprint compared to vchord
- Already using Timescale or have `pg_textsearch` available

**Switching backends:**

To switch between backends:
1. Set `HINDSIGHT_API_TEXT_SEARCH_EXTENSION` to your desired backend (`native`, `vchord`, or `pg_textsearch`)
2. If your database has existing data, you'll get an error with migration instructions
3. For empty databases, the columns/indexes will be automatically recreated on startup

**Note:** VectorChord uses the `llmlingua2` tokenizer for multilingual support, while native and pg_textsearch use PostgreSQL's English tokenizer.

### LLM Provider

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_LLM_PROVIDER` | Provider: `openai`, `openai-codex`, `claude-code`, `anthropic`, `gemini`, `groq`, `ollama`, `lmstudio`, `vertexai` | `openai` |
| `HINDSIGHT_API_LLM_API_KEY` | API key for LLM provider | - |
| `HINDSIGHT_API_LLM_MODEL` | Model name | `gpt-5-mini` |
| `HINDSIGHT_API_LLM_BASE_URL` | Custom LLM endpoint | Provider default |
| `HINDSIGHT_API_LLM_MAX_CONCURRENT` | Max concurrent LLM requests | `32` |
| `HINDSIGHT_API_LLM_MAX_RETRIES` | Max retry attempts for LLM API calls | `10` |
| `HINDSIGHT_API_LLM_INITIAL_BACKOFF` | Initial retry backoff in seconds (exponential backoff) | `1.0` |
| `HINDSIGHT_API_LLM_MAX_BACKOFF` | Max retry backoff cap in seconds | `60.0` |
| `HINDSIGHT_API_LLM_TIMEOUT` | LLM request timeout in seconds | `120` |
| `HINDSIGHT_API_LLM_GROQ_SERVICE_TIER` | Groq service tier: `on_demand`, `flex`, `auto` | `auto` |
| `HINDSIGHT_API_LLM_OPENAI_SERVICE_TIER` | OpenAI service tier: `flex` for 50% cost savings (OpenAI Flex Processing) | None (default) |

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
# Optional: Use Flex Processing for 50% cost savings (with variable latency)
# export HINDSIGHT_API_LLM_OPENAI_SERVICE_TIER=flex

# Gemini
export HINDSIGHT_API_LLM_PROVIDER=gemini
export HINDSIGHT_API_LLM_API_KEY=xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=gemini-2.0-flash

# Anthropic
export HINDSIGHT_API_LLM_PROVIDER=anthropic
export HINDSIGHT_API_LLM_API_KEY=sk-ant-xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=claude-sonnet-4-20250514

# Vertex AI (Google Cloud - uses native genai SDK)
export HINDSIGHT_API_LLM_PROVIDER=vertexai
export HINDSIGHT_API_LLM_MODEL=gemini-2.0-flash-001
export HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID=your-gcp-project-id
export HINDSIGHT_API_LLM_VERTEXAI_REGION=us-central1
# Optional: use ADC (gcloud auth application-default login) or provide service account key:
# export HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY=/path/to/service-account-key.json

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

# OpenAI Codex (ChatGPT Plus/Pro subscription - uses OAuth, no API key needed)
export HINDSIGHT_API_LLM_PROVIDER=openai-codex
export HINDSIGHT_API_LLM_MODEL=gpt-5.2-codex
# No API key needed - uses OAuth tokens from ~/.codex/auth.json

# Claude Code (Claude Pro/Max subscription - uses OAuth, no API key needed)
export HINDSIGHT_API_LLM_PROVIDER=claude-code
export HINDSIGHT_API_LLM_MODEL=claude-sonnet-4-5-20250929
# No API key needed - uses claude auth login credentials
```

:::tip OpenAI Codex & Claude Code Setup
For detailed setup instructions for **OpenAI Codex** (ChatGPT Plus/Pro) and **Claude Code** (Claude Pro/Max), see the [Models documentation](./models#openai-codex-setup-chatgpt-pluspro).
:::

#### Vertex AI Setup

Google Cloud's Vertex AI provides access to Gemini models via the native Google GenAI SDK. Hindsight supports two authentication methods:

**Prerequisites:**
- GCP project with Vertex AI API enabled
- IAM role `roles/aiplatform.user` for your credentials

**Environment Variables:**

| Variable | Description | Required |
|----------|-------------|----------|
| `HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID` | Your GCP project ID | Yes |
| `HINDSIGHT_API_LLM_VERTEXAI_REGION` | GCP region (e.g., `us-central1`) | No (default: `us-central1`) |
| `HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY` | Path to service account JSON key file | No (uses ADC if not set) |

**Authentication Methods:**

1. **Application Default Credentials (ADC)** - Recommended for development
   ```bash
   # Setup ADC
   gcloud auth application-default login

   # Configure Hindsight
   export HINDSIGHT_API_LLM_PROVIDER=vertexai
   export HINDSIGHT_API_LLM_MODEL=gemini-2.0-flash-001
   export HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID=your-project-id
   ```

2. **Service Account Key** - Recommended for production
   ```bash
   # Create service account and download key
   gcloud iam service-accounts create hindsight-api
   gcloud projects add-iam-policy-binding your-project-id \
     --member="serviceAccount:hindsight-api@your-project-id.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   gcloud iam service-accounts keys create key.json \
     --iam-account=hindsight-api@your-project-id.iam.gserviceaccount.com

   # Configure Hindsight
   export HINDSIGHT_API_LLM_PROVIDER=vertexai
   export HINDSIGHT_API_LLM_MODEL=gemini-2.0-flash-001
   export HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID=your-project-id
   export HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY=/path/to/key.json
   ```

**Notes:**
- Model names can optionally include the `google/` prefix (e.g., `google/gemini-2.0-flash-001`) - it will be stripped automatically
- The native SDK handles token refresh automatically
- Uses service account credentials if provided, otherwise falls back to ADC

### Per-Operation LLM Configuration

Different memory operations have different requirements. **Retain** (fact extraction) benefits from models with strong structured output capabilities, while **Reflect** (reasoning/response generation) can use lighter, faster models. Configure separate LLM models for each operation to optimize for cost and performance.

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_RETAIN_LLM_PROVIDER` | LLM provider for retain operations | Falls back to `HINDSIGHT_API_LLM_PROVIDER` |
| `HINDSIGHT_API_RETAIN_LLM_API_KEY` | API key for retain LLM | Falls back to `HINDSIGHT_API_LLM_API_KEY` |
| `HINDSIGHT_API_RETAIN_LLM_MODEL` | Model for retain operations | Falls back to `HINDSIGHT_API_LLM_MODEL` |
| `HINDSIGHT_API_RETAIN_LLM_BASE_URL` | Base URL for retain LLM | Falls back to `HINDSIGHT_API_LLM_BASE_URL` |
| `HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT` | Max concurrent requests for retain | Falls back to `HINDSIGHT_API_LLM_MAX_CONCURRENT` |
| `HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES` | Max retries for retain | Falls back to `HINDSIGHT_API_LLM_MAX_RETRIES` |
| `HINDSIGHT_API_RETAIN_LLM_INITIAL_BACKOFF` | Initial backoff for retain retries (seconds) | Falls back to `HINDSIGHT_API_LLM_INITIAL_BACKOFF` |
| `HINDSIGHT_API_RETAIN_LLM_MAX_BACKOFF` | Max backoff cap for retain retries (seconds) | Falls back to `HINDSIGHT_API_LLM_MAX_BACKOFF` |
| `HINDSIGHT_API_RETAIN_LLM_TIMEOUT` | Timeout for retain requests (seconds) | Falls back to `HINDSIGHT_API_LLM_TIMEOUT` |
| `HINDSIGHT_API_REFLECT_LLM_PROVIDER` | LLM provider for reflect operations | Falls back to `HINDSIGHT_API_LLM_PROVIDER` |
| `HINDSIGHT_API_REFLECT_LLM_API_KEY` | API key for reflect LLM | Falls back to `HINDSIGHT_API_LLM_API_KEY` |
| `HINDSIGHT_API_REFLECT_LLM_MODEL` | Model for reflect operations | Falls back to `HINDSIGHT_API_LLM_MODEL` |
| `HINDSIGHT_API_REFLECT_LLM_BASE_URL` | Base URL for reflect LLM | Falls back to `HINDSIGHT_API_LLM_BASE_URL` |
| `HINDSIGHT_API_REFLECT_LLM_MAX_CONCURRENT` | Max concurrent requests for reflect | Falls back to `HINDSIGHT_API_LLM_MAX_CONCURRENT` |
| `HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES` | Max retries for reflect | Falls back to `HINDSIGHT_API_LLM_MAX_RETRIES` |
| `HINDSIGHT_API_REFLECT_LLM_INITIAL_BACKOFF` | Initial backoff for reflect retries (seconds) | Falls back to `HINDSIGHT_API_LLM_INITIAL_BACKOFF` |
| `HINDSIGHT_API_REFLECT_LLM_MAX_BACKOFF` | Max backoff cap for reflect retries (seconds) | Falls back to `HINDSIGHT_API_LLM_MAX_BACKOFF` |
| `HINDSIGHT_API_REFLECT_LLM_TIMEOUT` | Timeout for reflect requests (seconds) | Falls back to `HINDSIGHT_API_LLM_TIMEOUT` |
| `HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER` | LLM provider for observation consolidation | Falls back to `HINDSIGHT_API_LLM_PROVIDER` |
| `HINDSIGHT_API_CONSOLIDATION_LLM_API_KEY` | API key for consolidation LLM | Falls back to `HINDSIGHT_API_LLM_API_KEY` |
| `HINDSIGHT_API_CONSOLIDATION_LLM_MODEL` | Model for consolidation operations | Falls back to `HINDSIGHT_API_LLM_MODEL` |
| `HINDSIGHT_API_CONSOLIDATION_LLM_BASE_URL` | Base URL for consolidation LLM | Falls back to `HINDSIGHT_API_LLM_BASE_URL` |
| `HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT` | Max concurrent requests for consolidation | Falls back to `HINDSIGHT_API_LLM_MAX_CONCURRENT` |
| `HINDSIGHT_API_CONSOLIDATION_LLM_MAX_RETRIES` | Max retries for consolidation | Falls back to `HINDSIGHT_API_LLM_MAX_RETRIES` |
| `HINDSIGHT_API_CONSOLIDATION_LLM_INITIAL_BACKOFF` | Initial backoff for consolidation retries (seconds) | Falls back to `HINDSIGHT_API_LLM_INITIAL_BACKOFF` |
| `HINDSIGHT_API_CONSOLIDATION_LLM_MAX_BACKOFF` | Max backoff cap for consolidation retries (seconds) | Falls back to `HINDSIGHT_API_LLM_MAX_BACKOFF` |
| `HINDSIGHT_API_CONSOLIDATION_LLM_TIMEOUT` | Timeout for consolidation requests (seconds) | Falls back to `HINDSIGHT_API_LLM_TIMEOUT` |

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

**Example: Tuning Retry Behavior for Rate-Limited APIs**

```bash
# For Anthropic with tight rate limits (10k output tokens/minute)
export HINDSIGHT_API_LLM_PROVIDER=anthropic
export HINDSIGHT_API_LLM_API_KEY=sk-ant-xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=claude-sonnet-4-20250514

# Reduce concurrent requests for retain to avoid rate limits
export HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT=3

# Fail faster with fewer retries
export HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES=3

# Or increase backoff times to wait out rate limit windows
export HINDSIGHT_API_RETAIN_LLM_INITIAL_BACKOFF=2.0  # Start at 2s instead of 1s
export HINDSIGHT_API_RETAIN_LLM_MAX_BACKOFF=120.0    # Cap at 2min instead of 1min
```

### Embeddings

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_EMBEDDINGS_PROVIDER` | Provider: `local`, `tei`, `openai`, `cohere`, `litellm`, or `litellm-sdk` | `local` |
| `HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL` | Model for local provider | `BAAI/bge-small-en-v1.5` |
| `HINDSIGHT_API_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE` | Allow loading models with custom code (security risk, disabled by default) | `false` |
| `HINDSIGHT_API_EMBEDDINGS_TEI_URL` | TEI server URL | - |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY` | OpenAI API key (falls back to `HINDSIGHT_API_LLM_API_KEY`) | - |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL` | Custom base URL for OpenAI-compatible API (e.g., Azure OpenAI) | - |
| `HINDSIGHT_API_EMBEDDINGS_COHERE_API_KEY` | Cohere API key for embeddings | - |
| `HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL` | Cohere embedding model | `embed-english-v3.0` |
| `HINDSIGHT_API_EMBEDDINGS_COHERE_BASE_URL` | Custom base URL for Cohere-compatible API (e.g., Azure-hosted) | - |
| `HINDSIGHT_API_EMBEDDINGS_LITELLM_API_BASE` | LiteLLM proxy base URL for embeddings | `http://localhost:4000` |
| `HINDSIGHT_API_EMBEDDINGS_LITELLM_API_KEY` | LiteLLM proxy API key for embeddings (optional, depends on proxy config) | - |
| `HINDSIGHT_API_EMBEDDINGS_LITELLM_MODEL` | LiteLLM embedding model (use provider prefix, e.g., `cohere/embed-english-v3.0`) | `text-embedding-3-small` |
| `HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_KEY` | LiteLLM SDK API key for direct embedding provider access | - |
| `HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_MODEL` | LiteLLM SDK embedding model (use provider prefix, e.g., `cohere/embed-english-v3.0`) | `cohere/embed-english-v3.0` |
| `HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_BASE` | Custom base URL for LiteLLM SDK embeddings (optional) | - |

```bash
# Local (default) - uses SentenceTransformers
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
export HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-small-en-v1.5

# Local with custom model requiring trust_remote_code
# WARNING: Only enable trust_remote_code for models you trust (security risk)
# export HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=your-custom-model
# export HINDSIGHT_API_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE=true

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
export HINDSIGHT_API_EMBEDDINGS_COHERE_API_KEY=your-api-key
export HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL=embed-english-v3.0  # 1024 dimensions

# Azure-hosted Cohere - embeddings via custom endpoint
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=cohere
export HINDSIGHT_API_EMBEDDINGS_COHERE_API_KEY=your-azure-api-key
export HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL=embed-english-v3.0
export HINDSIGHT_API_EMBEDDINGS_COHERE_BASE_URL=https://your-azure-cohere-endpoint.com

# LiteLLM proxy - unified gateway for multiple providers
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=litellm
export HINDSIGHT_API_EMBEDDINGS_LITELLM_API_BASE=http://localhost:4000
export HINDSIGHT_API_EMBEDDINGS_LITELLM_API_KEY=your-litellm-key  # optional
export HINDSIGHT_API_EMBEDDINGS_LITELLM_MODEL=text-embedding-3-small  # or cohere/embed-english-v3.0

# LiteLLM SDK - direct API access without proxy server (recommended)
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=litellm-sdk
export HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_KEY=your-provider-api-key
export HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_MODEL=cohere/embed-english-v3.0

# Supported LiteLLM SDK embedding providers:
# - cohere/embed-english-v3.0 (1024 dimensions)
# - openai/text-embedding-3-small (1536 dimensions)
# - together_ai/togethercomputer/m2-bert-80M-8k-retrieval
# - huggingface/sentence-transformers/all-MiniLM-L6-v2
# - voyage/voyage-2
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
| `HINDSIGHT_API_RERANKER_PROVIDER` | Provider: `local`, `tei`, `cohere`, `flashrank`, `litellm`, `litellm-sdk`, or `rrf` | `local` |
| `HINDSIGHT_API_RERANKER_LOCAL_MODEL` | Model for local provider | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `HINDSIGHT_API_RERANKER_LOCAL_MAX_CONCURRENT` | Max concurrent local reranking (prevents CPU thrashing under load) | `4` |
| `HINDSIGHT_API_RERANKER_LOCAL_TRUST_REMOTE_CODE` | Allow loading models with custom code (security risk, disabled by default) | `false` |
| `HINDSIGHT_API_RERANKER_TEI_URL` | TEI server URL | - |
| `HINDSIGHT_API_RERANKER_TEI_BATCH_SIZE` | Batch size for TEI reranking | `128` |
| `HINDSIGHT_API_RERANKER_TEI_MAX_CONCURRENT` | Max concurrent TEI reranking requests | `8` |
| `HINDSIGHT_API_RERANKER_COHERE_API_KEY` | Cohere API key for reranking | - |
| `HINDSIGHT_API_RERANKER_COHERE_MODEL` | Cohere rerank model | `rerank-english-v3.0` |
| `HINDSIGHT_API_RERANKER_COHERE_BASE_URL` | Custom base URL for Cohere-compatible API (e.g., Azure-hosted) | - |
| `HINDSIGHT_API_RERANKER_LITELLM_API_BASE` | LiteLLM proxy base URL for reranking | `http://localhost:4000` |
| `HINDSIGHT_API_RERANKER_LITELLM_API_KEY` | LiteLLM proxy API key for reranking (optional, depends on proxy config) | - |
| `HINDSIGHT_API_RERANKER_LITELLM_MODEL` | LiteLLM **proxy** rerank model (use provider prefix, e.g., `cohere/rerank-english-v3.0`) | `cohere/rerank-english-v3.0` |
| `HINDSIGHT_API_RERANKER_LITELLM_SDK_API_KEY` | LiteLLM **SDK** API key for direct reranking (no proxy needed) | - |
| `HINDSIGHT_API_RERANKER_LITELLM_SDK_MODEL` | LiteLLM SDK rerank model (e.g., `deepinfra/Qwen3-reranker-8B`) | `cohere/rerank-english-v3.0` |
| `HINDSIGHT_API_RERANKER_LITELLM_SDK_API_BASE` | Custom API base URL for LiteLLM SDK (optional) | - |
| `HINDSIGHT_API_RERANKER_FLASHRANK_MODEL` | FlashRank model for fast CPU-based reranking | `ms-marco-MiniLM-L-12-v2` |
| `HINDSIGHT_API_RERANKER_FLASHRANK_CACHE_DIR` | Cache directory for FlashRank models | System default |

```bash
# Local (default) - uses SentenceTransformers CrossEncoder
export HINDSIGHT_API_RERANKER_PROVIDER=local
export HINDSIGHT_API_RERANKER_LOCAL_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Local with custom model requiring trust_remote_code (e.g., jina-reranker-v2)
# WARNING: Only enable trust_remote_code for models you trust (security risk)
export HINDSIGHT_API_RERANKER_PROVIDER=local
export HINDSIGHT_API_RERANKER_LOCAL_MODEL=jinaai/jina-reranker-v2-base-multilingual
export HINDSIGHT_API_RERANKER_LOCAL_TRUST_REMOTE_CODE=true

# TEI - for high-performance inference
export HINDSIGHT_API_RERANKER_PROVIDER=tei
export HINDSIGHT_API_RERANKER_TEI_URL=http://localhost:8081

# Cohere - cloud-based reranking
export HINDSIGHT_API_RERANKER_PROVIDER=cohere
export HINDSIGHT_API_RERANKER_COHERE_API_KEY=your-api-key
export HINDSIGHT_API_RERANKER_COHERE_MODEL=rerank-english-v3.0

# Azure-hosted Cohere - reranking via custom endpoint
export HINDSIGHT_API_RERANKER_PROVIDER=cohere
export HINDSIGHT_API_RERANKER_COHERE_API_KEY=your-azure-api-key
export HINDSIGHT_API_RERANKER_COHERE_MODEL=rerank-english-v3.0
export HINDSIGHT_API_RERANKER_COHERE_BASE_URL=https://your-azure-cohere-endpoint.com

# LiteLLM proxy - unified gateway for multiple reranking providers (requires running LiteLLM proxy server)
export HINDSIGHT_API_RERANKER_PROVIDER=litellm
export HINDSIGHT_API_RERANKER_LITELLM_API_BASE=http://localhost:4000
export HINDSIGHT_API_RERANKER_LITELLM_API_KEY=your-litellm-key  # optional
export HINDSIGHT_API_RERANKER_LITELLM_MODEL=cohere/rerank-english-v3.0  # or voyage/rerank-2, together_ai/...

# LiteLLM SDK - direct API access without proxy (recommended for simplicity)
export HINDSIGHT_API_RERANKER_PROVIDER=litellm-sdk
export HINDSIGHT_API_RERANKER_LITELLM_SDK_API_KEY=your-deepinfra-api-key
export HINDSIGHT_API_RERANKER_LITELLM_SDK_MODEL=deepinfra/Qwen3-reranker-8B  # or cohere/rerank-english-v3.0, etc.
```

#### LiteLLM Proxy vs SDK

- **`litellm`**: Requires running a separate LiteLLM proxy server. Good for centralized configuration, rate limiting, and caching.
- **`litellm-sdk`**: Direct API access without proxy. Simpler setup, lower latency, fewer infrastructure components.

Both support the same providers:
- **Cohere** (`cohere/rerank-english-v3.0`, `cohere/rerank-multilingual-v3.0`)
- **DeepInfra** (`deepinfra/Qwen3-reranker-8B`, `deepinfra/bge-reranker-v2-m3`)
- **Together AI** (`together_ai/Salesforce/Llama-Rank-V1`)
- **HuggingFace** (`huggingface/BAAI/bge-reranker-v2-m3`)
- **Voyage AI** (`voyage/rerank-2`)
- **Jina AI** (`jina_ai/jina-reranker-v2`)
- **AWS Bedrock** (`bedrock/...`)

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
| `HINDSIGHT_API_BASE_PATH` | Base path for API when behind reverse proxy (e.g., `/hindsight`) | `""` (root) |
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
| `HINDSIGHT_API_RETAIN_BATCH_ENABLED` | Use LLM Batch API for fact extraction (50% cost savings, only with async operations) | `false` |
| `HINDSIGHT_API_RETAIN_BATCH_POLL_INTERVAL_SECONDS` | Batch API polling interval in seconds | `60` |

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

### File Processing

Configuration for the file upload and conversion pipeline (used by `POST /v1/default/banks/{bank_id}/files/retain`).

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_ENABLE_FILE_UPLOAD_API` | Enable the file upload API endpoint | `true` |
| `HINDSIGHT_API_FILE_PARSER` | File parser to use (`markitdown`) | `markitdown` |
| `HINDSIGHT_API_FILE_CONVERSION_MAX_BATCH_SIZE` | Max files per upload request | `10` |
| `HINDSIGHT_API_FILE_CONVERSION_MAX_BATCH_SIZE_MB` | Max total upload size per request (MB) | `100` |
| `HINDSIGHT_API_FILE_DELETE_AFTER_RETAIN` | Delete stored files after memory extraction completes | `true` |

**Supported formats (via markitdown):** PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, images (JPG, PNG, GIF — OCR), audio (MP3, WAV — transcription), HTML, TXT, MD, CSV, and more.

```bash
# Increase batch limits for large file imports
export HINDSIGHT_API_FILE_CONVERSION_MAX_BATCH_SIZE=20
export HINDSIGHT_API_FILE_CONVERSION_MAX_BATCH_SIZE_MB=500

# Keep files after processing (useful for debugging or re-processing)
export HINDSIGHT_API_FILE_DELETE_AFTER_RETAIN=false
```

### File Storage

Files uploaded via the file retain API are stored in an object storage backend before conversion. Choose the backend that fits your infrastructure.

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_FILE_STORAGE_TYPE` | Storage backend: `native`, `s3`, `gcs`, or `azure` | `native` |

#### Native (PostgreSQL)

Files are stored as `BYTEA` in the `file_storage` table. No additional infrastructure required. Suitable for development and small deployments.

```bash
# Native storage is the default — no additional configuration needed
export HINDSIGHT_API_FILE_STORAGE_TYPE=native
```

#### S3 / S3-Compatible

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_FILE_STORAGE_S3_BUCKET` | S3 bucket name | - |
| `HINDSIGHT_API_FILE_STORAGE_S3_REGION` | AWS region | - |
| `HINDSIGHT_API_FILE_STORAGE_S3_ENDPOINT` | Custom endpoint URL (for S3-compatible stores like MinIO, Cloudflare R2) | AWS default |
| `HINDSIGHT_API_FILE_STORAGE_S3_ACCESS_KEY_ID` | AWS access key ID | - |
| `HINDSIGHT_API_FILE_STORAGE_S3_SECRET_ACCESS_KEY` | AWS secret access key | - |

```bash
# AWS S3
export HINDSIGHT_API_FILE_STORAGE_TYPE=s3
export HINDSIGHT_API_FILE_STORAGE_S3_BUCKET=my-hindsight-files
export HINDSIGHT_API_FILE_STORAGE_S3_REGION=us-east-1
export HINDSIGHT_API_FILE_STORAGE_S3_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export HINDSIGHT_API_FILE_STORAGE_S3_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# S3-compatible (MinIO, Cloudflare R2, etc.)
export HINDSIGHT_API_FILE_STORAGE_TYPE=s3
export HINDSIGHT_API_FILE_STORAGE_S3_BUCKET=my-bucket
export HINDSIGHT_API_FILE_STORAGE_S3_ENDPOINT=https://your-minio.example.com
export HINDSIGHT_API_FILE_STORAGE_S3_ACCESS_KEY_ID=minioadmin
export HINDSIGHT_API_FILE_STORAGE_S3_SECRET_ACCESS_KEY=minioadmin
```

#### Google Cloud Storage

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_FILE_STORAGE_GCS_BUCKET` | GCS bucket name | - |
| `HINDSIGHT_API_FILE_STORAGE_GCS_SERVICE_ACCOUNT_KEY` | Path to service account JSON key file | ADC if not set |

```bash
export HINDSIGHT_API_FILE_STORAGE_TYPE=gcs
export HINDSIGHT_API_FILE_STORAGE_GCS_BUCKET=my-hindsight-files
# Optional: use service account key file (otherwise falls back to ADC)
export HINDSIGHT_API_FILE_STORAGE_GCS_SERVICE_ACCOUNT_KEY=/path/to/key.json
```

#### Azure Blob Storage

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_FILE_STORAGE_AZURE_CONTAINER` | Azure container name | - |
| `HINDSIGHT_API_FILE_STORAGE_AZURE_ACCOUNT_NAME` | Azure storage account name | - |
| `HINDSIGHT_API_FILE_STORAGE_AZURE_ACCOUNT_KEY` | Azure storage account key | - |

```bash
export HINDSIGHT_API_FILE_STORAGE_TYPE=azure
export HINDSIGHT_API_FILE_STORAGE_AZURE_CONTAINER=hindsight-files
export HINDSIGHT_API_FILE_STORAGE_AZURE_ACCOUNT_NAME=mystorageaccount
export HINDSIGHT_API_FILE_STORAGE_AZURE_ACCOUNT_KEY=base64encodedkey==
```

#### Storage Backend Comparison

| Backend | Best For | Notes |
|---------|----------|-------|
| `native` | Development, small deployments | No extra infrastructure, stored in PostgreSQL |
| `s3` | Production, AWS deployments | Works with any S3-compatible store |
| `gcs` | Production, GCP deployments | Supports ADC for keyless auth |
| `azure` | Production, Azure deployments | Uses account key auth |

:::tip Production Recommendation
For production deployments, use `s3`, `gcs`, or `azure` to avoid storing large binary files in your PostgreSQL database. Set `HINDSIGHT_API_FILE_DELETE_AFTER_RETAIN=true` (the default) to delete files after memory extraction, which minimizes storage costs.
:::

### Observations (Experimental)

Observations are consolidated knowledge synthesized from facts.

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_ENABLE_OBSERVATIONS` | Enable observation consolidation | `true` |
| `HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE` | Memories to load per batch (internal optimization) | `50` |
| `HINDSIGHT_API_CONSOLIDATION_MAX_TOKENS` | Max tokens for recall when finding related observations during consolidation | `1024` |

### Reflect

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_REFLECT_MAX_ITERATIONS` | Max tool call iterations before forcing a response | `10` |

### MCP Server

Configuration for MCP server endpoints.

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_MCP_ENABLED` | Enable MCP server at `/mcp/{bank_id}/` | `true` |
| `HINDSIGHT_API_MCP_AUTH_TOKEN` | Bearer token for MCP authentication (optional) | - |
| `HINDSIGHT_API_MCP_LOCAL_BANK_ID` | Memory bank ID for local MCP | `mcp` |
| `HINDSIGHT_API_MCP_INSTRUCTIONS` | Additional instructions appended to retain/recall tool descriptions | - |

**MCP Authentication:**

By default, the MCP endpoint is open. For production deployments, set `HINDSIGHT_API_MCP_AUTH_TOKEN` to require Bearer token authentication:

```bash
export HINDSIGHT_API_MCP_AUTH_TOKEN=your-secret-token
```

Clients must then include the token in the `Authorization` header. See [MCP Server documentation](./mcp-server.md#authentication) for details.

**Local MCP instructions:**

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
| `HINDSIGHT_API_WORKER_MAX_RETRIES` | Max retries before marking task failed | `3` |
| `HINDSIGHT_API_WORKER_HTTP_PORT` | HTTP port for worker metrics/health (worker CLI only) | `8889` |
| `HINDSIGHT_API_WORKER_MAX_SLOTS` | Maximum concurrent tasks per worker | `10` |
| `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS` | Maximum concurrent consolidation tasks per worker | `2` |

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

## Observability & Tracing

Hindsight provides OpenTelemetry-based observability for LLM calls, conforming to GenAI semantic conventions.

### OpenTelemetry Tracing

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_OTEL_TRACES_ENABLED` | Enable distributed tracing for LLM calls | `false` |
| `HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint URL (e.g., Grafana LGTM, Langfuse, etc.) | - |
| `HINDSIGHT_API_OTEL_EXPORTER_OTLP_HEADERS` | Headers for OTLP exporter (format: "key1=value1,key2=value2") | - |
| `HINDSIGHT_API_OTEL_SERVICE_NAME` | Service name for traces | `hindsight-api` |
| `HINDSIGHT_API_OTEL_DEPLOYMENT_ENVIRONMENT` | Deployment environment name (e.g., development, staging, production) | `development` |

**Features:**
- Full prompts and completions recorded as events
- Token usage tracking (input/output)
- Model and provider information
- Error tracking with finish reasons
- Conforms to OpenTelemetry GenAI semantic conventions v1.37+

**OTLP-Compatible Backends:**

The tracing implementation uses standard OTLP HTTP protocol, so it works with any OTLP-compatible backend:
- **Grafana LGTM** (Recommended for local dev): All-in-one stack with Tempo traces, Loki logs, Mimir metrics, and Grafana UI
- **Langfuse**: LLM-focused observability and analytics
- **OpenLIT**: Built-in LLM dashboards, cost tracking
- **DataDog, New Relic, Honeycomb**: Commercial platforms

**Example Configuration:**

```bash
# Enable tracing
export HINDSIGHT_API_OTEL_TRACES_ENABLED=true

# Configure endpoint (example: OpenLIT Cloud)
export HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.openlit.io
export HINDSIGHT_API_OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer olit-xxx"

# Optional: Custom service name and environment
export HINDSIGHT_API_OTEL_SERVICE_NAME=hindsight-production
export HINDSIGHT_API_OTEL_DEPLOYMENT_ENVIRONMENT=production
```

**Local Development:**

For local development, we recommend the Grafana LGTM stack which provides traces, metrics, and logs in a single container:

```bash
./scripts/dev/start-grafana.sh
```

See `scripts/dev/grafana/README.md` for detailed setup instructions.

Other options: See `scripts/dev/openlit/README.md` for OpenLIT or `scripts/dev/jaeger/README.md` for standalone Jaeger.

### Metrics

Hindsight exposes Prometheus metrics at the `/metrics` endpoint, including:
- LLM call duration and token usage
- Operation duration (retain/recall/reflect)
- HTTP request metrics
- Database connection pool metrics

Metrics are always enabled and available at `http://localhost:8888/metrics`.

---

## Control Plane

The Control Plane is the web UI for managing memory banks.

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_CP_DATAPLANE_API_URL` | URL of the API service | `http://localhost:8888` |
| `NEXT_PUBLIC_BASE_PATH` | Base path for Control Plane UI when behind reverse proxy (e.g., `/hindsight`) | `""` (root) |

```bash
# Point Control Plane to a remote API service
export HINDSIGHT_CP_DATAPLANE_API_URL=http://api.example.com:8888
```

### Hierarchical Configuration

Hindsight supports per-bank configuration overrides through a hierarchical system: **Global (env vars) → Tenant → Bank**.

#### Type-Safe Config Access

To prevent accidentally using global defaults when bank-specific overrides exist, Hindsight enforces type-safe config access:

**In Application Code:**
```python
from hindsight_api.config import get_config

# ✅ Access static (infrastructure) fields
config = get_config()
host = config.host  # OK - static field
port = config.port  # OK - static field

# ❌ Attempting to access bank-configurable fields raises an error
chunk_size = config.retain_chunk_size  # ConfigFieldAccessError!
```

**Error Message:**
```
ConfigFieldAccessError: Field 'retain_chunk_size' is bank-configurable and cannot
be accessed from global config. Use ConfigResolver.resolve_full_config(bank_id, context)
to get bank-specific config.
```

**For Bank-Specific Config:**
```python
# Internal code that needs bank-specific settings
from hindsight_api.config_resolver import ConfigResolver

# Resolve full config for a specific bank
config = await config_resolver.resolve_full_config(bank_id, request_context)
chunk_size = config.retain_chunk_size  # ✅ Uses bank-specific value
```

This design prevents bugs where global defaults are used instead of bank overrides, making it impossible to make this mistake at compile/development time.

#### Security Model

Configuration fields are categorized for security:

1. **Configurable Fields** - Safe behavioral settings that can be customized per-bank:
   - Retention: `retain_chunk_size`, `retain_extraction_mode`, `retain_custom_instructions`
   - Consolidation: `enable_observations`

2. **Credential Fields** - NEVER exposed or configurable via API:
   - API keys: `*_api_key` (all LLM API keys)
   - Infrastructure: `*_base_url` (all base URLs)

3. **Static Fields** - Server-level only, cannot be overridden:
   - Infrastructure: `database_url`, `port`, `host`, `worker_count`
   - Provider/Model selection: `llm_provider`, `llm_model` (requires presets - not yet implemented)
   - Performance tuning: `llm_max_concurrent`, `llm_timeout`, retrieval settings, optimization flags

#### Enabling the API

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_ENABLE_BANK_CONFIG_API` | Enable per-bank config API | `false` |

**Important:** The bank config API is **disabled by default** for security. Enable it explicitly:

```bash
export HINDSIGHT_API_ENABLE_BANK_CONFIG_API=true
```

#### API Endpoints

- `GET /v1/default/banks/{bank_id}/config` - View resolved config (filtered by permissions)
- `PATCH /v1/default/banks/{bank_id}/config` - Update bank overrides (only allowed fields)
- `DELETE /v1/default/banks/{bank_id}/config` - Reset to defaults

#### Permission System

Tenant extensions can control which fields banks are allowed to modify via `get_allowed_config_fields()`:

```python
class CustomTenantExtension(TenantExtension):
    async def get_allowed_config_fields(self, context, bank_id):
        # Option 1: Allow all configurable fields
        return None

        # Option 2: Allow specific fields only
        return {"retain_chunk_size", "retain_custom_instructions"}

        # Option 3: Read-only (no modifications)
        return set()
```

#### Examples

```bash
# Update retention settings for a bank
curl -X PATCH http://localhost:8888/v1/default/banks/my-bank/config \
  -H "Content-Type: application/json" \
  -d '{
    "updates": {
      "retain_chunk_size": 4000,
      "retain_extraction_mode": "custom",
      "retain_custom_instructions": "Focus on technical details and implementation specifics"
    }
  }'

# Note: retain_extraction_mode must be "custom" to use retain_custom_instructions

# View resolved config (respects permissions)
curl http://localhost:8888/v1/default/banks/my-bank/config

# Reset to defaults
curl -X DELETE http://localhost:8888/v1/default/banks/my-bank/config
```

**Security Notes:**
- Credentials (API keys, base URLs) are never returned in responses
- Only configurable fields can be modified
- Responses are filtered by tenant permissions
- Attempting to set credentials returns 400 error

### Reverse Proxy / Subpath Deployment

To deploy Hindsight under a subpath (e.g., `example.com/hindsight/`):

1. Set both environment variables to the same path:
   ```bash
   HINDSIGHT_API_BASE_PATH=/hindsight
   NEXT_PUBLIC_BASE_PATH=/hindsight
   ```

2. Configure your reverse proxy to:
   - Forward `/hindsight/*` requests to Hindsight
   - Preserve the full path in forwarded requests
   - Set appropriate proxy headers (X-Forwarded-Proto, X-Forwarded-For)

**Example: Nginx Configuration**

```nginx
location /hindsight/ {
    proxy_pass http://localhost:8888/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

**Example: Traefik Configuration**

```yaml
http:
  routers:
    hindsight:
      rule: "PathPrefix(`/hindsight`)"
      service: hindsight
      middlewares:
        - hindsight-stripprefix

  middlewares:
    hindsight-stripprefix:
      stripPrefix:
        prefixes:
          - "/hindsight"

  services:
    hindsight:
      loadBalancer:
        servers:
          - url: "http://localhost:8888"
```

**Important Notes:**
- The base path must start with `/` and should NOT end with `/`
- Both API and Control Plane should use the same base path
- After setting environment variables, restart both services
- OpenAPI docs will be available at `<base-path>/docs` (e.g., `/hindsight/docs`)

**Complete Examples:**

See `docker/compose-examples/` directory for:
- Nginx configuration files (`simple.conf`, `api-and-control-plane.conf`)
- Docker Compose setups (`docker-compose.yml`, `reverse-proxy-only.yml`)
- Traefik and other reverse proxy examples
- Full deployment documentation
---

## Example .env File

```bash
# API Service
HINDSIGHT_API_DATABASE_URL=postgresql://hindsight:hindsight_dev@localhost:5432/hindsight
# HINDSIGHT_API_DATABASE_SCHEMA=public  # optional, defaults to 'public'
HINDSIGHT_API_LLM_PROVIDER=groq
HINDSIGHT_API_LLM_API_KEY=gsk_xxxxxxxxxxxx

# Authentication (optional, recommended for production)
# HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
# HINDSIGHT_API_TENANT_API_KEY=your-secret-api-key

# File storage (optional, defaults to PostgreSQL native storage)
# HINDSIGHT_API_FILE_STORAGE_TYPE=s3
# HINDSIGHT_API_FILE_STORAGE_S3_BUCKET=my-hindsight-files
# HINDSIGHT_API_FILE_STORAGE_S3_REGION=us-east-1
# HINDSIGHT_API_FILE_STORAGE_S3_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
# HINDSIGHT_API_FILE_STORAGE_S3_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# Control Plane
HINDSIGHT_CP_DATAPLANE_API_URL=http://localhost:8888
```

---

For configuration issues not covered here, please [open an issue](https://github.com/vectorize-io/hindsight/issues) on GitHub.
