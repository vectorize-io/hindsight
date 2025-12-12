# Models

Hindsight uses several machine learning models for different tasks.

## Overview

| Model Type | Purpose | Default | Configurable |
|------------|---------|---------|--------------|
| **Embedding** | Vector representations for semantic search | `BAAI/bge-small-en-v1.5` | Yes |
| **Cross-Encoder** | Reranking search results | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Yes |
| **LLM** | Fact extraction, reasoning, generation | Provider-specific | Yes |

All local models (embedding, cross-encoder) are automatically downloaded from HuggingFace on first run.

---

## Embedding Model

Converts text into dense vector representations for semantic similarity search.

**Default:** `BAAI/bge-small-en-v1.5` (384 dimensions, ~130MB)

**Alternatives:**

| Model | Dimensions | Use Case |
|-------|------------|----------|
| `BAAI/bge-small-en-v1.5` | 384 | Default, fast, good quality |
| `BAAI/bge-base-en-v1.5` | 768 | Higher accuracy, slower |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual (50+ languages) |

:::warning
All embedding models must produce 384-dimensional vectors to match the database schema.
:::

**Configuration:**

```bash
# Local provider (default)
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
export HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-small-en-v1.5

# TEI provider (remote)
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=tei
export HINDSIGHT_API_EMBEDDINGS_TEI_URL=http://localhost:8080
```

---

## Cross-Encoder (Reranker)

Reranks initial search results to improve precision.

**Default:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~85MB)

**Alternatives:**

| Model | Use Case |
|-------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Default, fast |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Higher accuracy |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Multilingual |

**Configuration:**

```bash
# Local provider (default)
export HINDSIGHT_API_RERANKER_PROVIDER=local
export HINDSIGHT_API_RERANKER_LOCAL_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# TEI provider (remote)
export HINDSIGHT_API_RERANKER_PROVIDER=tei
export HINDSIGHT_API_RERANKER_TEI_URL=http://localhost:8081
```

---

## LLM

Used for fact extraction, entity resolution, opinion generation, and answer synthesis.

**Supported providers:** Groq, OpenAI, Gemini, Ollama

| Provider | Recommended Model | Best For |
|----------|------------------|----------|
| **Groq** | `openai/gpt-oss-20b` | Fast inference, high throughput (recommended) |
| **OpenAI** | `gpt-4o` | Good quality |
| **Gemini** | `gemini-2.0-flash` | Good quality, cost effective |
| **Ollama** | `llama3.1` | Local deployment, privacy |

**Configuration:**

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

# Ollama (local)
export HINDSIGHT_API_LLM_PROVIDER=ollama
export HINDSIGHT_API_LLM_BASE_URL=http://localhost:11434/v1
export HINDSIGHT_API_LLM_MODEL=llama3.1
```

**Note:** The LLM is the primary bottleneck for retain operations. See [Performance](./performance) for optimization strategies.
