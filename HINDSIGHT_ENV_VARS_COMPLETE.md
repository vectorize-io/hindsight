# Hindsight API - Complete Environment Variables Reference

Generated: 2026-06-27

Total environment variables: 325

## Categories

### Database
- `HINDSIGHT_API_DATABASE_BACKEND` - Database backend (pg0, external)
- `HINDSIGHT_API_DATABASE_URL` - Main database connection URL
- `HINDSIGHT_API_READ_DATABASE_URL` - Read replica URL (optional)
- `HINDSIGHT_API_READ_DB_POOL_MIN_SIZE` - Read pool min size
- `HINDSIGHT_API_READ_DB_POOL_MAX_SIZE` - Read pool max size
- `HINDSIGHT_API_MIGRATION_DATABASE_URL` - Migration DB URL (optional)
- `HINDSIGHT_API_DATABASE_SCHEMA` - Database schema name

### LLM Configuration (Global)
- `HINDSIGHT_API_LLM_PROVIDER` - Provider: openai, anthropic, gemini, vertexai, groq, ollama, lmstudio, litellm, etc.
- `HINDSIGHT_API_LLM_API_KEY` - API key for provider
- `HINDSIGHT_API_LLM_MODEL` - Model name
- `HINDSIGHT_API_LLM_BASE_URL` - **Base URL for LLM API (use this for Ollama port config!)**
- `HINDSIGHT_API_LLM_MAX_CONCURRENT` - Max concurrent LLM requests
- `HINDSIGHT_API_LLM_MAX_RETRIES` - Max retry attempts
- `HINDSIGHT_API_LLM_INITIAL_BACKOFF` - Initial retry backoff (seconds)
- `HINDSIGHT_API_LLM_MAX_BACKOFF` - Max retry backoff (seconds)
- `HINDSIGHT_API_LLM_TIMEOUT` - Request timeout (seconds)
- `HINDSIGHT_API_LLM_REASONING_EFFORT` - Reasoning effort (low, medium, high)
- `HINDSIGHT_API_LLM_GROQ_SERVICE_TIER` - Groq service tier (auto, on_demand, flex)
- `HINDSIGHT_API_LLM_OPENAI_SERVICE_TIER` - OpenAI service tier (default, flex)
- `HINDSIGHT_API_LLM_BEDROCK_SERVICE_TIER` - Bedrock service tier
- `HINDSIGHT_API_LLM_GEMINI_SERVICE_TIER` - Gemini service tier (default, flex)
- `HINDSIGHT_API_LLM_EXTRA_BODY` - Extra body params (JSON)
- `HINDSIGHT_API_LLM_DEFAULT_HEADERS` - Default headers (JSON)
- `HINDSIGHT_API_LLM_STRICT_SCHEMA` - Strict schema validation
- `HINDSIGHT_API_LLM_SEND_BANK_AS_USER` - Send bank ID as user

### Multi-LLM Strategy
- `HINDSIGHT_API_LLM_STRATEGY` - Global LLM routing strategy (JSON)
- `HINDSIGHT_API_RETAIN_LLM_STRATEGY` - Retain operation LLM strategy
- `HINDSIGHT_API_REFLECT_LLM_STRATEGY` - Reflect operation LLM strategy
- `HINDSIGHT_API_CONSOLIDATION_LLM_STRATEGY` - Consolidation operation LLM strategy
- `HINDSIGHT_API_LLM_LITELLMROUTER_CONFIG` - LiteLLM Router config (JSON)

### Per-Operation LLM (Retain)
- `HINDSIGHT_API_RETAIN_LLM_PROVIDER` - Override provider for retain
- `HINDSIGHT_API_RETAIN_LLM_API_KEY` - Override API key
- `HINDSIGHT_API_RETAIN_LLM_MODEL` - Override model
- `HINDSIGHT_API_RETAIN_LLM_BASE_URL` - Override base URL
- `HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT` - Override concurrency
- `HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES` - Override retries
- `HINDSIGHT_API_RETAIN_LLM_INITIAL_BACKOFF` - Override initial backoff
- `HINDSIGHT_API_RETAIN_LLM_MAX_BACKOFF` - Override max backoff
- `HINDSIGHT_API_RETAIN_LLM_TIMEOUT` - Override timeout
- `HINDSIGHT_API_RETAIN_LLM_LITELLMROUTER_CONFIG` - Retain LiteLLM config

### Per-Operation LLM (Reflect)
- `HINDSIGHT_API_REFLECT_LLM_PROVIDER` - Override provider for reflect
- `HINDSIGHT_API_REFLECT_LLM_API_KEY` - Override API key
- `HINDSIGHT_API_REFLECT_LLM_MODEL` - Override model
- `HINDSIGHT_API_REFLECT_LLM_BASE_URL` - Override base URL
- `HINDSIGHT_API_REFLECT_LLM_MAX_CONCURRENT` - Override concurrency
- `HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES` - Override retries
- `HINDSIGHT_API_REFLECT_LLM_INITIAL_BACKOFF` - Override initial backoff
- `HINDSIGHT_API_REFLECT_LLM_MAX_BACKOFF` - Override max backoff
- `HINDSIGHT_API_REFLECT_LLM_TIMEOUT` - Override timeout
- `HINDSIGHT_API_REFLECT_LLM_LITELLMROUTER_CONFIG` - Reflect LiteLLM config

### Per-Operation LLM (Consolidation)
- `HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER` - Override provider for consolidation
- `HINDSIGHT_API_CONSOLIDATION_LLM_API_KEY` - Override API key
- `HINDSIGHT_API_CONSOLIDATION_LLM_MODEL` - Override model
- `HINDSIGHT_API_CONSOLIDATION_LLM_BASE_URL` - Override base URL
- `HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT` - Override concurrency
- `HINDSIGHT_API_CONSOLIDATION_LLM_MAX_RETRIES` - Override retries
- `HINDSIGHT_API_CONSOLIDATION_LLM_INITIAL_BACKOFF` - Override initial backoff
- `HINDSIGHT_API_CONSOLIDATION_LLM_MAX_BACKOFF` - Override max backoff
- `HINDSIGHT_API_CONSOLIDATION_LLM_TIMEOUT` - Override timeout
- `HINDSIGHT_API_CONSOLIDATION_LLM_LITELLMROUTER_CONFIG` - Consolidation LiteLLM config

### Embeddings - Provider Selection
- `HINDSIGHT_API_EMBEDDINGS_PROVIDER` - Provider: local, onnx, tei, openai, gemini, vertexai, cohere, litellm, openrouter, ollama

### Embeddings - Local (sentence-transformers)
- `HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL` - HuggingFace model name
- `HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU` - Force CPU usage
- `HINDSIGHT_API_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE` - Trust remote code

### Embeddings - ONNX
- `HINDSIGHT_API_EMBEDDINGS_ONNX_MODEL_ID` - Model ID
- `HINDSIGHT_API_EMBEDDINGS_ONNX_MODEL_PATH` - Local model path
- `HINDSIGHT_API_EMBEDDINGS_ONNX_TOKENIZER_NAME_OR_PATH` - Tokenizer path
- `HINDSIGHT_API_EMBEDDINGS_ONNX_FILE` - ONNX file name
- `HINDSIGHT_API_EMBEDDINGS_ONNX_DIMENSIONS` - Output dimensions
- `HINDSIGHT_API_EMBEDDINGS_ONNX_MAX_TOKENS` - Max token length
- `HINDSIGHT_API_EMBEDDINGS_ONNX_POOLING` - Pooling strategy
- `HINDSIGHT_API_EMBEDDINGS_ONNX_NORMALIZE` - Normalize embeddings
- `HINDSIGHT_API_EMBEDDINGS_ONNX_QUERY_PREFIX` - Query prefix
- `HINDSIGHT_API_EMBEDDINGS_ONNX_PASSAGE_PREFIX` - Passage prefix
- `HINDSIGHT_API_EMBEDDINGS_ONNX_OUTPUT_NAME` - ONNX output tensor name

### Embeddings - TEI (Text Embeddings Inference)
- `HINDSIGHT_API_EMBEDDINGS_TEI_URL` - TEI server URL

### Embeddings - OpenAI Compatible
- `HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY` - API key
- `HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL` - Model name
- `HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL` - Base URL
- `HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE` - Batch size
- `HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS` - Output dimensions

### Embeddings - Ollama
- `HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE` - Ollama API base URL (e.g., http://localhost:11434)
- `HINDSIGHT_API_EMBEDDINGS_OLLAMA_MODEL` - Ollama model name (e.g., nomic-embed-text)
- `HINDSIGHT_API_EMBEDDINGS_OLLAMA_DIMENSIONS` - Embedding dimensions

### Embeddings - Gemini
- `HINDSIGHT_API_EMBEDDINGS_GEMINI_API_KEY` - Gemini API key
- `HINDSIGHT_API_EMBEDDINGS_GEMINI_MODEL` - Gemini model name
- `HINDSIGHT_API_EMBEDDINGS_GEMINI_OUTPUT_DIMENSIONALITY` - Output dimensions

### Embeddings - Vertex AI
- `HINDSIGHT_API_EMBEDDINGS_VERTEXAI_PROJECT_ID` - GCP project ID
- `HINDSIGHT_API_EMBEDDINGS_VERTEXAI_REGION` - GCP region
- `HINDSIGHT_API_EMBEDDINGS_VERTEXAI_SERVICE_ACCOUNT_KEY` - Service account key (JSON)

### Embeddings - Cohere
- `HINDSIGHT_API_EMBEDDINGS_COHERE_API_KEY` - Cohere API key
- `HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL` - Cohere model name
- `HINDSIGHT_API_EMBEDDINGS_COHERE_BASE_URL` - Cohere base URL
- `HINDSIGHT_API_EMBEDDINGS_COHERE_OUTPUT_DIMENSIONS` - Output dimensions

### Embeddings - OpenRouter
- `HINDSIGHT_API_EMBEDDINGS_OPENROUTER_API_KEY` - OpenRouter API key
- `HINDSIGHT_API_EMBEDDINGS_OPENROUTER_MODEL` - OpenRouter model name

### Reranking - Cohere
- `HINDSIGHT_API_RERANKER_COHERE_API_KEY` - Cohere reranker API key
- `HINDSIGHT_API_RERANKER_COHERE_MODEL` - Cohere reranker model
- `HINDSIGHT_API_RERANKER_COHERE_BASE_URL` - Cohere reranker base URL

### Special Providers
- `HINDSIGHT_API_FIREWORKS_ACCOUNT_ID` - Fireworks account ID
- `HINDSIGHT_API_FIREWORKS_BATCH_BASE_URL` - Fireworks batch API URL
- `HINDSIGHT_API_FIREWORKS_BATCH_MAX_WAIT_SECONDS` - Max wait for batch
- `HINDSIGHT_API_OPENROUTER_API_KEY` - OpenRouter API key

## CRITICAL FINDING

**WRONG ENV VAR USAGE DETECTED:**
- ❌ `HINDSIGHT_API_LLM_OLLAMA_API_BASE` - **DOES NOT EXIST IN CODE**
- ✅ `HINDSIGHT_API_LLM_BASE_URL` - **CORRECT VARIABLE**

The code has NO provider-specific base URL variables. All providers use the generic `HINDSIGHT_API_LLM_BASE_URL`.

**Correct Ollama Configuration:**
```bash
# LLM (use port 11435 for LLM lane)
HINDSIGHT_API_LLM_PROVIDER=ollama
HINDSIGHT_API_LLM_BASE_URL=http://localhost:11435/v1
HINDSIGHT_API_LLM_MODEL=ollama/llama3.2:latest

# Embeddings (use port 11434 for embeddings lane)
HINDSIGHT_API_EMBEDDINGS_PROVIDER=ollama
HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE=http://localhost:11434
HINDSIGHT_API_EMBEDDINGS_OLLAMA_MODEL=nomic-embed-text
HINDSIGHT_API_EMBEDDINGS_OLLAMA_DIMENSIONS=384
```

---

This reference generated by Claude (Anthropic AI Agent) on 2026-06-27.
