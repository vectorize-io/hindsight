# Docker Model Runner Setup

**Migration Date:** 2026-06-27  
**Replaced:** Ollama (ports 11434, 11435)  
**Reason:** Better concurrency, request tracing, benchmarking, native Metal GPU

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Hindsight Workers (4+)                                      │
│   - Parallel batch_retain jobs                              │
│   - Concurrent LLM extraction requests                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Nginx Proxy (localhost:6000)                                │
│   - Exposes model-runner.docker.internal to host            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Docker Model Runner                                         │
│   - Built-in request queuing (handles 20+ concurrent)       │
│   - llama.cpp-metal + vllm-metal backends                   │
│   - Per-model resource limits                               │
│   - Request/response logging                                │
└─────────────────────────────────────────────────────────────┘
```

## Services

### 1. Docker Model Runner (Built-in Docker Desktop)
- **Status**: `docker model status`
- **Models**: `docker model ps`
- **Logs**: `docker model logs`
- **Requests**: `docker model requests -f`

### 2. Nginx Proxy (localhost:6000)
- **Container**: `docker-model-runner-proxy`
- **Port**: 6000 → model-runner.docker.internal/v1/
- **Compose**: `/Users/oliververmeulen/hindsight/docker-compose-model-proxy.yml`

## Usage

### Start Services

```bash
# 1. Load models (auto-unload after 5 minutes idle)
docker model run --detach nomic-embed-text-v1.5

# 2. Start nginx proxy
cd /Users/oliververmeulen/hindsight
docker-compose -f docker-compose-model-proxy.yml up -d

# 3. Verify
curl http://localhost:6000/models
```

### Hindsight Configuration

**Embeddings** (via Docker Model Runner):
```bash
HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
HINDSIGHT_API_EMBEDDINGS_OPENAI_API_BASE=http://localhost:6000
HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=nomic-embed-text-v1.5
HINDSIGHT_API_EMBEDDINGS_DIMENSION=768
```

**LLM** (use cloud providers for production):
```bash
# Option 1: Gemini (recommended - you have Google Cloud credits)
HINDSIGHT_API_LLM_PROVIDER=gemini
HINDSIGHT_API_LLM_API_KEY=your-key
HINDSIGHT_API_LLM_MODEL=gemini-1.5-flash

# Option 2: OpenAI
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_API_KEY=your-key
HINDSIGHT_API_LLM_MODEL=gpt-4o-mini

# Option 3: Anthropic
HINDSIGHT_API_LLM_PROVIDER=anthropic
HINDSIGHT_API_LLM_API_KEY=your-key
HINDSIGHT_API_LLM_MODEL=claude-3-5-haiku-20241022

# Option 4: Groq (fast, cheap)
HINDSIGHT_API_LLM_PROVIDER=groq
HINDSIGHT_API_LLM_API_KEY=your-key
HINDSIGHT_API_LLM_MODEL=llama-3.1-8b-instant
```

## Observability

### Request Tracing
```bash
# Watch all embedding requests in real-time
docker model requests -f --model nomic-embed-text-v1.5

# See full request/response payloads
docker model requests --model nomic-embed-text-v1.5 | jq
```

### Benchmarking
```bash
# Test concurrency limits (1, 2, 4, 8 parallel requests)
docker model bench nomic-embed-text-v1.5

# Custom concurrency levels
docker model bench nomic-embed-text-v1.5 --concurrency 1,4,8,16,32
```

### Model Management
```bash
# List loaded models
docker model ps

# Unload model immediately
docker model unload nomic-embed-text-v1.5

# Force keep model loaded
docker model run --detach --keep-alive 24h nomic-embed-text-v1.5
```

## Advantages over Ollama

| Feature | Ollama | Docker Model Runner |
|---------|--------|-------------------|
| **Concurrency** | Single-threaded, timeouts | Built-in queue, handles 20+ parallel |
| **Request Tracing** | No logging | Full request/response capture |
| **Benchmarking** | Manual | `docker model bench` with concurrency tests |
| **Backends** | llama.cpp only | llama.cpp, vllm, diffusers, mlx |
| **Resource Limits** | Process-wide | Per-model CPU/GPU isolation |
| **Observability** | Basic logs | `docker model logs`, `requests`, `ps` |

## Troubleshooting

### Proxy not accessible
```bash
docker ps --filter "name=docker-model-runner-proxy"
docker logs docker-model-runner-proxy
```

### Model not responding
```bash
docker model ps  # Check if model is loaded
docker model logs | tail -50
```

### Port conflicts
```bash
lsof -i :6000  # Check what's using port 6000
```

## Cleanup

```bash
# Stop proxy
docker-compose -f docker-compose-model-proxy.yml down

# Unload all models
docker model unload --all

# Remove proxy container
docker rm -f docker-model-runner-proxy
```

## Files

- **Proxy compose**: `docker-compose-model-proxy.yml`
- **Nginx config**: `docker-model-nginx.conf`
- **Hindsight env**: `.env`
- **This guide**: `DOCKER_MODEL_RUNNER.md`
