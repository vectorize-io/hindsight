# Docker Testing

Scripts for testing Hindsight Docker images locally and in CI.

## Scripts

### `test-image.sh`

General-purpose Docker image test script. Starts a container and verifies it becomes healthy.

**Usage:**
```bash
./docker/test-image.sh <image> [target]
```

**Arguments:**
- `image` - Docker image to test (e.g., `hindsight:test`, `ghcr.io/vectorize-io/hindsight:latest`)
- `target` - Optional: `cp-only` for control plane, `api-only` for API, or `standalone` (default)

**Environment Variables:**
- `GROQ_API_KEY` - Required for API/standalone images
- `HINDSIGHT_API_LLM_PROVIDER` - LLM provider (default: `groq`)
- `HINDSIGHT_API_LLM_MODEL` - LLM model (default: `llama-3.3-70b-versatile`)
- `HINDSIGHT_API_EMBEDDINGS_PROVIDER` - Embeddings provider (for slim images)
- `HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY` - OpenAI API key for embeddings
- `HINDSIGHT_API_RERANKER_PROVIDER` - Reranker provider (for slim images)
- `HINDSIGHT_API_COHERE_API_KEY` - Cohere API key for reranking
- `SMOKE_TEST_TIMEOUT` - Timeout in seconds (default: 120)

**Examples:**

Test a full image (with local ML models):
```bash
export GROQ_API_KEY=gsk_xxx
./docker/test-image.sh hindsight:test
```

Test a slim image (with external providers):
```bash
export GROQ_API_KEY=gsk_xxx
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
export HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=sk-xxx
export HINDSIGHT_API_RERANKER_PROVIDER=cohere
export HINDSIGHT_API_COHERE_API_KEY=xxx
./docker/test-image.sh hindsight-slim:test
```

### `test-slim-local.sh`

Convenience wrapper for testing slim images locally. Automatically configures external providers.

**Usage:**
```bash
# Set API keys
export GROQ_API_KEY=gsk_xxx
export OPENAI_API_KEY=sk-xxx
export COHERE_API_KEY=xxx

# Run test
./docker/test-slim-local.sh [image]
```

**Or inline:**
```bash
GROQ_API_KEY=gsk_xxx \
OPENAI_API_KEY=sk-xxx \
COHERE_API_KEY=xxx \
./docker/test-slim-local.sh hindsight-slim:test
```

This script:
- ✅ Validates API keys are set
- ✅ Configures OpenAI embeddings automatically
- ✅ Configures Cohere reranking automatically
- ✅ Calls `test-image.sh` with the right configuration

## Building and Testing Locally

### Build a slim image

```bash
docker build \
  --build-arg INCLUDE_LOCAL_MODELS=false \
  --build-arg PRELOAD_ML_MODELS=false \
  --target standalone \
  -t hindsight-slim:test \
  -f docker/standalone/Dockerfile \
  .
```

### Test the slim image

```bash
# With API keys
export GROQ_API_KEY=gsk_xxx
export OPENAI_API_KEY=sk-xxx
export COHERE_API_KEY=xxx

# Run test
./docker/test-slim-local.sh hindsight-slim:test
```

## Expected Output

**Successful test:**
```
Starting smoke test for: hindsight-slim:test
  Target: standalone
  Health endpoint: http://localhost:8888/health
  Timeout: 120s

Starting container...
Waiting for health endpoint at http://localhost:8888/health...
  Still waiting... (10s)
  Still waiting... (20s)

Container is healthy after 25s

=== Health Response ===
{
    "status": "healthy",
    "database": "connected"
}

Smoke test PASSED
```

## CI Integration

These scripts are used in CI to validate Docker images on every PR:

- `.github/workflows/test.yml` - Runs `test-image.sh` for slim variants with OpenAI/Cohere
- `.github/workflows/release.yml` - Can optionally run smoke tests during release

See the workflows for the exact configuration.
