# Ollama Provider Adapter

**Status**: Hardened (AI-005)  
**Last Updated**: 2026-06-13

Ollama is a lightweight local LLM runtime that runs on user machines or local servers. Central API uses the Ollama adapter to manage model inventory and health.

## Overview

Ollama runs locally (default port 11434) and exposes a REST API compatible with OpenAI. Central API uses the Ollama adapter to:
- Check health (root `/` endpoint)
- List available models (`/api/tags`)
- Normalize model metadata (extract quantization, family, size, capabilities)

No direct model calls go through the adapter — chat/embed endpoints are reserved for future governance layers (AI-006).

## Configuration

### Environment Variables

```bash
OLLAMA_BASE_URL=http://localhost:11434
```

- `OLLAMA_BASE_URL` — Ollama endpoint (local dev) or remote endpoint
- No API key required (local provider)

### Initialization

The provider registry auto-seeds Ollama from `OLLAMA_BASE_URL` if configured:

```python
from app.ai import providers
await providers._ensure_seeded()  # Called on first /api/ai/providers request
```

The provider record includes:
- `api_key_configured`: Always false (Ollama is local, no auth)
- `health_status`: Current state ("ok", "degraded", "down")
- `last_health_check`: Timestamp of most recent health call

## API Endpoints

### Health Check

```bash
GET /api/ai/providers/ollama/health
```

Returns:
```json
{
  "provider_id": "ollama",
  "status": "ok|degraded|down",
  "details": { "code": 200 }
}
```

**Behavior**:
- 5s timeout on health requests
- HTTP 200 → status = "ok"
- HTTP non-2xx → status = "degraded"
- Connection error / timeout → status = "down"

### List Models

```bash
GET /api/ai/models/ollama
```

Returns:
```json
{
  "models": [
    {
      "provider_id": "ollama",
      "model_id": "llama2:7b-q4_0",
      "display_name": "Llama2 7b Q4 0",
      "family": "llama",
      "capabilities": { "chat": true, "embedding": false, "vision": false, ... },
      "context_window": null,
      "cost": { "input_per_1m": null, "output_per_1m": null, "currency": "USD" },
      "health": "unknown",
      "metadata": {
        "size": 3826798592,
        "modified_at": "2024-01-01T00:00:00Z",
        "quantization": "q4_0"
      }
    }
  ],
  "count": 1
}
```

**Behavior**:
- 15s timeout on model listing
- Ollama response: `{"models": [{"name": "llama2:7b", "size": ..., "modified_at": ..., "digest": ...}, ...]}`
- Normalization: extracts quantization, family, size, and capabilities
- Error (timeout/malformed) → empty list (provider marked degraded separately)

## Normalization

The adapter normalizes Ollama responses to a standardized CollabMind model schema:

### Input
Ollama `/api/tags` returns:
```json
{
  "models": [
    {
      "name": "llama2:7b-q4_0",
      "size": 3826798592,
      "modified_at": "2024-01-01T00:00:00Z",
      "digest": "sha256:deadbeef..."
    }
  ]
}
```

### Output (Normalized)
```json
{
  "provider_id": "ollama",
  "model_id": "llama2:7b-q4_0",
  "display_name": "Llama2 7b Q4 0",
  "family": "llama",
  "capabilities": {
    "chat": true,
    "completion": true,
    "embedding": false,
    "audio": false,
    "tools": false,
    "streaming": true,
    "vision": false
  },
  "context_window": null,
  "cost": { "input_per_1m": null, "output_per_1m": null, "currency": "USD" },
  "health": "unknown",
  "metadata": {
    "size": 3826798592,
    "modified_at": "2024-01-01T00:00:00Z",
    "quantization": "q4_0"
  }
}
```

### Metadata Extraction

The adapter extracts and preserves rich metadata:

| Field | Source | Example |
|-------|--------|---------|
| `size` | `model.size` | 3826798592 (bytes) |
| `modified_at` | `model.modified_at` | "2024-01-01T00:00:00Z" |
| `quantization` | Model ID pattern | "q4_0" from "llama2:7b-**q4_0**" |

### Family Detection

Detected from model_id via keyword matching:

| Model ID | Family |
|----------|--------|
| llama2:7b | llama |
| mistral:7b | mistral |
| gemma:2b | gemini |
| phi:2b | phi |
| unknown-name | unknown |

### Capability Inference

Capabilities detected from model_id:
- `chat` → true if NOT embedding model
- `completion` → true if NOT embedding model
- `embedding` → true if "embed" in model_id
- `vision` → true if "vision" or "llava" in model_id
- `tools` → false (Ollama lacks native tool support)
- `streaming` → true (Ollama always streams)

## Timeouts

| Operation | Timeout | Behavior |
|-----------|---------|----------|
| Health check | 5s | Returns "down" on timeout |
| Model listing | 15s | Returns empty list on timeout |
| Chat/embed | 120s | Not used (interface only) |

## Error Handling

### Unavailable Ollama
If Ollama is unreachable or offline:
1. Health check fails → provider status = "down"
2. Model listing returns `[]`
3. Router avoids selecting Ollama models
4. Fallback to cloud providers or returns "no_selection"

### Malformed Response
If Ollama returns invalid JSON or unexpected structure:
1. Logged as warning
2. Returns empty model list
3. Provider not immediately marked down (health check separate)

### Connection Refused
If Ollama port is not listening:
1. Captured as connection error in health check
2. Status = "down"
3. Retry on next health check cycle (30s)

## Security

### No Secrets Exposed
- No API key required (Ollama is local)
- Model metadata does not leak authentication material
- All responses safe for untrusted frontends

### Local-Only Provider
- Ollama is never exposed as public edge
- Always accessed through Central API governance layer
- `/api/ai/providers/ollama/*` requires JWT authentication
- Tenant isolation enforced at Central API level

### Tenant Isolation
Ollama provider endpoints respect Central API tenant context:
- `X-CM-Context` header ensures tenant_id isolation
- All operations logged with tenant_id for audit trails
- No cross-tenant model access possible

## Monitoring

### Health Check Frequency
The Console Model Center refreshes health every 30s when open.

### Metrics Logged
Each health check logs:
- Tenant ID, actor ID
- Provider ID, status, latency
- Operation: "provider_health_check"

Example:
```python
log_event(
    tenant_id=ctx.tenant_id,
    actor_id=ctx.actor_id,
    operation="provider_health_check",
    resource_type="provider",
    resource_id="ollama",
    outcome="success",
    metadata={"status": "ok"}
)
```

## Testing

All Ollama adapter tests pass:
- `test_health_ok` — Ollama healthy
- `test_health_degraded` — Ollama returns 503
- `test_health_timeout` — Ollama timeout
- `test_health_connection_error` — Ollama refuses connection
- `test_list_models_success` — Model listing and normalization
- `test_list_models_empty` — Empty model list
- `test_list_models_timeout` — Timeout handling
- `test_list_models_malformed_response` — Malformed JSON handling
- `test_list_models_http_error` — HTTP error handling
- `test_normalize_model_chat` — Chat model normalization
- `test_normalize_model_embedding` — Embedding model normalization
- `test_normalize_model_vision` — Vision capability detection
- `test_normalize_extracts_quantization` — Quantization extraction
- `test_family_detection` — Model family detection

Run:
```bash
cd central-api
python3 -m pytest tests/test_ai_gateway.py::TestOllamaAdapter -v
```

## Local Development

### Starting Ollama

```bash
# macOS (via Homebrew)
brew services start ollama

# Docker
docker run -d --name ollama -p 11434:11434 ollama/ollama

# Manual
ollama serve
```

### Pulling Models

```bash
ollama pull llama2:7b
ollama pull mistral:7b
ollama pull nomic-embed-text  # embedding model
ollama pull llava:latest      # vision model
```

### Listing Models

```bash
curl http://localhost:11434/api/tags
```

## Future Work

1. **Full chat integration** (AI-006) — Route chat requests to Ollama with governance policy checks
2. **Context window detection** — Extract from model metadata or pull from model card
3. **Per-tenant model selection** — Allow operators to restrict which Ollama models are available per tenant
4. **Cost estimation** — Infer cost from model size and quantization
5. **GPU monitoring** — Track GPU memory usage and inference latency
