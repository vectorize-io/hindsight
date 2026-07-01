# LiteLLM Provider Adapter

**Status**: Hardened (AI-004)  
**Last Updated**: 2026-06-13

LiteLLM is a gateway provider backend that proxies to multiple LLM APIs (OpenAI, Anthropic, Google, xAI, Azure, etc.).

## Overview

LiteLLM runs independently and exposes an OpenAI-compatible API surface. Central API uses the LiteLLM adapter to:
- Check health (`/health`)
- List available models (`/models`)
- Normalize model metadata (remove secrets, standardize schema)

No direct model calls go through the adapter — chat/completions endpoints are reserved for future governance layers.

## Configuration

### Environment Variables

```bash
LITELLM_BASE_URL=http://litellm.internal:4000
LITELLM_API_KEY=<key>  # Optional; Bearer token if LiteLLM requires auth
```

- `LITELLM_BASE_URL` — LiteLLM proxy endpoint (e.g., localhost:4000 for local dev)
- `LITELLM_API_KEY` — Optional Bearer token. If set, `Authorization: Bearer <key>` is sent with all requests.

### Initialization

The provider registry auto-seeds LiteLLM from `LITELLM_BASE_URL` if configured:

```python
from app.ai import providers
await providers._ensure_seeded()  # Called on first /api/ai/providers request
```

The provider record includes:
- `api_key_configured`: Boolean (true if LITELLM_API_KEY is set)
- `health_status`: Current state ("ok", "degraded", "down")
- `last_health_check`: Timestamp of most recent health call

## API Endpoints

### Health Check

```bash
GET /api/ai/providers/litellm/health
```

Returns:
```json
{
  "provider_id": "litellm",
  "status": "ok|degraded|down",
  "details": { "code": 200, ... }
}
```

**Behavior**:
- 5s timeout on health requests
- HTTP 200 → status = "ok"
- HTTP non-2xx → status = "degraded"
- Connection error / timeout → status = "down"

### List Models

```bash
GET /api/ai/models/litellm
```

Returns:
```json
{
  "models": [
    {
      "provider_id": "litellm",
      "model_id": "gpt-4",
      "display_name": "Gpt 4",
      "family": "unknown",
      "capabilities": { "chat": true, "completion": true, ... },
      "context_window": null,
      "cost": { "input_per_1m": null, "output_per_1m": null, "currency": "USD" },
      "health": "unknown",
      "metadata": {}
    }
  ],
  "count": 1
}
```

**Behavior**:
- 15s timeout on model listing
- LiteLLM response: `{"data": [{"id": "gpt-4", "object": "model", ...}, ...]}`
- Normalization: filters secrets, extracts model_id, standardizes capabilities
- Error (timeout/malformed) → empty list (provider marked degraded separately)

## Normalization

The adapter normalizes LiteLLM responses to a standardized CollabMind model schema:

### Input
LiteLLM `/models` returns:
```json
{
  "data": [
    {
      "id": "gpt-4-turbo",
      "object": "model",
      "created": 1234567890,
      "owned_by": "openai",
      "api_key": "sk-secret"  // ← Never exposed
    }
  ]
}
```

### Output (Normalized)
```json
{
  "provider_id": "litellm",
  "model_id": "gpt-4-turbo",
  "display_name": "Gpt 4 Turbo",
  "family": "unknown",
  "capabilities": {
    "chat": true,
    "completion": true,
    "embedding": false,
    "audio": false,
    "tools": true,
    "streaming": true,
    "vision": false
  },
  "context_window": null,
  "cost": { "input_per_1m": null, "output_per_1m": null, "currency": "USD" },
  "health": "unknown",
  "metadata": {}  // Safe metadata only; secrets filtered
}
```

### Secret Filtering

The adapter actively filters sensitive fields:
- Removes any field containing: `key`, `secret`, `token`, `auth`, `credential`
- Only preserves: `owned_by`, `created`, and model-specific safe fields

## Timeouts

| Operation | Timeout | Behavior |
|-----------|---------|----------|
| Health check | 5s | Returns "down" on timeout |
| Model listing | 15s | Returns empty list on timeout |
| Chat/completions | 120s | Not used (interface only) |

## Error Handling

### Unavailable LiteLLM
If LiteLLM is unreachable:
1. Health check fails → provider status = "down"
2. Model listing returns `[]`
3. Router avoids selecting LiteLLM models
4. Fallback to local providers (LocalAI, Ollama) or returns "no_selection"

### Malformed Response
If LiteLLM returns invalid JSON or unexpected structure:
1. Logged as warning
2. Returns empty model list
3. Provider not immediately marked down (health check separate)

### Rate Limiting
If LiteLLM responds with 429 (Too Many Requests):
1. Status = "degraded" (not "down" — transient)
2. Next health check in 30s may recover

## Security

### No Secrets Exposed
- API key never returned in `/api/ai/providers/*` responses
- Adapter filters all secret-bearing fields from model metadata
- Health endpoints do not leak backend details

### Optional Authentication
- If `LITELLM_API_KEY` is empty: requests are unauthenticated
- If set: passed as `Authorization: Bearer <key>` header
- Falls back gracefully if LiteLLM is public

### Tenant Isolation
LiteLLM provider endpoints respect Central API tenant context:
- `X-CM-Context` header ensures tenant_id isolation
- All operations logged with tenant_id for audit trails
- No cross-tenant data leakage

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
    resource_id="litellm",
    outcome="success",
    metadata={"status": "ok"}
)
```

## Testing

All LiteLLM adapter tests pass:
- `test_health_ok` — LiteLLM healthy
- `test_health_degraded` — LiteLLM returns 503
- `test_health_timeout` — LiteLLM timeout
- `test_list_models_ok` — Model normalization
- `test_list_models_timeout` — Error handling
- `test_normalize_filters_secrets` — Secret filtering

Run:
```bash
cd central-api
python3 -m pytest tests/test_ai_gateway.py::TestLiteLLMAdapter -v
```

## Future Work

1. **Full chat integration** (AI-006) — Route chat requests to LiteLLM with governance policy checks
2. **Cost estimation** — Extract real costs from LiteLLM model metadata
3. **Capability inference** — Guess vision/audio/tools support from model_id patterns
4. **Per-tenant configuration** — Allow different LiteLLM backends per tenant
