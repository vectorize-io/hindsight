# AI Gateway Foundation — AI-GW-001

## Purpose

The AI Gateway is the governed control plane for all model provider access in CollabMind.
No service (MCP, console, retrieval) may call LocalAI, Ollama, LiteLLM, or cloud models directly.
All model requests go through Central API → AI Gateway → provider adapter.

## Architecture

```
Console / MCP / Chat
       │
       ▼
  Central API :8000
  ├─ /api/ai/providers        Provider registry
  ├─ /api/ai/models           Model inventory
  ├─ /api/ai/route/preview    Cognitive Router preview (no model call)
  └─ /api/router/decisions    Decision audit log
       │
       ▼
  Provider Adapters
  ├─ LocalAI     (localai.py)   → https://localai.collabmind.dev
  ├─ Ollama      (ollama.py)    → CENTRAL_API_OLLAMA_URL
  └─ LiteLLM     (litellm.py)  → CENTRAL_API_LITELLM_URL
```

## Provider Registry

- Defined in `app/ai/providers.py`
- DB table: `provider_endpoints` (SQLAlchemy Core, `app/db/tables.py`)
- Static config defaults per provider_id: `_CONFIG_DEFAULTS` dict
- Auto-seeded on first request from env vars
- Phase 1: global registry (not tenant-scoped). Tenant-scoped providers are Phase 2.

### Registered providers (Phase 1)

| provider_id    | type         | api_style          | local adapter |
|----------------|--------------|--------------------|---------------|
| localai        | local        | localai            | ✓             |
| ollama         | local        | ollama             | ✓             |
| litellm        | gateway      | litellm            | ✓             |
| openai         | cloud        | openai_compatible  | —             |
| anthropic      | cloud        | native             | —             |
| google         | cloud        | native             | —             |
| xai            | cloud        | openai_compatible  | —             |
| amazon_q       | enterprise   | bedrock            | —             |
| vertex_ai      | enterprise   | vertex             | —             |
| azure_openai   | enterprise   | openai_compatible  | —             |
| vllm           | local        | openai_compatible  | —             |
| sglang         | local        | openai_compatible  | —             |

## Adapters

Each adapter in `app/ai/` implements:
- `health() → dict` — returns `{status: healthy|degraded|down, ...}`
- `list_models() → list[dict]` — returns normalized model shape
- `chat_completion()` / `embeddings()` — interface stubs (AI-003)

## Model Normalization

All adapters return the same shape:
```json
{
  "provider_id": "localai",
  "model_id": "llama-3-8b",
  "display_name": "Llama 3 8B",
  "family": "llama",
  "capabilities": { "chat": true, "completion": true, "embedding": false,
                    "audio": false, "tools": false, "streaming": true, "vision": false },
  "context_window": null,
  "cost": { "input_per_1m": null, "output_per_1m": null, "currency": "USD" },
  "latency_ms": null,
  "health": "unknown",
  "metadata": {}
}
```

## Router Preview

`POST /api/ai/route/preview` evaluates healthy candidates against constraints without calling any model.
See `model-router-policy.md` for policy rules.

## Router Decisions

`GET /api/router/decisions` — returns decisions ordered newest first, tenant-isolated.
`write_decision()` in `app/router/service.py` — used by route_preview and future real model calls.
Health checks and model listings do NOT create router decisions.

## Audit Events

| operation                | when                                  |
|--------------------------|---------------------------------------|
| `provider_health_check`  | GET /api/ai/providers/{id}/health     |
| `model_inventory_refresh`| GET /api/ai/models                    |
| `route_preview`          | POST /api/ai/route/preview            |

## Governance Boundary

- Tenant/actor always from `ContextDep` (auth context), never from request body
- Provider secrets in env only (`CENTRAL_API_*` prefix), never in DB or API responses
- `api_key_configured` boolean in DB — never the actual key
- MCP must not call provider adapters directly
