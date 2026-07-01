# LocalAI Provider — AI-GW-001

## Endpoint

- Public: `https://localai.collabmind.dev`
- OpenAPI spec: `https://localai.collabmind.dev/swagger/doc.json` ✓ reachable

## Env vars

| Var                          | Required | Default                           |
|------------------------------|----------|-----------------------------------|
| `CENTRAL_API_LOCALAI_BASE_URL` | No     | `https://localai.collabmind.dev`  |
| `CENTRAL_API_LOCALAI_API_KEY`  | No     | `""` (no auth)                    |

## Discovered API (from OpenAPI spec)

| Endpoint                  | Method | Purpose                        |
|---------------------------|--------|--------------------------------|
| `/readyz`                 | GET    | Health check                   |
| `/v1/models`              | GET    | List models (OpenAI compat)    |
| `/v1/chat/completions`    | POST   | Chat (OpenAI compat)           |
| `/v1/embeddings`          | POST   | Embeddings (OpenAI compat)     |
| `/api/agent/jobs`         | GET    | Agent job listing              |

## Health behavior

- `GET /readyz` → `200` = healthy, non-2xx = degraded, connection error = down
- Health status is persisted to `provider_endpoints.health_status` after each check

## Model listing behavior

- `GET /v1/models` → `{"data": [{"id": "model-name", "object": "model", ...}]}`
- Models are normalized via `normalize_model()` in `app/ai/localai.py`
- Capability inference from model ID (embed keywords, vision keywords, instruct suffix)
- Family inference by regex patterns

## Limitations (Phase 1)

- `chat_completion()` and `embeddings()` are interface-only stubs — not called from governance routes yet
- Audio endpoint not verified
- Agent jobs endpoint not integrated
- Model capability inference is heuristic (no introspection endpoint in LocalAI spec)
