# Governed Chat Playground

**Status**: Complete (AI-006)  
**Last Updated**: 2026-06-13

The Governed Chat Playground provides a full end-to-end chat flow through the CollabMind governance layer, routing through hardened provider backends (LocalAI, Ollama, LiteLLM).

## Overview

The playground is NOT persistent chat memory. It's a testing and exploration interface where:
- Users can send prompts to any governed provider
- Router optionally selects the best provider/model
- All requests are audited and logged
- Decision tracking is optional (record_decision toggle)
- No chat history is stored (session-local only)

Future work will layer memory persistence and governance workflows (GOV-002) on top.

## Architecture

```
Console (UI) → collabmind-api (:3050) → Central API (:8000)
                                            ↓
                                    Router + Provider Selection
                                            ↓
                            LocalAI | Ollama | LiteLLM
```

### Flow

1. User opens Playground page in Console
2. Selects provider (or lets router choose)
3. Selects model from provider's inventory
4. Enters prompt + optional system prompt
5. Sets temperature, max_tokens (optional)
6. Toggles record_decision
7. Clicks "Send"
8. Central API `/api/ai/chat` endpoint receives request
9. Router may re-select provider/model if not explicit
10. Provider adapter executes chat call
11. Response returned with metadata (provider, model, latency, decision_id)
12. Console displays response + metadata

## Backend API

### Endpoint

```
POST /api/ai/chat
```

### Request

```json
{
  "prompt": "hello",
  "system_prompt": "optional",
  "provider": "localai|ollama|litellm|null",
  "model": "model-id|null",
  "temperature": 0.0-2.0,
  "max_tokens": 1-4096,
  "record_decision": true|false
}
```

- `prompt` (required): User message
- `system_prompt` (optional): System instructions
- `provider` (optional): Explicit provider ID; if omitted, router selects
- `model` (optional): Explicit model ID; if omitted, router selects
- `temperature` (optional): Sampling temperature (0-2)
- `max_tokens` (optional): Max response tokens (1-4096)
- `record_decision` (optional): If true, router decision is recorded to `router_decisions` table

### Response

```json
{
  "response": "response text",
  "provider": "localai",
  "model": "llama-3-8b",
  "status": "ok|degraded|error",
  "latency_ms": 1234,
  "decision_id": "dec-uuid-or-null",
  "warnings": ["list of warnings"]
}
```

- `response`: Chat response text
- `provider`: Provider used
- `model`: Model used
- `status`: "ok" (success), "degraded" (provider had issues), "error" (failed)
- `latency_ms`: Round-trip latency in milliseconds
- `decision_id`: Router decision ID if recorded
- `warnings`: Any warnings during execution

## Frontend UI

Located at: `/playground`

### Components

1. **Provider Selector**
   - Dropdown: LocalAI, Ollama, LiteLLM
   - Loads available providers from `/api/ai/providers`
   - Changes refresh the model selector

2. **Model Selector**
   - Dropdown: Models from selected provider
   - Loads from `/api/ai/models/{provider_id}`
   - Defaults to first available

3. **Prompt Input**
   - Textarea: Main user prompt
   - Required field

4. **System Prompt Input** (optional)
   - Textarea: System instructions
   - E.g., "You are a helpful assistant"

5. **Temperature & Max Tokens** (optional)
   - Number inputs
   - Temperature: 0-2 (default 0.7)
   - Max tokens: 1-4096 (default 256)

6. **Record Decision Toggle**
   - Checkbox: If checked, router decision is recorded
   - Useful for collecting training data

7. **Send Button**
   - Triggers `/api/ai/chat` POST request
   - Shows BlockingOverlay during execution

### Response Panel

After sending:
- **Response Text**: The model's reply in a monospace box
- **Metadata**:
  - Provider used
  - Model used
  - Status (ok/degraded/error)
  - Latency in ms
  - Decision ID (if recorded)
  - Warnings (if any)

## Governance & Audit

### Implicit Governance

The playground leverages the Central API governance layer:
- All requests go through `collabmind-api` edge (JWT auth)
- Tenant isolation: responses include only current tenant's models
- Request audit logging captures: prompt length, response length, provider, model, latency
- No sensitive data returned (provider keys are never exposed)

### Decision Recording

When `record_decision=true`:
- Router decision is recorded to `router_decisions` table
- Includes: provider selected, model selected, selection_reason, latency
- Allows operators to analyze routing behavior

### No Secret Leakage

- Request: No API keys sent
- Response: No provider credentials, no config details
- Logs: Only non-sensitive metadata
- All secrets stay on backend

## Error Handling

| Error | Response |
|-------|----------|
| Empty prompt | 400/422 validation error |
| Provider unavailable | status: "degraded", response: "Provider unavailable" |
| Provider timeout | status: "degraded", response: "Timeout" |
| Malformed response | status: "error", response: empty |
| Router no_selection | status: "error", response: "No suitable model available" |
| Invalid temperature/tokens | 400/422 validation error |

## Session State

The playground maintains minimal session state:
- Selected provider
- Selected model
- Recent prompt (for convenience, not persisted)
- Response history (last response only, cleared on new request)

No state is persisted to disk or database. Refresh → all state lost.

## Future Work

### Memory Integration (Phase 2)
- Optional "Store to memory" button
- Prompt user before storing (explicit permission)
- Integrate with write-gate governance layer

### Conversation Context (Phase 3)
- Multi-turn conversations (session-scoped)
- Maintain message history in session storage
- Use `record_decision=false` for intermediate turns

### Advanced Routing (Phase 4)
- Route constraints: privacy level, requires_tools, max_cost
- Explicit fallback chains
- Cost estimation and tracking

### User Preferences (Phase 5)
- Save favorite provider/model combinations
- Temperature presets
- System prompt templates

## Testing

Backend tests cover:
- Explicit provider/model selection
- Router-based selection
- Provider unavailable (degraded state)
- Malformed provider responses
- Temperature & max_tokens passthrough
- System prompt inclusion
- Decision recording
- No secret leakage

Run:
```bash
cd central-api
python3 -m pytest tests/test_chat_playground.py -v
```

All 12 tests pass.

## Security

### Authentication & Authorization

- Requires valid JWT from Authentik
- Tenant ID enforced from token (never from request)
- Actor ID tracked for audit

### No Persistence

- Playground chat is ephemeral
- No automatic memory writes
- Session-scoped only
- Refresh clears state

### Secret Isolation

- Provider API keys kept in backend only
- Never sent to Console or exposed in responses
- Audit logs redact sensitive fields
- Config values are internal only

## Limitations

1. **No History**: Each refresh clears state
2. **No Multi-turn**: Conversations not supported (yet)
3. **No Streaming**: Full responses only (future: SSE streaming)
4. **No File Upload**: Text-only prompts
5. **No Vision**: Even if provider supports it (UI not designed for images)
6. **No Tools**: Tool-use not exposed (future: integrated tool calling)

## Known Issues

None at release.

## References

- Central API: `/api/ai/chat` endpoint
- Provider backends: LocalAI, Ollama, LiteLLM adapters
- Router: `/api/ai/route/preview` logic
- Audit: `log_event()` in routes
