# Model Router Policy — AI-GW-001

## Overview

The Cognitive Router preview (`POST /api/ai/route/preview`) evaluates a routing
request against available healthy providers and returns a ranked selection.
**It never calls any model.**

## First-pass policy rules (v1)

Evaluated in order:

1. **Candidate filtering**: Start with all models where `health ∈ {healthy, unknown}`.
2. **Provider/model scoping**: If `candidate_providers` or `candidate_models` are specified in the request, filter to those.
3. **Capability filters**:
   - `request_type=embedding` → only models with `capabilities.embedding=true`
   - `requires_tools=true` → only models with `capabilities.tools=true`
   - `requires_audio=true` → only models with `capabilities.audio=true`
   - `requires_vision=true` → only models with `capabilities.vision=true`
4. **Cost filter**: If `max_cost` is set, exclude models with `cost.input_per_1m > max_cost` (null cost = pass).
5. **Privacy / local preference**:
   - `privacy_level ∈ {internal, sensitive}` → prefer local providers (`localai`, `ollama`, `vllm`, `sglang`)
   - `prefer_local=true` → same preference
6. **No candidates** → return `no_selection` with clear reason.

## Provider preference order

```
localai → ollama → vllm → sglang → litellm → openai → anthropic → google → xai → amazon_q → vertex_ai → azure_openai
```

Local providers are ranked first when `prefer_local` or `privacy_applied` is true (score adjustment: -100).

## Fallback chain

The response includes up to 4 fallback candidates after the selected model.
Format: `"{provider_id}/{model_id}"`.

## Output policy trace

```json
{
  "policy": {
    "prefer_local": true,
    "privacy_applied": true,
    "cost_limited": false
  }
}
```

## Future improvements

- Latency-based scoring (use measured `latency_ms` from health checks)
- Cost-aware ranking (use `cost.input_per_1m` when available)
- Tenant-specific provider overrides
- Model capability ground truth (replace heuristic inference)
- Streaming-only vs batch scoring
- Load-aware routing (provider queue depth)
- A/B routing experiments
