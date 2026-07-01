"""Cognitive Router preview — AI-GW-001 first-pass policy.

Evaluates healthy provider/model candidates and returns the best selection
without calling any external model.

Policy rules (v1):
1. privacy_level=internal|sensitive → prefer local providers first (localai, ollama, vllm)
2. request_type=embedding → filter to providers with embedding capability
3. requires_tools → filter out models without tools capability
4. requires_audio → filter out models without audio capability
5. prefer_local=True → prefer localai/ollama/vllm
6. No healthy candidates → return no_selection cleanly
"""

from __future__ import annotations

from app.ai.schemas import RoutePreviewRequest, RoutePreviewResponse, RoutePolicy

_LOCAL_PROVIDERS = {"localai", "ollama", "vllm", "sglang"}

# Provider ordering preference for fallback chain
_PROVIDER_PREFERENCE_ORDER = [
    "localai", "ollama", "vllm", "sglang",
    "litellm", "openai", "anthropic", "google", "xai",
    "amazon_q", "vertex_ai", "azure_openai",
]


def _score_provider(pid: str, prefer_local: bool) -> int:
    """Lower score = higher priority."""
    try:
        idx = _PROVIDER_PREFERENCE_ORDER.index(pid)
    except ValueError:
        idx = len(_PROVIDER_PREFERENCE_ORDER)
    if prefer_local and pid in _LOCAL_PROVIDERS:
        idx -= 100
    return idx


def preview(
    req: RoutePreviewRequest,
    available_models: list[dict],
) -> RoutePreviewResponse:
    """Return routing decision for req without calling any model.

    Args:
        req: RoutePreviewRequest with constraints and request type.
        available_models: Normalized ModelResponse dicts from provider registry.

    Returns:
        RoutePreviewResponse with selection and policy trace.
    """
    c = req.constraints
    privacy_applies = c.privacy_level in ("internal", "sensitive")
    prefer_local = c.prefer_local or privacy_applies

    policy = RoutePolicy(
        prefer_local=prefer_local,
        privacy_applied=privacy_applies,
        cost_limited=c.max_cost is not None,
    )

    # Start with all healthy (or unknown-health) candidates
    candidates = [m for m in available_models if m.get("health") in ("healthy", "unknown")]

    # If caller specified candidate_providers/models, filter to those
    if req.candidate_providers:
        candidates = [m for m in candidates if m["provider_id"] in req.candidate_providers]
    if req.candidate_models:
        candidates = [m for m in candidates if m["model_id"] in req.candidate_models]

    # Capability filters
    if req.request_type == "embedding":
        candidates = [m for m in candidates if m["capabilities"].get("embedding")]
    if c.requires_tools:
        candidates = [m for m in candidates if m["capabilities"].get("tools")]
    if c.requires_audio:
        candidates = [m for m in candidates if m["capabilities"].get("audio")]
    if c.requires_vision:
        candidates = [m for m in candidates if m["capabilities"].get("vision")]

    # Cost filter
    if c.max_cost is not None:
        cost_ok = []
        for m in candidates:
            input_cost = m.get("cost", {}).get("input_per_1m")
            if input_cost is None or input_cost <= c.max_cost:
                cost_ok.append(m)
        candidates = cost_ok

    if not candidates:
        return RoutePreviewResponse(
            selected_provider=None,
            selected_model=None,
            candidate_models=[],
            selection_reason="no_selection: no healthy providers match the constraints",
            fallback_chain=[],
            policy=policy,
        )

    # Sort by policy preference
    candidates.sort(key=lambda m: _score_provider(m["provider_id"], prefer_local))

    selected = candidates[0]
    fallback = [f"{m['provider_id']}/{m['model_id']}" for m in candidates[1:5]]

    reason_parts = []
    if privacy_applies:
        reason_parts.append(f"privacy_level={c.privacy_level} → prefer local")
    if c.prefer_local:
        reason_parts.append("prefer_local=true")
    if req.request_type == "embedding":
        reason_parts.append("embedding request")
    reason_parts.append(f"selected {selected['provider_id']}/{selected['model_id']} (highest policy rank)")

    return RoutePreviewResponse(
        selected_provider=selected["provider_id"],
        selected_model=selected["model_id"],
        candidate_models=[m["model_id"] for m in candidates],
        selection_reason="; ".join(reason_parts),
        fallback_chain=fallback,
        estimated_cost=None,
        expected_latency_ms=selected.get("latency_ms"),
        policy=policy,
    )
