"""AI Gateway routes — AI-GW-001 + AI-002.

Provider registry (read/write):
  GET    /api/ai/providers
  POST   /api/ai/providers                        (AI-002: register new provider)
  GET    /api/ai/providers/{provider_id}
  PUT    /api/ai/providers/{provider_id}/enabled  (AI-002: enable/disable toggle)
  GET    /api/ai/providers/{provider_id}/health

Model inventory (AI-002 — durable, queryable):
  GET    /api/ai/models                           query params: provider_id, family, capability, health, active_only
  GET    /api/ai/models/{provider_id}             + refresh query param
  GET    /api/ai/models/{provider_id}/{model_id}  single model detail
  POST   /api/ai/models/refresh                   full inventory refresh (all enabled providers)
  POST   /api/ai/models/{provider_id}/refresh     per-provider refresh
  GET    /api/ai/models/inventory/stats           aggregate stats

Router:
  POST   /api/ai/route/preview
  POST   /api/ai/chat
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.ai import ollama, localai, litellm as litellm_adapter
from app.ai import inventory
from app.ai import providers as registry
from app.ai import router_preview as preview_engine
from app.ai.schemas import (
    InventoryStatsResponse,
    ModelInventoryQuery,
    ModelsResponse,
    ProviderEnableRequest,
    ProviderHealthResponse,
    ProviderResponse,
    ProvidersResponse,
    RegisterProviderRequest,
    RoutePreviewRequest,
    RoutePreviewResponse,
    ChatRequest,
    ChatResponse,
    ModelResponse,
)
from app.audit.logger import log_event
from app.auth.context import ContextDep

router = APIRouter(prefix="/api/ai", tags=["ai-gateway"])

_ADAPTERS = {
    "localai": localai,
    "ollama": ollama,
    "litellm": litellm_adapter,
}


# ── Provider registry ─────────────────────────────────────────────────────────

@router.get("/providers", response_model=ProvidersResponse)
async def get_providers(ctx: ContextDep) -> ProvidersResponse:
    providers = await registry.list_providers(enabled_only=True)
    return ProvidersResponse(
        providers=[ProviderResponse(**p) for p in providers],
        count=len(providers),
    )


@router.post("/providers", response_model=ProviderResponse, status_code=201)
async def register_provider(req: RegisterProviderRequest, ctx: ContextDep) -> ProviderResponse:
    """Register or update a provider endpoint (AI-002).

    Idempotent: if ``provider_id`` already exists, updates base_url and config.
    """
    p = await registry.register_provider(
        provider_id=req.provider_id,
        base_url=str(req.base_url),
        api_style=req.api_style,
        api_key_configured=req.api_key_configured,
        enabled=req.enabled,
        config=req.config,
    )
    log_event(
        tenant_id=ctx.tenant_id,
        actor_id=ctx.actor_id,
        operation="provider_register",
        resource_type="provider",
        resource_id=req.provider_id,
        outcome="success",
        metadata={"base_url": str(req.base_url), "enabled": req.enabled},
    )
    return ProviderResponse(**p)


@router.get("/providers/{provider_id}", response_model=ProviderResponse)
async def get_provider(provider_id: str, ctx: ContextDep) -> ProviderResponse:
    p = await registry.get_provider(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"provider {provider_id!r} not found")
    return ProviderResponse(**p)


@router.put("/providers/{provider_id}/enabled", response_model=ProviderResponse)
async def set_provider_enabled(
    provider_id: str,
    req: ProviderEnableRequest,
    ctx: ContextDep,
) -> ProviderResponse:
    """Enable or disable a provider (AI-002 operator control)."""
    p = await registry.set_enabled(provider_id, enabled=req.enabled)
    if not p:
        raise HTTPException(status_code=404, detail=f"provider {provider_id!r} not found")
    log_event(
        tenant_id=ctx.tenant_id,
        actor_id=ctx.actor_id,
        operation="provider_enable_toggle",
        resource_type="provider",
        resource_id=provider_id,
        outcome="success",
        metadata={"enabled": req.enabled},
    )
    return ProviderResponse(**p)


@router.get("/providers/{provider_id}/health", response_model=ProviderHealthResponse)
async def get_provider_health(provider_id: str, ctx: ContextDep) -> ProviderHealthResponse:
    p = await registry.get_provider(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"provider {provider_id!r} not found")

    adapter = _ADAPTERS.get(provider_id)
    if adapter is None:
        return ProviderHealthResponse(
            provider_id=provider_id,
            status=p["health_status"],
            details={"note": "no local health adapter; status from registry"},
        )

    result = await adapter.health()
    status: str = result.get("status", "unknown")
    if status == "ok":
        status = "healthy"

    await registry.update_health(provider_id, status)

    # AI-002: propagate health change to inventory models
    await inventory.propagate_provider_health(provider_id, status)

    log_event(
        tenant_id=ctx.tenant_id,
        actor_id=ctx.actor_id,
        operation="provider_health_check",
        resource_type="provider",
        resource_id=provider_id,
        outcome="success",
        metadata={"status": status},
    )
    return ProviderHealthResponse(provider_id=provider_id, status=status, details=result)  # type: ignore[arg-type]


# ── Model inventory (AI-002) ──────────────────────────────────────────────────

@router.get("/models/inventory/stats", response_model=InventoryStatsResponse)
async def get_inventory_stats(ctx: ContextDep) -> InventoryStatsResponse:
    """Aggregate model inventory statistics (AI-002)."""
    stats = await inventory.inventory_stats()
    return InventoryStatsResponse(**stats)


@router.post("/models/refresh", response_model=ModelsResponse)
async def refresh_all_models(ctx: ContextDep) -> ModelsResponse:
    """Refresh model inventory for all enabled providers (AI-002).

    Fetches live model lists, upserts into inventory, marks missing models inactive.
    """
    providers = await registry.list_providers(enabled_only=True)
    all_models: list[dict] = []
    refresh_summary: dict[str, dict] = {}

    for p in providers:
        adapter = _ADAPTERS.get(p["provider_id"])
        if adapter:
            models = await adapter.list_models()
            # Annotate with current provider health before upserting
            for m in models:
                m["health"] = p["health_status"]
            summary = await inventory.refresh_provider(p["provider_id"], models)
            refresh_summary[p["provider_id"]] = summary
            all_models.extend(models)

    log_event(
        tenant_id=ctx.tenant_id,
        actor_id=ctx.actor_id,
        operation="model_inventory_refresh_all",
        resource_type="model_inventory",
        outcome="success",
        metadata={"count": len(all_models), "providers": refresh_summary},
    )
    return ModelsResponse(
        models=[ModelResponse(**m) for m in all_models],
        count=len(all_models),
        source="live",
    )


@router.post("/models/{provider_id}/refresh", response_model=ModelsResponse)
async def refresh_provider_models(provider_id: str, ctx: ContextDep) -> ModelsResponse:
    """Refresh model inventory for a single provider (AI-002)."""
    p = await registry.get_provider(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"provider {provider_id!r} not found")

    adapter = _ADAPTERS.get(provider_id)
    if adapter is None:
        raise HTTPException(
            status_code=422,
            detail=f"no adapter available for provider {provider_id!r}",
        )

    models = await adapter.list_models()
    for m in models:
        m["health"] = p["health_status"]
    summary = await inventory.refresh_provider(provider_id, models)

    log_event(
        tenant_id=ctx.tenant_id,
        actor_id=ctx.actor_id,
        operation="model_inventory_refresh",
        resource_type="model_inventory",
        outcome="success",
        metadata={"provider_id": provider_id, **summary},
    )
    return ModelsResponse(
        models=[ModelResponse(**m) for m in models],
        count=len(models),
        provider_id=provider_id,
        source="live",
    )


@router.get("/models", response_model=ModelsResponse)
async def get_all_models(
    ctx: ContextDep,
    provider_id: str | None = Query(default=None, description="Filter by provider"),
    family: str | None = Query(default=None, description="Filter by model family"),
    capability: str | None = Query(default=None, description="Filter by capability key"),
    health: str | None = Query(default=None, description="Filter by health status"),
    active_only: bool = Query(default=True, description="Return only active models"),
    source: str = Query(default="inventory", description="'inventory' (DB) or 'live' (fetch)"),
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> ModelsResponse:
    """List models with optional filtering (AI-002).

    Default ``source=inventory`` returns from the durable inventory table.
    Use ``source=live`` to fetch directly from providers (no DB write).
    """
    if source == "live":
        # Live fetch path — no inventory write
        providers = await registry.list_providers(enabled_only=True)
        if provider_id:
            providers = [p for p in providers if p["provider_id"] == provider_id]
        all_models: list[dict] = []
        for p in providers:
            adapter = _ADAPTERS.get(p["provider_id"])
            if adapter:
                models = await adapter.list_models()
                if capability:
                    models = [m for m in models if m.get("capabilities", {}).get(capability)]
                if health:
                    hv = "healthy" if health == "healthy" else health
                    models = [m for m in models if p["health_status"] in (hv, health)]
                all_models.extend(models)

        log_event(
            tenant_id=ctx.tenant_id,
            actor_id=ctx.actor_id,
            operation="model_inventory_refresh",
            resource_type="model_inventory",
            outcome="success",
            metadata={"count": len(all_models), "source": "live"},
        )
        all_models_sliced = all_models[offset : offset + limit]
        return ModelsResponse(
            models=[ModelResponse(**m) for m in all_models_sliced],
            count=len(all_models_sliced),
            source="live",
        )

    # Inventory path (default)
    models = await inventory.query_models(
        provider_id=provider_id,
        family=family,
        capability=capability,
        health=health,
        active_only=active_only,
        limit=limit,
        offset=offset,
    )

    # Fallback to live if inventory empty (helps first-run / no prior refresh)
    src = source
    if not models and not (provider_id or family or capability or health):
        providers = await registry.list_providers(enabled_only=True)
        live_models: list[dict] = []
        for p in providers:
            adapter = _ADAPTERS.get(p["provider_id"])
            if adapter:
                lm = await adapter.list_models()
                for m in lm:
                    m["health"] = p["health_status"]
                live_models.extend(lm)
        if live_models:
            models = live_models
            src = "live"

    # Dev/demo seed: if still no models for ollama (common when no `ollama pull`), inject a demo
    if not models and (not provider_id or provider_id == "ollama"):
        ollama_p = next((p for p in await registry.list_providers(enabled_only=True) if p["provider_id"] == "ollama"), None)
        if ollama_p and ollama_p.get("health_status") in ("healthy", "ok", "degraded"):
            demo = {
                "provider_id": "ollama",
                "model_id": "llama3.2:latest",
                "display_name": "Llama 3.2 (demo)",
                "family": "llama",
                "capabilities": {"chat": True, "completion": True, "embedding": False, "audio": False, "tools": False, "streaming": True, "vision": False},
                "context_window": 128000,
                "cost": {"input_per_1m": None, "output_per_1m": None, "currency": "USD"},
                "latency_ms": None,
                "health": "unknown",
            }
            models = [demo]

    return ModelsResponse(
        models=[ModelResponse(**m) for m in models],
        count=len(models),
        provider_id=provider_id,
        source=src,
    )


@router.get("/models/{provider_id}", response_model=ModelsResponse)
async def get_provider_models(
    provider_id: str,
    ctx: ContextDep,
    active_only: bool = Query(default=True),
) -> ModelsResponse:
    p = await registry.get_provider(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"provider {provider_id!r} not found")

    models = await inventory.query_models(
        provider_id=provider_id,
        active_only=active_only,
    )
    if not models:
        # Fallback: live fetch if inventory empty
        adapter = _ADAPTERS.get(provider_id)
        if adapter:
            live = await adapter.list_models()
            return ModelsResponse(
                models=[ModelResponse(**m) for m in live],
                count=len(live),
                provider_id=provider_id,
                source="live",
            )
    return ModelsResponse(
        models=[ModelResponse(**m) for m in models],
        count=len(models),
        provider_id=provider_id,
        source="inventory",
    )


@router.get("/models/{provider_id}/{model_id}", response_model=ModelResponse)
async def get_single_model(provider_id: str, model_id: str, ctx: ContextDep) -> ModelResponse:
    """Return a single model's inventory record (AI-002)."""
    p = await registry.get_provider(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"provider {provider_id!r} not found")

    m = await inventory.get_model(provider_id, model_id)
    if not m:
        raise HTTPException(
            status_code=404,
            detail=f"model {model_id!r} not found in inventory for provider {provider_id!r}",
        )
    return ModelResponse(**m)


# ── Router preview ────────────────────────────────────────────────────────────

@router.post("/route/preview", response_model=RoutePreviewResponse)
async def route_preview(req: RoutePreviewRequest, ctx: ContextDep) -> RoutePreviewResponse:
    """Cognitive router preview — does NOT call any model.

    If record_decision=true, records the decision to router_decisions table.
    Respects tenant/actor from authenticated context, never from request.
    """
    # AI-002: prefer inventory for model candidates (falls back to live if empty)
    all_models = await inventory.query_models(active_only=True, limit=1000)
    if not all_models:
        # Inventory empty — fall back to live fetch
        providers = await registry.list_providers(enabled_only=True)
        for p in providers:
            adapter = _ADAPTERS.get(p["provider_id"])
            if adapter:
                models = await adapter.list_models()
                for m in models:
                    m["health"] = p["health_status"]
                all_models.extend(models)

    result = preview_engine.preview(req, all_models)
    log_event(
        tenant_id=ctx.tenant_id,
        actor_id=ctx.actor_id,
        operation="route_preview",
        resource_type="router",
        outcome="success",
        metadata={
            "request_type": req.request_type,
            "selected_provider": result.selected_provider,
            "selected_model": result.selected_model,
        },
    )

    if req.record_decision:
        from app.router import service as router_service
        status = "selected" if result.selected_model else "no_selection"
        await router_service.write_decision(
            tenant_id=ctx.tenant_id,
            actor_id=ctx.actor_id,
            request_type=req.request_type,
            selected_model=result.selected_model,
            selected_provider=result.selected_provider,
            candidate_models=result.candidate_models,
            selection_reason=result.selection_reason,
            latency_ms=result.expected_latency_ms,
            estimated_cost=result.estimated_cost,
            fallback_chain=result.fallback_chain,
            status=status,
        )

    return result


@router.post("/chat", response_model=ChatResponse)
async def chat_playground(req: ChatRequest, ctx: ContextDep) -> ChatResponse:
    """Governed chat playground endpoint — AI-006 (non-streaming).

    Routes through provider backends (LocalAI, Ollama, LiteLLM) with governance layer.
    No memory persistence; playground only.
    """
    from app.ai import chat

    try:
        return await chat.handle_chat(req, ctx.tenant_id, ctx.actor_id)
    except ValueError as ve:
        log_event(
            tenant_id=ctx.tenant_id,
            actor_id=ctx.actor_id,
            operation="chat_playground",
            resource_type="chat",
            outcome="validation_error",
            metadata={"error": str(ve)},
        )
        return ChatResponse(
            response="",
            provider="none",
            model="none",
            status="error",
            warnings=[str(ve)],
        )


@router.post("/chat/stream")
async def chat_playground_stream(req: ChatRequest, ctx: ContextDep):
    """Governed streaming chat playground endpoint — AI-006 (streaming).

    Returns Server-Sent Events (SSE) stream of response chunks.
    Routes through provider backends with governance layer.
    """
    from fastapi.responses import StreamingResponse
    from app.ai import chat

    return StreamingResponse(
        chat.handle_chat_stream(req, ctx.tenant_id, ctx.actor_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
