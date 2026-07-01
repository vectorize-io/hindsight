"""AI provider schemas — AI-GW-001."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

ProviderHealthStatus = Literal["unknown", "healthy", "degraded", "down"]
ProviderType = Literal["local", "gateway", "cloud", "enterprise", "openai_compatible"]
ApiStyle = Literal["openai_compatible", "native", "litellm", "ollama", "localai", "vertex", "bedrock"]
AuthType = Literal["none", "api_key", "bearer", "oauth", "service_account"]
RequestType = Literal["chat", "reason", "tool", "retrieval", "voice", "embedding", "other"]
RouterMode = Literal["standard", "reason", "deep_research", "multi_agent", "architect"]
PrivacyLevel = Literal["normal", "sensitive", "internal"]


class ModelCapabilities(BaseModel):
    chat: bool = False
    completion: bool = False
    embedding: bool = False
    audio: bool = False
    tools: bool = False
    streaming: bool = False
    vision: bool = False


class ModelCost(BaseModel):
    input_per_1m: Optional[float] = None
    output_per_1m: Optional[float] = None
    currency: str = "USD"


class ModelResponse(BaseModel):
    provider_id: str
    model_id: str
    display_name: str
    family: str
    capabilities: ModelCapabilities
    context_window: Optional[int] = None
    cost: ModelCost = Field(default_factory=ModelCost)
    latency_ms: Optional[int] = None
    health: ProviderHealthStatus = "unknown"
    # AI-002: inventory fields (present when served from DB, absent on live-fetch)
    is_active: Optional[bool] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    id: str
    provider_id: str
    display_name: str
    base_url: str
    provider_type: ProviderType
    api_style: ApiStyle
    auth_type: AuthType
    enabled: bool
    supports_chat: bool = False
    supports_completion: bool = False
    supports_embeddings: bool = False
    supports_audio: bool = False
    supports_tools: bool = False
    supports_streaming: bool = False
    health_status: ProviderHealthStatus
    last_health_check: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ProvidersResponse(BaseModel):
    providers: list[ProviderResponse] = Field(default_factory=list)
    count: int = 0


class ProviderHealthResponse(BaseModel):
    provider_id: str
    status: ProviderHealthStatus
    details: dict[str, Any] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class ModelsResponse(BaseModel):
    models: list[ModelResponse] = Field(default_factory=list)
    count: int = 0
    provider_id: Optional[str] = None
    # AI-002: query metadata
    source: str = "live"  # "live" | "inventory"
    inventory_stats: Optional[dict[str, Any]] = None


# ── AI-002: Inventory query parameters ─────────────────────

class ModelInventoryQuery(BaseModel):
    provider_id: Optional[str] = None
    family: Optional[str] = None
    capability: Optional[str] = None          # "chat"|"embedding"|"tools"|"audio"|"vision"
    health: Optional[ProviderHealthStatus] = None
    active_only: bool = True
    limit: int = Field(default=500, ge=1, le=5000)
    offset: int = Field(default=0, ge=0)


class InventoryStatsResponse(BaseModel):
    total: int
    active: int
    inactive: int
    healthy: int
    by_provider: dict[str, int] = Field(default_factory=dict)


# ── AI-002: Provider registration ─────────────────────────

class RegisterProviderRequest(BaseModel):
    provider_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9_-]+$")
    base_url: str = Field(..., min_length=1, max_length=2048)
    api_style: ApiStyle = "openai_compatible"
    api_key_configured: bool = False
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class ProviderEnableRequest(BaseModel):
    enabled: bool


# ── Route preview ─────────────────────────────────────────────

class RouteConstraints(BaseModel):
    prefer_local: bool = False
    max_cost: Optional[float] = None
    requires_tools: bool = False
    requires_vision: bool = False
    requires_audio: bool = False
    requires_embedding: bool = False
    privacy_level: PrivacyLevel = "normal"


class RoutePreviewRequest(BaseModel):
    request_type: RequestType = "chat"
    mode: RouterMode = "standard"
    constraints: RouteConstraints = Field(default_factory=RouteConstraints)
    candidate_providers: list[str] = Field(default_factory=list)
    candidate_models: list[str] = Field(default_factory=list)
    record_decision: bool = False


class RoutePolicy(BaseModel):
    prefer_local: bool = False
    privacy_applied: bool = False
    cost_limited: bool = False


class RoutePreviewResponse(BaseModel):
    selected_provider: Optional[str] = None
    selected_model: Optional[str] = None
    candidate_models: list[str] = Field(default_factory=list)
    selection_reason: str
    fallback_chain: list[str] = Field(default_factory=list)
    estimated_cost: Optional[float] = None
    expected_latency_ms: Optional[int] = None
    policy: RoutePolicy = Field(default_factory=RoutePolicy)


# ── Chat Playground ──────────────────────────────────────────

class ChatRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    provider: Optional[str] = None  # Optional; router selects if omitted
    model: Optional[str] = None     # Optional; router selects if omitted
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    record_decision: bool = False


class ChatResponse(BaseModel):
    response: str
    provider: str
    model: str
    status: Literal["ok", "degraded", "error"] = "ok"
    latency_ms: int = 0
    decision_id: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
