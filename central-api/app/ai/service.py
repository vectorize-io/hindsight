"""AI provider service — AI-001."""
from datetime import datetime
from app.ai import localai
from app.ai.schemas import ProvidersResponse, ProviderResponse, ProviderHealthResponse, ModelsResponse, ModelResponse
from app.config import settings

async def list_providers() -> ProvidersResponse:
    """List all configured AI providers."""
    providers = [
        ProviderResponse(
            id="localai-default",
            provider_id="localai",
            base_url=settings.localai_base_url,
            api_style="openai_compatible",
            enabled=True,
            health_status="unknown",
            last_health_check=None,
            config={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    ]
    return ProvidersResponse(providers=providers, count=len(providers))

async def get_provider_health(provider_id: str) -> ProviderHealthResponse:
    """Check health of a specific provider."""
    if provider_id != "localai":
        raise ValueError(f"Unknown provider: {provider_id}")
    health_result = await localai.health()
    return ProviderHealthResponse(
        provider_id=provider_id,
        status=health_result.get("status", "unknown"),
        details=health_result,
        checked_at=datetime.utcnow(),
    )

async def list_models(provider_id: str | None = None) -> ModelsResponse:
    """List all models, optionally filtered by provider."""
    if provider_id and provider_id != "localai":
        return ModelsResponse(models=[], count=0, provider_id=provider_id)
    raw_models = await localai.list_models()
    models = [ModelResponse(**m) for m in raw_models]
    return ModelsResponse(models=models, count=len(models), provider_id=provider_id)
