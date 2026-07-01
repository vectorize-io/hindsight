"""Provider registry service — AI-GW-001.

Single source of truth for provider configuration.

Phase 1 limitation: provider registry is global (not tenant-scoped).
Tenant-scoped provider configuration is deferred to Phase 2.

Architecture:
- CONFIG_DEFAULTS: static defaults per provider_id (display_name, type, capabilities, etc.)
- provider_endpoints table: base_url, enabled, health_status, api_key_configured, api_style
- Merged at query time: config defaults + DB overrides
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update

from app.config import settings
from app.db.engine import session_scope
from app.db.ids import new_id
from app.db.tables import provider_endpoints

# Static config defaults for all known providers.
# base_url falls back to settings env vars where applicable.
_CONFIG_DEFAULTS: dict[str, dict[str, Any]] = {
    "localai": {
        "display_name": "LocalAI",
        "provider_type": "local",
        "api_style": "localai",
        "auth_type": "bearer",
        "supports_chat": True,
        "supports_completion": True,
        "supports_embeddings": True,
        "supports_audio": False,
        "supports_tools": True,
        "supports_streaming": True,
    },
    "ollama": {
        "display_name": "Ollama",
        "provider_type": "local",
        "api_style": "ollama",
        "auth_type": "none",
        "supports_chat": True,
        "supports_completion": True,
        "supports_embeddings": True,
        "supports_audio": False,
        "supports_tools": False,
        "supports_streaming": True,
    },
    "litellm": {
        "display_name": "LiteLLM Proxy",
        "provider_type": "gateway",
        "api_style": "litellm",
        "auth_type": "bearer",
        "supports_chat": True,
        "supports_completion": True,
        "supports_embeddings": True,
        "supports_audio": False,
        "supports_tools": True,
        "supports_streaming": True,
    },
    "openai": {
        "display_name": "OpenAI",
        "provider_type": "cloud",
        "api_style": "openai_compatible",
        "auth_type": "api_key",
        "supports_chat": True,
        "supports_completion": True,
        "supports_embeddings": True,
        "supports_audio": True,
        "supports_tools": True,
        "supports_streaming": True,
    },
    "anthropic": {
        "display_name": "Anthropic",
        "provider_type": "cloud",
        "api_style": "native",
        "auth_type": "api_key",
        "supports_chat": True,
        "supports_completion": False,
        "supports_embeddings": False,
        "supports_audio": False,
        "supports_tools": True,
        "supports_streaming": True,
    },
    "google": {
        "display_name": "Google AI",
        "provider_type": "cloud",
        "api_style": "native",
        "auth_type": "api_key",
        "supports_chat": True,
        "supports_completion": True,
        "supports_embeddings": True,
        "supports_audio": True,
        "supports_tools": True,
        "supports_streaming": True,
    },
    "xai": {
        "display_name": "xAI / Grok",
        "provider_type": "cloud",
        "api_style": "openai_compatible",
        "auth_type": "api_key",
        "supports_chat": True,
        "supports_completion": True,
        "supports_embeddings": False,
        "supports_audio": False,
        "supports_tools": True,
        "supports_streaming": True,
    },
    "amazon_q": {
        "display_name": "Amazon Q / Bedrock",
        "provider_type": "enterprise",
        "api_style": "bedrock",
        "auth_type": "service_account",
        "supports_chat": True,
        "supports_completion": True,
        "supports_embeddings": True,
        "supports_audio": False,
        "supports_tools": True,
        "supports_streaming": True,
    },
    "vertex_ai": {
        "display_name": "Google Vertex AI",
        "provider_type": "enterprise",
        "api_style": "vertex",
        "auth_type": "service_account",
        "supports_chat": True,
        "supports_completion": True,
        "supports_embeddings": True,
        "supports_audio": False,
        "supports_tools": True,
        "supports_streaming": True,
    },
    "azure_openai": {
        "display_name": "Azure OpenAI",
        "provider_type": "enterprise",
        "api_style": "openai_compatible",
        "auth_type": "api_key",
        "supports_chat": True,
        "supports_completion": True,
        "supports_embeddings": True,
        "supports_audio": True,
        "supports_tools": True,
        "supports_streaming": True,
    },
    "vllm": {
        "display_name": "vLLM",
        "provider_type": "local",
        "api_style": "openai_compatible",
        "auth_type": "none",
        "supports_chat": True,
        "supports_completion": True,
        "supports_embeddings": True,
        "supports_audio": False,
        "supports_tools": True,
        "supports_streaming": True,
    },
    "sglang": {
        "display_name": "SGLang",
        "provider_type": "local",
        "api_style": "openai_compatible",
        "auth_type": "none",
        "supports_chat": True,
        "supports_completion": True,
        "supports_embeddings": False,
        "supports_audio": False,
        "supports_tools": True,
        "supports_streaming": True,
    },
}

# Env-configured base URLs. Only providers with a configured URL are auto-seeded.
def _env_base_urls() -> dict[str, str]:
    return {
        k: v for k, v in {
            "localai": settings.localai_base_url,
            "ollama": settings.ollama_url,
            "litellm": settings.litellm_url,
        }.items() if v
    }


async def _ensure_seeded() -> None:
    """Upsert env-configured providers into provider_endpoints if not present."""
    env_urls = _env_base_urls()
    if not env_urls:
        return

    now = datetime.now(timezone.utc)
    async with session_scope() as session:
        for pid, base_url in env_urls.items():
            result = await session.execute(
                select(provider_endpoints).where(provider_endpoints.c.provider_id == pid)
            )
            row = result.mappings().first()
            if not row:
                cfg = _CONFIG_DEFAULTS.get(pid, {})
                await session.execute(
                    provider_endpoints.insert().values(
                        id=new_id(),
                        provider_id=pid,
                        base_url=base_url,
                        api_key_configured=bool(
                            getattr(settings, f"{pid}_api_key", "") or
                            (pid == "litellm" and settings.litellm_api_key) or
                            (pid == "localai" and settings.localai_api_key)
                        ),
                        api_style=cfg.get("api_style", "openai_compatible"),
                        enabled=True,
                        health_status="unknown",
                        last_health_check=None,
                        config={},
                        created_at=now,
                        updated_at=now,
                    )
                )


def _merge(row: dict, defaults: dict) -> dict:
    """Merge DB row with static defaults to produce a full provider dict."""
    return {
        "id": row["id"],
        "provider_id": row["provider_id"],
        "display_name": defaults.get("display_name", row["provider_id"].title()),
        "base_url": row["base_url"],
        "provider_type": defaults.get("provider_type", "openai_compatible"),
        "api_style": row["api_style"] or defaults.get("api_style", "openai_compatible"),
        "auth_type": defaults.get("auth_type", "api_key"),
        "enabled": row["enabled"],
        "supports_chat": defaults.get("supports_chat", False),
        "supports_completion": defaults.get("supports_completion", False),
        "supports_embeddings": defaults.get("supports_embeddings", False),
        "supports_audio": defaults.get("supports_audio", False),
        "supports_tools": defaults.get("supports_tools", False),
        "supports_streaming": defaults.get("supports_streaming", False),
        "health_status": row["health_status"],
        "last_health_check": row["last_health_check"],
        "metadata": row.get("config") or {},
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


async def list_providers(enabled_only: bool = True) -> list[dict]:
    """Return all providers from DB, merged with config defaults."""
    await _ensure_seeded()
    async with session_scope() as session:
        q = select(provider_endpoints)
        if enabled_only:
            q = q.where(provider_endpoints.c.enabled == True)
        result = await session.execute(q)
        rows = result.mappings().all()

    return [_merge(dict(r), _CONFIG_DEFAULTS.get(r["provider_id"], {})) for r in rows]


async def get_provider(provider_id: str) -> dict | None:
    """Return a single provider by provider_id."""
    await _ensure_seeded()
    async with session_scope() as session:
        result = await session.execute(
            select(provider_endpoints).where(provider_endpoints.c.provider_id == provider_id)
        )
        row = result.mappings().first()

    if not row:
        return None
    return _merge(dict(row), _CONFIG_DEFAULTS.get(provider_id, {}))


async def update_health(provider_id: str, status: str) -> None:
    """Update provider health_status and last_health_check timestamp."""
    now = datetime.now(timezone.utc)
    async with session_scope() as session:
        await session.execute(
            update(provider_endpoints)
            .where(provider_endpoints.c.provider_id == provider_id)
            .values(health_status=status, last_health_check=now, updated_at=now)
        )


async def set_enabled(provider_id: str, *, enabled: bool) -> dict | None:
    """Enable or disable a provider by provider_id.

    Returns the updated provider dict, or None if the provider doesn't exist.
    Operators use this to take a provider offline without deleting its config.
    """
    now = datetime.now(timezone.utc)
    async with session_scope() as session:
        result = await session.execute(
            select(provider_endpoints).where(provider_endpoints.c.provider_id == provider_id)
        )
        row = result.mappings().first()
        if not row:
            return None
        await session.execute(
            update(provider_endpoints)
            .where(provider_endpoints.c.provider_id == provider_id)
            .values(enabled=enabled, updated_at=now)
        )
    return await get_provider(provider_id)


async def register_provider(
    *,
    provider_id: str,
    base_url: str,
    api_style: str = "openai_compatible",
    api_key_configured: bool = False,
    enabled: bool = True,
    config: dict | None = None,
) -> dict:
    """Register a new provider endpoint, or update base_url/config if it already exists.

    This is the programmatic equivalent of setting env vars for a new provider.
    Only operator-controlled fields are accepted; display_name and capability
    flags come from ``_CONFIG_DEFAULTS`` (or "unknown" if provider_id is new).

    Returns the full merged provider dict after upsert.
    """
    now = datetime.now(timezone.utc)
    async with session_scope() as session:
        result = await session.execute(
            select(provider_endpoints).where(provider_endpoints.c.provider_id == provider_id)
        )
        row = result.mappings().first()
        if row:
            await session.execute(
                update(provider_endpoints)
                .where(provider_endpoints.c.provider_id == provider_id)
                .values(
                    base_url=base_url,
                    api_style=api_style,
                    api_key_configured=api_key_configured,
                    enabled=enabled,
                    config=config or {},
                    updated_at=now,
                )
            )
        else:
            cfg = _CONFIG_DEFAULTS.get(provider_id, {})
            await session.execute(
                provider_endpoints.insert().values(
                    id=new_id(),
                    provider_id=provider_id,
                    base_url=base_url,
                    api_key_configured=api_key_configured,
                    api_style=api_style or cfg.get("api_style", "openai_compatible"),
                    enabled=enabled,
                    health_status="unknown",
                    last_health_check=None,
                    config=config or {},
                    created_at=now,
                    updated_at=now,
                )
            )
    result = await get_provider(provider_id)
    assert result is not None
    return result
