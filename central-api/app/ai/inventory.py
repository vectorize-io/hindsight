"""Model inventory service — AI-002.

Persists a snapshot of every model seen from each provider into the
``model_inventory`` table. The registry is the durable source of truth: the
live ``/v1/models`` endpoints are authoritative at refresh time, but
historical models remain visible with ``is_active=False`` so dashboards don't
show empty states when a provider is temporarily unreachable.

Key behaviours
--------------
* **Upsert-on-refresh**: each full provider refresh upserts all returned models
  and marks any previously-active models that are no longer returned as
  ``is_active=False``.
* **Health propagation**: when provider health changes, all active models for
  that provider inherit the new health value.
* **Query helpers**: filtering by provider, family, capability, and health so
  routes don't need to assemble queries.
* **No fake data**: context_window, cost, and latency are only set if the
  upstream payload includes them; NULL is preferred over fabricated values.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select, update

from app.db.engine import session_scope
from app.db.ids import new_id
from app.db.tables import model_inventory

logger = logging.getLogger(__name__)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _row_from_model(m: dict, now: datetime) -> dict:
    """Convert a normalized model dict to a model_inventory row dict."""
    caps = m.get("capabilities") or {}
    cost = m.get("cost") or {}
    return {
        "provider_id": m["provider_id"],
        "model_id": m["model_id"],
        "display_name": m.get("display_name"),
        "family": m.get("family", "unknown"),
        "capabilities": caps,
        "context_window": m.get("context_window"),
        "cost_input_per_1m": cost.get("input_per_1m"),
        "cost_output_per_1m": cost.get("output_per_1m"),
        "cost_currency": cost.get("currency", "USD"),
        "latency_ms": m.get("latency_ms"),
        "health": m.get("health", "unknown"),
        "is_active": True,
        "last_seen": now,
        "updated_at": now,
        "extra_metadata": m.get("metadata") or {},
    }


# ── Public API ────────────────────────────────────────────────────────────────

async def refresh_provider(provider_id: str, models: list[dict]) -> dict:
    """Upsert all ``models`` returned by a provider, mark missing ones inactive.

    Args:
        provider_id: The provider identifier (e.g. ``"localai"``).
        models: Normalized model dicts from the provider adapter.

    Returns:
        Summary dict: ``{"upserted": N, "deactivated": N, "total_active": N}``.
    """
    now = _now()
    incoming_ids = {m["model_id"] for m in models if m.get("model_id")}

    async with session_scope() as session:
        # 1. Fetch all existing rows for this provider
        existing_result = await session.execute(
            select(model_inventory).where(model_inventory.c.provider_id == provider_id)
        )
        existing_rows = {r["model_id"]: dict(r) for r in existing_result.mappings()}

        upserted = 0
        for m in models:
            mid = m.get("model_id")
            if not mid:
                continue
            row_data = _row_from_model(m, now)
            if mid in existing_rows:
                await session.execute(
                    update(model_inventory)
                    .where(
                        model_inventory.c.provider_id == provider_id,
                        model_inventory.c.model_id == mid,
                    )
                    .values(
                        display_name=row_data["display_name"],
                        family=row_data["family"],
                        capabilities=row_data["capabilities"],
                        context_window=row_data["context_window"],
                        cost_input_per_1m=row_data["cost_input_per_1m"],
                        cost_output_per_1m=row_data["cost_output_per_1m"],
                        cost_currency=row_data["cost_currency"],
                        latency_ms=row_data["latency_ms"],
                        health=row_data["health"],
                        is_active=True,
                        last_seen=now,
                        updated_at=now,
                        extra_metadata=row_data["extra_metadata"],
                    )
                )
            else:
                row_data["id"] = new_id()
                row_data["first_seen"] = now
                await session.execute(model_inventory.insert().values(**row_data))
            upserted += 1

        # 2. Mark previously-active models no longer returned as inactive
        deactivated = 0
        for mid, row in existing_rows.items():
            if row.get("is_active") and mid not in incoming_ids:
                await session.execute(
                    update(model_inventory)
                    .where(
                        model_inventory.c.provider_id == provider_id,
                        model_inventory.c.model_id == mid,
                    )
                    .values(is_active=False, updated_at=now)
                )
                deactivated += 1

    total_active = len(incoming_ids)
    logger.info(
        "model_inventory refresh provider=%s upserted=%d deactivated=%d total_active=%d",
        provider_id, upserted, deactivated, total_active,
    )
    return {"upserted": upserted, "deactivated": deactivated, "total_active": total_active}


async def propagate_provider_health(provider_id: str, health_status: str) -> int:
    """Set ``health`` on all active models for a provider.

    Returns the number of rows updated.
    """
    # Map provider health values to model health values
    # provider returns "ok" (ollama/litellm), model uses "healthy"
    model_health = "healthy" if health_status in ("healthy", "ok") else health_status
    now = _now()
    async with session_scope() as session:
        result = await session.execute(
            update(model_inventory)
            .where(
                model_inventory.c.provider_id == provider_id,
                model_inventory.c.is_active == True,
            )
            .values(health=model_health, updated_at=now)
        )
        count: int = result.rowcount  # type: ignore[assignment]
    logger.debug("propagated health=%s to %d models for provider=%s", model_health, count, provider_id)
    return count


async def query_models(
    *,
    provider_id: str | None = None,
    family: str | None = None,
    capability: str | None = None,       # "chat"|"embedding"|"tools"|"audio"|"vision"
    health: str | None = None,            # "healthy"|"degraded"|"down"|"unknown"
    active_only: bool = True,
    limit: int = 500,
    offset: int = 0,
) -> list[dict]:
    """Query model_inventory with optional filters.

    Args:
        provider_id: Filter to a specific provider.
        family: Filter by model family (e.g. ``"llama"``).
        capability: Filter by capability key (returns only models where
            ``capabilities[capability] == True``).
        health: Filter by current health value.
        active_only: If True (default), only return ``is_active=True`` rows.
        limit: Max rows to return.
        offset: Rows to skip.

    Returns:
        List of model dicts in normalized ModelResponse shape.
    """
    q = select(model_inventory)
    if active_only:
        q = q.where(model_inventory.c.is_active == True)
    if provider_id:
        q = q.where(model_inventory.c.provider_id == provider_id)
    if family:
        q = q.where(model_inventory.c.family == family)
    if health:
        q = q.where(model_inventory.c.health == health)
    q = q.order_by(model_inventory.c.provider_id, model_inventory.c.model_id)
    q = q.limit(limit).offset(offset)

    async with session_scope() as session:
        result = await session.execute(q)
        rows = result.mappings().all()

    out = []
    for r in rows:
        r = dict(r)
        caps = r.get("capabilities") or {}

        # Capability filter (JSON field — SQL can't easily filter inside JSON on SQLite)
        if capability and not caps.get(capability):
            continue

        out.append(_to_model_response(r))
    return out


async def get_model(provider_id: str, model_id: str) -> dict | None:
    """Return a single model from inventory, or None if not found."""
    async with session_scope() as session:
        result = await session.execute(
            select(model_inventory).where(
                model_inventory.c.provider_id == provider_id,
                model_inventory.c.model_id == model_id,
            )
        )
        row = result.mappings().first()
    if not row:
        return None
    return _to_model_response(dict(row))


async def inventory_stats() -> dict:
    """Return aggregate stats for the operator dashboard."""
    from sqlalchemy import func as sa_func

    async with session_scope() as session:
        total_result = await session.execute(
            select(sa_func.count()).select_from(model_inventory)
        )
        total: int = total_result.scalar() or 0

        active_result = await session.execute(
            select(sa_func.count()).select_from(model_inventory).where(
                model_inventory.c.is_active == True
            )
        )
        active: int = active_result.scalar() or 0

        healthy_result = await session.execute(
            select(sa_func.count()).select_from(model_inventory).where(
                model_inventory.c.is_active == True,
                model_inventory.c.health.in_(("healthy", "ok")),
            )
        )
        healthy: int = healthy_result.scalar() or 0

        provider_result = await session.execute(
            select(
                model_inventory.c.provider_id,
                sa_func.count().label("count"),
            )
            .where(model_inventory.c.is_active == True)
            .group_by(model_inventory.c.provider_id)
        )
        by_provider = {r["provider_id"]: r["count"] for r in provider_result.mappings()}

    return {
        "total": total,
        "active": active,
        "inactive": total - active,
        "healthy": healthy,
        "by_provider": by_provider,
    }


# ── Serialization ─────────────────────────────────────────────────────────────

def _to_model_response(r: dict) -> dict:
    """Convert a model_inventory DB row to normalized ModelResponse dict."""
    caps = r.get("capabilities") or {}
    return {
        "provider_id": r["provider_id"],
        "model_id": r["model_id"],
        "display_name": r.get("display_name") or r["model_id"],
        "family": r.get("family") or "unknown",
        "capabilities": {
            "chat": bool(caps.get("chat")),
            "completion": bool(caps.get("completion")),
            "embedding": bool(caps.get("embedding")),
            "audio": bool(caps.get("audio")),
            "tools": bool(caps.get("tools")),
            "streaming": bool(caps.get("streaming")),
            "vision": bool(caps.get("vision")),
        },
        "context_window": r.get("context_window"),
        "cost": {
            "input_per_1m": r.get("cost_input_per_1m"),
            "output_per_1m": r.get("cost_output_per_1m"),
            "currency": r.get("cost_currency") or "USD",
        },
        "latency_ms": r.get("latency_ms"),
        "health": r.get("health") or "unknown",
        "is_active": r.get("is_active", True),
        "first_seen": r.get("first_seen"),
        "last_seen": r.get("last_seen"),
        "metadata": r.get("extra_metadata") or {},
    }
