"""Memory context-pack routes.

This router now bridges the memory-controller through the Central API so the
operator panel can probe the live memory stack through the single control-plane
surface.
"""

from typing import Annotated

import httpx

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse

from app.auth.context import ContextDep, RequestContext
from app.auth.internal import INTERNAL_CONTEXT_HEADER, InternalContext, sign_context
from app.config import settings
from app.db.engine import get_session
from app.memory.context_pack import create_context_pack
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/memory", tags=["memory"])
MEMORY_CONTROLLER_URL = settings.memory_controller_url.rstrip("/")


def _signed_context(context: RequestContext) -> str:
    return sign_context(InternalContext(
        tenant_id=context.tenant_id,
        actor_id=context.actor_id,
        user_id=context.user_id,
        source_app_id=context.source_app_id,
        roles=context.roles,
        scopes=context.scopes,
        auth_method=context.auth_method,
    ))


async def _proxy_memory(
    context: RequestContext,
    path: str,
    method: str,
    *,
    body: dict | None = None,
    params: dict | None = None,
) -> Response:
    headers = {
        INTERNAL_CONTEXT_HEADER: _signed_context(context),
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.request(
                method,
                f"{MEMORY_CONTROLLER_URL}{path}",
                headers=headers,
                json=body,
                params=params,
            )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"memory-controller unreachable: {exc}") from exc

    content_type = resp.headers.get("content-type", "application/json")
    if "application/json" in content_type:
        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001
            payload = {"error": "invalid_json", "detail": resp.text}
        return JSONResponse(payload, status_code=resp.status_code)

    return Response(content=resp.text, status_code=resp.status_code, media_type=content_type)


@router.post("/context-pack")
async def build_context_pack(
    session: Annotated[AsyncSession, Depends(get_session)],
    context: ContextDep = None,
    tag: str = "default",
    version: int = 1,
) -> dict:
    """Build and return context-pack snapshot."""
    tenant_id = context.tenant_id if context else "default"
    try:
        pack = await create_context_pack(tenant_id, tag, version, session)
        return pack.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context-pack/schema")
async def get_context_pack_schema(context: ContextDep) -> dict:
    """Get context-pack JSON schema for validation."""
    return {
        "type": "object",
        "properties": {
            "tenant_id": {"type": "string"},
            "version": {"type": "integer"},
            "tag": {"type": "string"},
            "content_hash": {"type": "string"},
            "created_at": {"type": "string", "format": "date-time"},
            "content": {
                "type": "object",
                "properties": {
                    "tenant_id": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "execution_ledger": {"type": "array"},
                    "governance_state": {"type": "object"},
                    "metadata": {"type": "object"},
                },
            },
        },
        "required": ["tenant_id", "version", "tag", "content"],
    }


@router.get("/stats")
async def memory_stats(context: ContextDep) -> Response:
    """Proxy to memory-controller /api/memory/stats (dashboard panel)."""
    return await _proxy_memory(context, "/api/memory/stats", "GET")


@router.get("/audit")
async def memory_audit(context: ContextDep, limit: int = 25) -> Response:
    """Proxy to memory-controller /api/memory/audit (dashboard panel)."""
    return await _proxy_memory(context, "/api/memory/audit", "GET", params={"limit": limit})


@router.get("/categories")
async def memory_categories(context: ContextDep) -> Response:
    return await _proxy_memory(context, "/api/memory/categories", "GET")


@router.get("/route")
async def memory_route(context: ContextDep, memory_type: str) -> Response:
    return await _proxy_memory(context, "/api/memory/route", "POST", body={"memory_type": memory_type})


@router.post("/search/unified")
async def memory_search_unified(context: ContextDep, body: dict) -> Response:
    return await _proxy_memory(context, "/api/memory/search/unified", "POST", body=body)


@router.post("/search")
async def memory_search(context: ContextDep, body: dict) -> Response:
    return await _proxy_memory(context, "/api/memory/search", "POST", body=body)


@router.post("/governance/policy-check")
async def memory_policy_check(context: ContextDep, body: dict) -> Response:
    return await _proxy_memory(context, "/api/memory/governance/policy-check", "POST", body=body)
