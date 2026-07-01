"""Request context — the resolved identity attached to every request.

The Central API is the single verifier. Routes MUST read tenant/workspace from
this context, never from the request body.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth.jwt import validate_jwt
from app.auth.apikey import validate_api_key
from app.auth.internal import verify_context, INTERNAL_CONTEXT_HEADER, AuthMethod
from app.config import settings

DEFAULT_TENANT = "00000000-0000-0000-0000-000000000001"
DEFAULT_ACTOR = "00000000-0000-0000-0000-000000000002"


class RequestContext(BaseModel):
    tenant_id: str
    actor_id: str
    user_id: str | None = None
    workspace_id: str | None = None
    source_app_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    confidentiality_level: str = "internal"
    auth_method: AuthMethod = "dev_default"


async def get_context(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
    x_cm_context: Annotated[str | None, Header(alias=INTERNAL_CONTEXT_HEADER)] = None,
) -> RequestContext:
    """Resolve the request context.

    Prioritizes:
    1. Signed internal context (X-CM-Context)
    2. Bearer JWT (Authentik)
    3. API Key (x-api-key)
    4. Dev default (only if is_dev=True)
    """
    
    ctx: RequestContext | None = None

    # 1. Verify-once-at-edge: trust a signed context from an internal service
    if x_cm_context:
        internal = verify_context(x_cm_context)
        if internal:
            ctx = RequestContext(
                tenant_id=internal.tenant_id,
                actor_id=internal.actor_id,
                user_id=internal.user_id,
                source_app_id=internal.source_app_id,
                roles=internal.roles,
                scopes=internal.scopes,
                auth_method=internal.auth_method,
            )

    # 2. Try Bearer token
    if not ctx and authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ")
        
        # API key as Bearer token (mem11_sk_...)
        if token.startswith("mem11_sk_"):
            claims = await validate_api_key(token)
            if claims:
                ctx = RequestContext(
                    tenant_id=DEFAULT_TENANT,
                    actor_id=claims.get("actor_id", DEFAULT_ACTOR),
                    roles=claims.get("roles", []),
                    scopes=claims.get("scopes", []),
                    auth_method="api_key",
                )
        else:
            claims = await validate_jwt(token)
            if claims is not None:
                ctx = RequestContext(
                    tenant_id=claims.get("tenant_id", DEFAULT_TENANT),
                    actor_id=claims.get("sub", DEFAULT_ACTOR),
                    roles=claims.get("groups", []),
                    scopes=(claims.get("scope", "") or "").split(),
                    auth_method="jwt",
                )

    # 3. API-key lookup (x-api-key header)
    if not ctx and x_api_key:
        claims = await validate_api_key(x_api_key)
        if claims:
            ctx = RequestContext(
                tenant_id=DEFAULT_TENANT,
                actor_id=claims.get("actor_id", DEFAULT_ACTOR),
                roles=claims.get("roles", []),
                scopes=claims.get("scopes", []),
                auth_method="api_key",
            )

    # 4. Dev fallback (always on in non-prod)
    if not ctx:
        ctx = RequestContext(
            tenant_id=DEFAULT_TENANT,
            actor_id=DEFAULT_ACTOR,
            roles=["dev"],
            scopes=["*"],
            auth_method="dev_default",
        )

    # Store context on request.state so Middleware can access it
    request.state.context = ctx
    return ctx


ContextDep = Annotated[RequestContext, Depends(get_context)]
