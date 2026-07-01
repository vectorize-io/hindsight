"""Authorization helpers — scope checks and confidentiality ceiling."""

from __future__ import annotations

from fastapi import Depends, HTTPException

from app.auth.context import ContextDep, RequestContext

CONFIDENTIALITY_ORDER = ["public", "internal", "private", "sensitive", "secret_blocked"]


def has_scope(ctx: RequestContext, scope: str) -> bool:
    return "*" in ctx.scopes or scope in ctx.scopes


def require_scope(scope: str):
    """FastAPI dependency factory enforcing a scope on a route."""

    def _dep(ctx: ContextDep) -> RequestContext:
        if not has_scope(ctx, scope):
            raise HTTPException(status_code=403, detail=f"missing scope: {scope}")
        return ctx

    return Depends(_dep)


def within_confidentiality(ctx: RequestContext, level: str | None) -> bool:
    """True if the actor's clearance covers the given content level."""
    if not level:
        return True
    have = CONFIDENTIALITY_ORDER.index(ctx.confidentiality_level) \
        if ctx.confidentiality_level in CONFIDENTIALITY_ORDER else -1
    need = CONFIDENTIALITY_ORDER.index(level) if level in CONFIDENTIALITY_ORDER else -1
    return not (have >= 0 and need >= 0 and need > have)
