"""Audit middleware — automated request logging."""

from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.audit.logger import log_event


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # After the request is processed, check if a context was resolved.
        # Note: In FastAPI, dependencies (like get_context) run inside the route,
        # so we can't easily access the resolved context object here without
        # some extra wiring (e.g. request.state). 
        # For the scaffold, we'll log based on the presence of auth headers.
        
        ctx = getattr(request.state, "context", None)
        if ctx:
            log_event(
                tenant_id=ctx.tenant_id,
                actor_id=ctx.actor_id,
                source_app_id=ctx.source_app_id,
                operation=f"{request.method} {request.url.path}",
                outcome="success" if response.status_code < 400 else "error",
                metadata={"status_code": response.status_code}
            )
            
        return response
