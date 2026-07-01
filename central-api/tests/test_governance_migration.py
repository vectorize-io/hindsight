"""Surgical test for migrated governance logic."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request, Header, Depends
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.auth.internal import sign_context, verify_context, InternalContext
from app.auth.context import get_context, RequestContext
from app.audit.logger import get_events, _EVENTS

def test_signed_context_roundtrip():
    ctx = InternalContext(
        tenant_id="test-tenant",
        actor_id="test-actor",
        auth_method="internal_service"
    )
    signed = sign_context(ctx)
    assert "." in signed
    
    verified = verify_context(signed)
    assert verified is not None
    assert verified.tenant_id == "test-tenant"
    assert verified.auth_method == "internal_service"

def test_context_dependency_with_signed_header():
    app = FastAPI()
    
    @app.get("/test")
    async def test_route(ctx: RequestContext = Depends(get_context)):
        return {"tenant": ctx.tenant_id}
    
    client = TestClient(app)
    
    ctx = InternalContext(
        tenant_id="governance-tenant",
        actor_id="gov-actor",
        auth_method="internal_service"
    )
    header_val = sign_context(ctx)
    
    response = client.get("/test", headers={"x-cm-context": header_val})
    assert response.status_code == 200
    assert response.json() == {"tenant": "governance-tenant"}

def test_audit_middleware_records_event():
    from app.audit.middleware import AuditMiddleware
    app = FastAPI()
    app.add_middleware(AuditMiddleware)
    
    @app.get("/audited")
    async def audited_route(request: Request):
        # Manually set context as if get_context ran
        request.state.context = RequestContext(
            tenant_id="audit-tenant",
            actor_id="audit-actor",
            auth_method="jwt"
        )
        return {"ok": True}
    
    client = TestClient(app)
    _EVENTS.clear() # Reset for test
    
    response = client.get("/audited")
    assert response.status_code == 200
    
    events = get_events("audit-tenant")
    assert len(events) == 1
    assert events[0].operation == "GET /audited"
    assert events[0].actor_id == "audit-actor"
