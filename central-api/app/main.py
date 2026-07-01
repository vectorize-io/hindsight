"""CollabMind Central API — FastAPI application entry point.

The control plane: the only public surface. Engines sit behind adapters; this
app enforces identity, routing, policy, audit, and context-pack construction.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.ai.routes import router as ai_router
from app.api_center.routes import router as api_center_router
from app.audit.middleware import AuditMiddleware
from app.audit.routes import router as audit_router
from app.config import settings
from app.connectors.routes import router as connectors_router
from app.controlplane.routes import router as controlplane_router
from app.db.engine import init_models
from app.governance.approval_routes import router as approval_router
from app.health.routes import router as health_router
from app.health.dashboard_routes import router as dashboard_router
from app.execution.routes import router as execution_router
from app.mcp.routes import router as mcp_router
from app.memory.routes import router as memory_router
from app.observability.routes import router as observability_router, public_router as observability_public_router
from app.operator.routes import router as operator_router
from app.retrieval.routes import router as retrieval_router
from app.router.routes import router as router_router
from app.security.middleware import add_security_headers, setup_cors, RateLimitMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dev/test convenience: ensure control-plane tables exist. Production runs
    # Alembic migrations and may set CENTRAL_API_AUTO_CREATE_TABLES=0 to skip.
    if settings.is_dev:
        await init_models()
    yield


app = FastAPI(
    title="CollabMind Central API",
    version=__version__,
    description="Memory control plane. The only public surface; engines/connectors are backends.",
    lifespan=lifespan,
)

# Security middleware (order matters)
setup_cors(app)
app.add_middleware(RateLimitMiddleware)
add_security_headers(app)

# Audit logging — fires for all identity-resolved requests
app.add_middleware(AuditMiddleware)

# Public liveness + engine health
app.include_router(health_router)
app.include_router(observability_public_router)
# Dashboard audit (must be before controlplane to catch /audit-events without workspace_id)
app.include_router(audit_router)
# Control-plane (users, workspaces, documents, jobs, audit, agent activity)
app.include_router(controlplane_router)
app.include_router(connectors_router)
# Governed memory domains
app.include_router(memory_router)
app.include_router(retrieval_router)
app.include_router(operator_router)
app.include_router(router_router)
# MCP gateway — tool registry
app.include_router(mcp_router)
# AI provider registry and model inventory
app.include_router(ai_router)
# API Center — route registry and service catalog
app.include_router(api_center_router)
# Governance — approval queue and quarantine
app.include_router(approval_router)
# Execution ledger
app.include_router(execution_router)
# Observability — runtime health and metrics
app.include_router(observability_router)
# Dashboard overrides (LAST - overrides workspace-scoped routes when no workspace_id)
app.include_router(dashboard_router)


@app.get("/", tags=["meta"])
async def root() -> dict:
    return {
        "service": "central-api",
        "version": __version__,
        "env": settings.env,
        "docs": "/docs",
    }
