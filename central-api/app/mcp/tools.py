"""MCP tools (v0.1) — thin, governed wrappers over the control-plane service layer.

Each tool: (1) records an mcp_tool_called audit event, (2) enforces workspace
membership, (3) routes through repositories/services. There is deliberately NO
path from here to Google Drive, Qdrant, the raw DB, or token storage.

``search_governed_documents`` applies fail-closed retrieval governance: it
returns only enabled documents the principal is permitted to access on the
source. (No chat/vector search yet — this proves the governance gate.)
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.connectors.google_drive import audit_actions as A
from app.connectors.google_drive import service as gdrive
from app.db import repositories as repo
from app.governance.retrieval_policy import Principal, evaluate_retrieval

# The only tools exposed in v0.1.
ALLOWED_TOOLS = (
    "list_workspaces",
    "list_connected_sources",
    "list_source_documents",
    "sync_source",
    "get_source_audit",
    "list_ingestion_jobs",
    "search_governed_documents",
)


async def _audit_call(session: AsyncSession, *, tool: str, actor_id: str,
                      workspace_id: str | None, **meta) -> None:
    await repo.write_audit(session, action=A.MCP_TOOL_CALLED, actor_id=actor_id,
                           workspace_id=workspace_id, source="mcp",
                           metadata={"tool": tool, **meta})


async def _require_member(session: AsyncSession, *, workspace_id: str, user_id: str) -> bool:
    return await repo.is_workspace_member(session, workspace_id=workspace_id, user_id=user_id)


async def list_workspaces(session: AsyncSession, *, actor_id: str) -> dict:
    await _audit_call(session, tool="list_workspaces", actor_id=actor_id, workspace_id=None)
    return {"workspaces": await repo.list_workspaces_for_user(session, actor_id)}


async def list_connected_sources(session: AsyncSession, *, actor_id: str,
                                 workspace_id: str) -> dict:
    await _audit_call(session, tool="list_connected_sources", actor_id=actor_id,
                      workspace_id=workspace_id)
    if not await _require_member(session, workspace_id=workspace_id, user_id=actor_id):
        return {"error": "not_workspace_member"}
    return {"connectors": await repo.list_connectors(session, workspace_id)}


async def list_source_documents(session: AsyncSession, *, actor_id: str,
                                workspace_id: str) -> dict:
    await _audit_call(session, tool="list_source_documents", actor_id=actor_id,
                      workspace_id=workspace_id)
    if not await _require_member(session, workspace_id=workspace_id, user_id=actor_id):
        return {"error": "not_workspace_member"}
    return {"documents": await repo.list_documents(session, workspace_id=workspace_id)}


async def sync_source(session: AsyncSession, *, actor_id: str, workspace_id: str) -> dict:
    await _audit_call(session, tool="sync_source", actor_id=actor_id, workspace_id=workspace_id)
    if not await _require_member(session, workspace_id=workspace_id, user_id=actor_id):
        return {"error": "not_workspace_member"}
    return await gdrive.sync(session, workspace_id=workspace_id, actor_id=actor_id)


async def get_source_audit(session: AsyncSession, *, actor_id: str, workspace_id: str,
                           limit: int = 100) -> dict:
    await _audit_call(session, tool="get_source_audit", actor_id=actor_id,
                      workspace_id=workspace_id)
    if not await _require_member(session, workspace_id=workspace_id, user_id=actor_id):
        return {"error": "not_workspace_member"}
    return {"events": await repo.list_audit(session, workspace_id=workspace_id, limit=limit)}


async def list_ingestion_jobs(session: AsyncSession, *, actor_id: str, workspace_id: str) -> dict:
    await _audit_call(session, tool="list_ingestion_jobs", actor_id=actor_id,
                      workspace_id=workspace_id)
    if not await _require_member(session, workspace_id=workspace_id, user_id=actor_id):
        return {"error": "not_workspace_member"}
    return {"jobs": await repo.list_jobs(session, workspace_id=workspace_id)}


async def search_governed_documents(session: AsyncSession, *, actor_id: str, workspace_id: str,
                                    email: str | None = None, domain: str | None = None) -> dict:
    """Return only documents the principal is permitted to access. Fail closed."""
    await _audit_call(session, tool="search_governed_documents", actor_id=actor_id,
                      workspace_id=workspace_id)
    is_member = await _require_member(session, workspace_id=workspace_id, user_id=actor_id)
    if not is_member:
        return {"error": "not_workspace_member"}

    principal = Principal(user_id=actor_id, email=email, domain=domain, is_workspace_member=True)
    docs = await repo.list_documents(session, workspace_id=workspace_id)
    results = []
    denied = 0
    for doc in docs:
        connector = await repo.get_connector(session, doc["connector_id"])
        perms = await repo.list_document_permissions(session, doc["id"])
        decision = evaluate_retrieval(principal=principal, connector=connector, document=doc,
                                      permissions=perms or None)
        if decision.allowed:
            results.append({"document_id": doc["id"], "name": doc["name"],
                            "external_id": doc["external_id"]})
        else:
            denied += 1
            await repo.write_audit(session, action=A.RETRIEVAL_PERMISSION_DENIED, actor_id=actor_id,
                                   workspace_id=workspace_id, source=doc["provider"],
                                   source_file_id=doc["external_id"], status="denied",
                                   metadata={"reason": decision.reason})
    return {"results": results, "denied_count": denied}
