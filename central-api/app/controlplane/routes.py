"""Core control-plane endpoints (non-connector-specific)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.controlplane.deps import PrincipalDep, require_member
from app.db import repositories as repo
from app.governance import agent_control

router = APIRouter(tags=["control-plane"])


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class AgentActionRequest(BaseModel):
    agent_id: str
    action: str
    workspace_id: str | None = None
    target_resource: str | None = None
    metadata: dict = Field(default_factory=dict)


@router.get("/me")
async def me(principal: PrincipalDep) -> dict:
    user, context, _ = principal
    return {
        "user": {"id": user["id"], "email": user["email"],
                 "is_operator": user["is_operator"]},
        "auth_method": context.auth_method,
        "scopes": context.scopes,
    }


@router.get("/workspaces")
async def list_workspaces(principal: PrincipalDep) -> dict:
    user, _, session = principal
    return {"workspaces": await repo.list_workspaces_for_user(session, user["id"])}


@router.post("/workspaces", status_code=201)
async def create_workspace(body: WorkspaceCreate, principal: PrincipalDep) -> dict:
    user, _, session = principal
    return await repo.create_workspace(session, name=body.name, owner_id=user["id"])


@router.get("/source-documents")
async def list_source_documents(workspace_id: str, principal: PrincipalDep,
                                connector_id: str | None = None) -> dict:
    user, _, session = principal
    await require_member(session, workspace_id=workspace_id, user_id=user["id"])
    return {"documents": await repo.list_documents(session, workspace_id=workspace_id,
                                                    connector_id=connector_id)}


@router.post("/source-documents/{document_id}/disable")
async def disable_document(document_id: str, workspace_id: str, principal: PrincipalDep) -> dict:
    user, context, session = principal
    await require_member(session, workspace_id=workspace_id, user_id=user["id"])
    doc = await repo.get_document(session, document_id)
    if not doc or doc["workspace_id"] != workspace_id:
        return {"error": "not_found"}
    await repo.set_document_enabled(session, document_id, False)
    await repo.write_audit(session, action="file_disabled", actor_id=context.actor_id,
                           workspace_id=workspace_id, source=doc["provider"],
                           source_file_id=doc["external_id"])
    return {"document_id": document_id, "enabled": False}


@router.get("/ingestion-jobs")
async def list_ingestion_jobs(workspace_id: str, principal: PrincipalDep,
                              status: str | None = None) -> dict:
    user, _, session = principal
    await require_member(session, workspace_id=workspace_id, user_id=user["id"])
    return {"jobs": await repo.list_jobs(session, workspace_id=workspace_id, status=status)}


@router.get("/audit-events")
async def list_audit_events(workspace_id: str, principal: PrincipalDep, limit: int = 100) -> dict:
    user, _, session = principal
    await require_member(session, workspace_id=workspace_id, user_id=user["id"])
    return {"events": await repo.list_audit(session, workspace_id=workspace_id, limit=limit)}


@router.get("/agent-activity")
async def list_agent_activity(workspace_id: str, principal: PrincipalDep, limit: int = 100) -> dict:
    user, _, session = principal
    await require_member(session, workspace_id=workspace_id, user_id=user["id"])
    return {"activity": await repo.list_agent_activity(session, workspace_id=workspace_id,
                                                        limit=limit)}


@router.post("/agent-activity")
async def request_agent_action(body: AgentActionRequest, principal: PrincipalDep) -> dict:
    user, _, session = principal
    if body.workspace_id:
        await require_member(session, workspace_id=body.workspace_id, user_id=user["id"])
    return await agent_control.request_action(
        session, agent_id=body.agent_id, action=body.action, workspace_id=body.workspace_id,
        target_resource=body.target_resource, metadata=body.metadata,
    )
