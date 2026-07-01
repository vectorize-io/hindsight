"""Connector endpoints, incl. the read-only Google Drive connector.

v0.1: manual sync only — no webhooks, no background sync, no write-back, no
deletion from Drive. Agents/MCP/GUI reach Drive only through these endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.connectors.google_drive import oauth
from app.connectors.google_drive import service as gdrive
from app.controlplane.deps import PrincipalDep, require_member
from app.db import repositories as repo
from app.ingestion import worker

router = APIRouter(tags=["connectors"])


class GDriveConnect(BaseModel):
    workspace_id: str
    folder_ids: list[str] = Field(default_factory=list)
    account_email: str | None = None
    # Dev/test only: preload a fake Drive. Ignored when real OAuth is configured.
    seed: dict | None = None


class WorkspaceBody(BaseModel):
    workspace_id: str


# ---- generic connector views -------------------------------------------

@router.get("/connectors")
async def list_connectors(workspace_id: str, principal: PrincipalDep) -> dict:
    user, _, session = principal
    await require_member(session, workspace_id=workspace_id, user_id=user["id"])
    rows = await repo.list_connectors(session, workspace_id)
    # Never leak token material; connectors hold only references/status.
    return {"connectors": rows}


# ---- Google Drive -------------------------------------------------------
# NOTE: the specific google-drive routes are declared BEFORE the generic
# /connectors/{connector_id}/status route so "google-drive" is never captured
# as a connector_id path param.

@router.get("/connectors/google-drive/oauth-config")
async def gdrive_oauth_config() -> dict:
    """Non-secret OAuth status (configured?, redirect, scopes). No secrets."""
    return oauth.oauth_status()


@router.post("/connectors/google-drive/connect")
async def gdrive_connect(body: GDriveConnect, principal: PrincipalDep) -> dict:
    user, context, session = principal
    await require_member(session, workspace_id=body.workspace_id, user_id=user["id"])
    connector = await gdrive.connect(
        session, workspace_id=body.workspace_id, actor_id=context.actor_id,
        folder_ids=body.folder_ids, account_email=body.account_email, seed=body.seed,
    )
    return {"connector_id": connector["id"], "status": connector["status"]}


@router.post("/connectors/google-drive/disconnect")
async def gdrive_disconnect(body: WorkspaceBody, principal: PrincipalDep) -> dict:
    user, context, session = principal
    await require_member(session, workspace_id=body.workspace_id, user_id=user["id"])
    c = await gdrive.disconnect(session, workspace_id=body.workspace_id, actor_id=context.actor_id)
    if not c:
        return {"error": "not_connected"}
    return {"connector_id": c["id"], "status": c["status"]}


@router.get("/connectors/google-drive/status")
async def gdrive_status(workspace_id: str, principal: PrincipalDep) -> dict:
    user, _, session = principal
    await require_member(session, workspace_id=workspace_id, user_id=user["id"])
    return await gdrive.status(session, workspace_id=workspace_id)


@router.post("/connectors/google-drive/sync")
async def gdrive_sync(body: WorkspaceBody, principal: PrincipalDep) -> dict:
    user, context, session = principal
    await require_member(session, workspace_id=body.workspace_id, user_id=user["id"])
    return await gdrive.sync(session, workspace_id=body.workspace_id, actor_id=context.actor_id)


@router.get("/connectors/google-drive/files")
async def gdrive_files(workspace_id: str, principal: PrincipalDep) -> dict:
    user, _, session = principal
    await require_member(session, workspace_id=workspace_id, user_id=user["id"])
    c = await repo.get_connector_by_provider(session, workspace_id=workspace_id,
                                              provider=gdrive.PROVIDER)
    cid = c["id"] if c else None
    docs = await repo.list_documents(session, workspace_id=workspace_id, connector_id=cid)
    return {"files": docs}


@router.get("/connectors/google-drive/files/{document_id}")
async def gdrive_file(document_id: str, workspace_id: str, principal: PrincipalDep) -> dict:
    user, _, session = principal
    await require_member(session, workspace_id=workspace_id, user_id=user["id"])
    doc = await repo.get_document(session, document_id)
    if not doc or doc["workspace_id"] != workspace_id:
        return {"error": "not_found"}
    return doc


@router.get("/connectors/google-drive/files/{document_id}/permissions")
async def gdrive_file_permissions(document_id: str, workspace_id: str,
                                  principal: PrincipalDep) -> dict:
    user, _, session = principal
    await require_member(session, workspace_id=workspace_id, user_id=user["id"])
    doc = await repo.get_document(session, document_id)
    if not doc or doc["workspace_id"] != workspace_id:
        return {"error": "not_found"}
    return {"permissions": await repo.list_document_permissions(session, document_id)}


@router.get("/connectors/google-drive/audit")
async def gdrive_audit(workspace_id: str, principal: PrincipalDep, limit: int = 100) -> dict:
    user, _, session = principal
    await require_member(session, workspace_id=workspace_id, user_id=user["id"])
    events = await repo.list_audit(session, workspace_id=workspace_id, limit=limit)
    return {"events": [e for e in events if e.get("source") == gdrive.PROVIDER]}


# ---- ingestion processing (operator-triggered drain) -------------------

@router.post("/ingestion/process")
async def process_ingestion(body: WorkspaceBody, principal: PrincipalDep) -> dict:
    user, _, session = principal
    await require_member(session, workspace_id=body.workspace_id, user_id=user["id"])
    return await worker.process_pending(session, workspace_id=body.workspace_id)


# ---- generic connector status (declared last: {connector_id} must not shadow
# the specific /connectors/google-drive/* routes above) ------------------

@router.get("/connectors/{connector_id}/status")
async def connector_status(connector_id: str, workspace_id: str, principal: PrincipalDep) -> dict:
    user, _, session = principal
    await require_member(session, workspace_id=workspace_id, user_id=user["id"])
    c = await repo.get_connector(session, connector_id)
    if not c or c["workspace_id"] != workspace_id:
        return {"error": "not_found"}
    return {"connector_id": connector_id, "provider": c["provider"], "status": c["status"]}
