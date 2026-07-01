"""Repository helpers — the only place that builds SQL for control-plane tables.

Plain async functions over an ``AsyncSession`` (Core ``insert``/``select``/
``update``). Routers and services call these; they never write SQL inline.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import tables as t
from app.db.ids import new_id, utcnow


async def _insert(session: AsyncSession, table: sa.Table, **values: Any) -> dict:
    values.setdefault("id", new_id())
    now = utcnow()
    for col in ("created_at", "updated_at", "snapshot_at"):
        if col in table.c and col not in values:
            values[col] = now
    await session.execute(sa.insert(table).values(**values))
    result = await session.execute(sa.select(table).where(table.c.id == values["id"]))
    return dict(result.mappings().one())


async def _all(session: AsyncSession, stmt: sa.Select) -> list[dict]:
    return [dict(r) for r in (await session.execute(stmt)).mappings().all()]


async def _one(session: AsyncSession, stmt: sa.Select) -> dict | None:
    row = (await session.execute(stmt)).mappings().first()
    return dict(row) if row else None


# ---- users / workspaces -------------------------------------------------

async def create_user(session: AsyncSession, *, email: str, display_name: str | None = None,
                      is_operator: bool = False) -> dict:
    return await _insert(session, t.users, email=email, display_name=display_name,
                         is_operator=is_operator)


async def get_user(session: AsyncSession, user_id: str) -> dict | None:
    return await _one(session, sa.select(t.users).where(t.users.c.id == user_id))


async def create_workspace(session: AsyncSession, *, name: str, owner_id: str) -> dict:
    ws = await _insert(session, t.workspaces, name=name, owner_id=owner_id)
    await _insert(session, t.workspace_members, workspace_id=ws["id"], user_id=owner_id,
                  role="owner")
    return ws


async def list_workspaces_for_user(session: AsyncSession, user_id: str) -> list[dict]:
    stmt = (
        sa.select(t.workspaces)
        .join(t.workspace_members, t.workspace_members.c.workspace_id == t.workspaces.c.id)
        .where(t.workspace_members.c.user_id == user_id)
        .order_by(t.workspaces.c.created_at)
    )
    return await _all(session, stmt)


async def list_all_workspaces(session: AsyncSession) -> list[dict]:
    """List all workspaces (for dashboard)."""
    return await _all(session, sa.select(t.workspaces).order_by(
        t.workspaces.c.created_at.desc()))


async def is_workspace_member(session: AsyncSession, *, workspace_id: str, user_id: str) -> bool:
    stmt = sa.select(t.workspace_members.c.id).where(
        t.workspace_members.c.workspace_id == workspace_id,
        t.workspace_members.c.user_id == user_id,
    )
    return (await session.execute(stmt)).first() is not None


# ---- connectors ---------------------------------------------------------

async def upsert_connector(session: AsyncSession, *, workspace_id: str, provider: str,
                           **fields: Any) -> dict:
    existing = await _one(session, sa.select(t.source_connectors).where(
        t.source_connectors.c.workspace_id == workspace_id,
        t.source_connectors.c.provider == provider,
    ))
    if existing:
        fields["updated_at"] = utcnow()
        await session.execute(sa.update(t.source_connectors)
                              .where(t.source_connectors.c.id == existing["id"]).values(**fields))
        return await get_connector(session, existing["id"])  # type: ignore[return-value]
    return await _insert(session, t.source_connectors, workspace_id=workspace_id,
                         provider=provider, **fields)


async def get_connector(session: AsyncSession, connector_id: str) -> dict | None:
    return await _one(session, sa.select(t.source_connectors).where(
        t.source_connectors.c.id == connector_id))


async def get_connector_by_provider(session: AsyncSession, *, workspace_id: str,
                                     provider: str) -> dict | None:
    return await _one(session, sa.select(t.source_connectors).where(
        t.source_connectors.c.workspace_id == workspace_id,
        t.source_connectors.c.provider == provider,
    ))


async def list_connectors(session: AsyncSession, workspace_id: str) -> list[dict]:
    return await _all(session, sa.select(t.source_connectors).where(
        t.source_connectors.c.workspace_id == workspace_id))


async def list_all_connectors(session: AsyncSession) -> list[dict]:
    """List all connectors across all workspaces (for dashboard)."""
    return await _all(session, sa.select(t.source_connectors).order_by(
        t.source_connectors.c.created_at.desc()))


async def set_connector_status(session: AsyncSession, connector_id: str, status: str) -> None:
    await session.execute(sa.update(t.source_connectors)
                          .where(t.source_connectors.c.id == connector_id)
                          .values(status=status, updated_at=utcnow()))


# ---- source documents ---------------------------------------------------

async def upsert_document(session: AsyncSession, *, connector_id: str, external_id: str,
                          **fields: Any) -> dict:
    existing = await _one(session, sa.select(t.source_documents).where(
        t.source_documents.c.connector_id == connector_id,
        t.source_documents.c.external_id == external_id,
    ))
    if existing:
        fields["updated_at"] = utcnow()
        await session.execute(sa.update(t.source_documents)
                              .where(t.source_documents.c.id == existing["id"]).values(**fields))
        return await get_document(session, existing["id"])  # type: ignore[return-value]
    return await _insert(session, t.source_documents, connector_id=connector_id,
                         external_id=external_id, **fields)


async def get_document(session: AsyncSession, document_id: str) -> dict | None:
    return await _one(session, sa.select(t.source_documents).where(
        t.source_documents.c.id == document_id))


async def list_documents(session: AsyncSession, *, workspace_id: str,
                         connector_id: str | None = None) -> list[dict]:
    stmt = sa.select(t.source_documents).where(t.source_documents.c.workspace_id == workspace_id)
    if connector_id:
        stmt = stmt.where(t.source_documents.c.connector_id == connector_id)
    return await _all(session, stmt.order_by(t.source_documents.c.created_at.desc()))


async def set_document_enabled(session: AsyncSession, document_id: str, enabled: bool) -> None:
    await session.execute(sa.update(t.source_documents)
                          .where(t.source_documents.c.id == document_id)
                          .values(enabled=enabled, updated_at=utcnow()))


# ---- permissions --------------------------------------------------------

async def replace_document_permissions(session: AsyncSession, document_id: str,
                                        perms: list[dict]) -> int:
    await session.execute(sa.delete(t.source_document_permissions).where(
        t.source_document_permissions.c.document_id == document_id))
    for p in perms:
        await _insert(session, t.source_document_permissions, document_id=document_id, **p)
    return len(perms)


async def list_document_permissions(session: AsyncSession, document_id: str) -> list[dict]:
    return await _all(session, sa.select(t.source_document_permissions).where(
        t.source_document_permissions.c.document_id == document_id))


# ---- ingestion jobs -----------------------------------------------------

async def create_job(session: AsyncSession, **fields: Any) -> dict:
    return await _insert(session, t.source_ingestion_jobs, **fields)


async def update_job_status(session: AsyncSession, job_id: str, status: str,
                            error: str | None = None) -> None:
    await session.execute(sa.update(t.source_ingestion_jobs)
                          .where(t.source_ingestion_jobs.c.id == job_id)
                          .values(status=status, error=error, updated_at=utcnow()))


async def list_jobs(session: AsyncSession, *, workspace_id: str,
                    status: str | None = None) -> list[dict]:
    stmt = sa.select(t.source_ingestion_jobs).where(
        t.source_ingestion_jobs.c.workspace_id == workspace_id)
    if status:
        stmt = stmt.where(t.source_ingestion_jobs.c.status == status)
    return await _all(session, stmt.order_by(t.source_ingestion_jobs.c.created_at.desc()))


async def list_all_jobs(session: AsyncSession, *, status: str | None = None,
                        limit: int = 100) -> list[dict]:
    """List ingestion jobs across all workspaces (for dashboard)."""
    stmt = sa.select(t.source_ingestion_jobs)
    if status:
        stmt = stmt.where(t.source_ingestion_jobs.c.status == status)
    return await _all(session, stmt.order_by(
        t.source_ingestion_jobs.c.created_at.desc()).limit(limit))


async def get_job(session: AsyncSession, job_id: str) -> dict | None:
    return await _one(session, sa.select(t.source_ingestion_jobs).where(
        t.source_ingestion_jobs.c.id == job_id))


# ---- sync state ---------------------------------------------------------

async def upsert_sync_state(session: AsyncSession, *, connector_id: str, **fields: Any) -> dict:
    existing = await _one(session, sa.select(t.source_sync_state).where(
        t.source_sync_state.c.connector_id == connector_id))
    if existing:
        fields["updated_at"] = utcnow()
        await session.execute(sa.update(t.source_sync_state)
                              .where(t.source_sync_state.c.id == existing["id"]).values(**fields))
        return await _one(session, sa.select(t.source_sync_state).where(  # type: ignore[return-value]
            t.source_sync_state.c.id == existing["id"]))
    return await _insert(session, t.source_sync_state, connector_id=connector_id, **fields)


# ---- audit / agent activity / approvals ---------------------------------

async def write_audit(session: AsyncSession, *, action: str, actor_id: str | None = None,
                      workspace_id: str | None = None, source: str | None = None,
                      source_file_id: str | None = None, status: str = "success",
                      metadata: dict | None = None) -> dict:
    return await _insert(session, t.source_audit_events, action=action, actor_id=actor_id,
                         workspace_id=workspace_id, source=source, source_file_id=source_file_id,
                         status=status, metadata=metadata or {})


async def list_audit(session: AsyncSession, *, workspace_id: str | None = None,
                     source_file_id: str | None = None, limit: int = 100) -> list[dict]:
    stmt = sa.select(t.source_audit_events)
    if workspace_id:
        stmt = stmt.where(t.source_audit_events.c.workspace_id == workspace_id)
    if source_file_id:
        stmt = stmt.where(t.source_audit_events.c.source_file_id == source_file_id)
    stmt = stmt.order_by(t.source_audit_events.c.created_at.desc()).limit(limit)
    return await _all(session, stmt)


async def record_agent_activity(session: AsyncSession, *, agent_id: str, requested_action: str,
                                 decision: str, workspace_id: str | None = None,
                                 target_resource: str | None = None, reason: str | None = None,
                                 metadata: dict | None = None) -> dict:
    return await _insert(session, t.agent_activity, agent_id=agent_id,
                         requested_action=requested_action, decision=decision,
                         workspace_id=workspace_id, target_resource=target_resource,
                         reason=reason, metadata=metadata or {})


async def list_agent_activity(session: AsyncSession, *, workspace_id: str | None = None,
                              limit: int = 100) -> list[dict]:
    stmt = sa.select(t.agent_activity)
    if workspace_id:
        stmt = stmt.where(t.agent_activity.c.workspace_id == workspace_id)
    return await _all(session, stmt.order_by(t.agent_activity.c.created_at.desc()).limit(limit))


async def create_approval(session: AsyncSession, *, requested_by: str, action: str,
                          workspace_id: str | None = None, target_resource: str | None = None,
                          metadata: dict | None = None) -> dict:
    return await _insert(session, t.operator_approvals, requested_by=requested_by, action=action,
                         workspace_id=workspace_id, target_resource=target_resource,
                         metadata=metadata or {})
