"""Execution ledger service layer — persist action records and lineage."""

import json
from datetime import datetime, timezone
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.ids import new_id
from app.db.tables import execution_ledger, execution_lineage


async def record_execution(
    session: AsyncSession,
    *,
    tenant_id: str,
    action_type: str,
    target: str,
    agent_id: str,
    agent_role: str,
    risk_level: str,
    params: dict,
    status: str = "staged",
) -> str:
    """Record a proposed action in the execution ledger.
    
    Args:
        session: AsyncSession
        tenant_id: Workspace/tenant ID
        action_type: Type of action (docker_deploy, update_env, etc)
        target: Resource identifier (service name, file path, etc)
        agent_id: Agent that proposed
        agent_role: Agent's role (devops-agent, dev-agent, etc)
        risk_level: Risk classification (low, medium, high, critical)
        params: Action parameters (secrets redacted)
        status: Initial status (staged, approved, etc)
    
    Returns:
        Execution ID
    """
    exec_id = new_id()
    
    stmt = execution_ledger.insert().values(
        id=exec_id,
        tenant_id=tenant_id,
        action_type=action_type,
        target=target,
        agent_id=agent_id,
        agent_role=agent_role,
        risk_level=risk_level,
        status=status,
        params=params,
        created_at=datetime.now(timezone.utc),
    )
    
    await session.execute(stmt)
    await session.commit()
    
    return exec_id


async def approve_execution(
    session: AsyncSession,
    *,
    execution_id: str,
    tenant_id: str,
    approver_id: str,
    approval_note: Optional[str] = None,
) -> bool:
    """Mark execution as approved by operator.
    
    Args:
        session: AsyncSession
        execution_id: ID to approve
        tenant_id: Tenant scope check
        approver_id: Operator who approved
        approval_note: Optional approval note
    
    Returns:
        Success
    """
    stmt = (
        execution_ledger.update()
        .where(
            sa.and_(
                execution_ledger.c.id == execution_id,
                execution_ledger.c.tenant_id == tenant_id,
            )
        )
        .values(
            status="approved",
            approver_id=approver_id,
            approval_note=approval_note,
            approved_at=datetime.now(timezone.utc),
        )
    )
    
    result = await session.execute(stmt)
    await session.commit()
    
    return result.rowcount > 0


async def record_execution_result(
    session: AsyncSession,
    *,
    execution_id: str,
    tenant_id: str,
    status: str,
    result: Optional[dict] = None,
    error_message: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> bool:
    """Record execution completion (success or failure).
    
    Called after adapter execution completes to persist outcome.
    
    Args:
        session: AsyncSession
        execution_id: ID to update
        tenant_id: Tenant scope check
        status: Final status (completed, failed)
        result: Execution result (exit code, output, etc) — from adapter.execute()
        error_message: Error if failed
        duration_seconds: Execution duration
    
    Returns:
        Success
    
    Example call site:
        result = await adapter.execute(execution)
        await record_execution_result(
            session,
            execution_id=exec_id,
            tenant_id=tenant_id,
            status="completed",
            result=result,  # {exit_code, output, result}
            duration_seconds=(time.time() - start),
        )
    """
    stmt = (
        execution_ledger.update()
        .where(
            sa.and_(
                execution_ledger.c.id == execution_id,
                execution_ledger.c.tenant_id == tenant_id,
            )
        )
        .values(
            status=status,
            result=result,
            error_message=error_message,
            duration_seconds=duration_seconds,
            started_at=execution_ledger.c.started_at or datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
    )
    
    result_set = await session.execute(stmt)
    await session.commit()
    
    return result_set.rowcount > 0


async def link_executions(
    session: AsyncSession,
    *,
    parent_id: str,
    child_id: str,
    relationship: str,
    context: Optional[dict] = None,
) -> str:
    """Record parent-child relationship between executions (lineage).
    
    Args:
        session: AsyncSession
        parent_id: Parent execution ID
        child_id: Child execution ID
        relationship: Type (triggered_by, depends_on, rollback_of)
        context: Additional relationship context
    
    Returns:
        Lineage record ID
    """
    lineage_id = new_id()
    
    stmt = execution_lineage.insert().values(
        id=lineage_id,
        parent_execution_id=parent_id,
        child_execution_id=child_id,
        relationship=relationship,
        context=context,
        created_at=datetime.now(timezone.utc),
    )
    
    await session.execute(stmt)
    await session.commit()
    
    return lineage_id


async def get_execution(
    session: AsyncSession,
    *,
    execution_id: str,
    tenant_id: str,
) -> Optional[dict]:
    """Retrieve execution record.
    
    Args:
        session: AsyncSession
        execution_id: ID to fetch
        tenant_id: Tenant scope check
    
    Returns:
        Execution record or None
    """
    stmt = (
        sa.select(execution_ledger).where(
            sa.and_(
                execution_ledger.c.id == execution_id,
                execution_ledger.c.tenant_id == tenant_id,
            )
        )
    )
    
    result = await session.execute(stmt)
    row = result.one_or_none()
    
    if not row:
        return None
    
    return dict(row._mapping)


async def list_executions(
    session: AsyncSession,
    *,
    tenant_id: str,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List execution records for tenant.
    
    Args:
        session: AsyncSession
        tenant_id: Tenant scope
        status: Optional status filter
        limit: Pagination limit
        offset: Pagination offset
    
    Returns:
        List of execution records
    """
    stmt = sa.select(execution_ledger).where(
        execution_ledger.c.tenant_id == tenant_id
    )
    
    if status:
        stmt = stmt.where(execution_ledger.c.status == status)
    
    stmt = (
        stmt.order_by(execution_ledger.c.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    
    result = await session.execute(stmt)
    rows = result.fetchall()
    
    return [dict(row._mapping) for row in rows]


async def get_lineage(
    session: AsyncSession,
    *,
    execution_id: str,
) -> list[dict]:
    """Get lineage records for an execution (both parent and child links).
    
    Args:
        session: AsyncSession
        execution_id: Execution ID to trace
    
    Returns:
        List of lineage records
    """
    stmt = sa.select(execution_lineage).where(
        sa.or_(
            execution_lineage.c.parent_execution_id == execution_id,
            execution_lineage.c.child_execution_id == execution_id,
        )
    )
    
    result = await session.execute(stmt)
    rows = result.fetchall()
    
    return [dict(row._mapping) for row in rows]
