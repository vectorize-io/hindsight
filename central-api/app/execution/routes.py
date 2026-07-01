"""Execution ledger API routes."""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.context import ContextDep
from app.db.engine import get_session
from app.execution.ledger import (
    get_execution,
    get_lineage,
    list_executions,
    record_execution,
)

router = APIRouter(prefix="/api/executions", tags=["executions"])


@router.post("/record")
async def record_run(
    context: ContextDep,
    session: Annotated[AsyncSession, Depends(get_session)],
    action_type: str,
    target: str,
    agent_id: str,
    agent_role: str,
    risk_level: str,
    params: dict,
) -> dict:
    """Record a proposed execution in the ledger.
    
    Args:
        context: Request context (tenant_id, user_id, roles)
        session: DB session
        action_type: Type of action
        target: Resource identifier
        agent_id: Agent that proposed
        agent_role: Agent's role
        risk_level: Risk level
        params: Action parameters
    
    Returns:
        {execution_id, status}
    """
    exec_id = await record_execution(
        session,
        tenant_id=context.tenant_id,
        action_type=action_type,
        target=target,
        agent_id=agent_id,
        agent_role=agent_role,
        risk_level=risk_level,
        params=params,
        status="staged",
    )
    
    return {"execution_id": exec_id, "status": "staged"}


@router.get("/history")
async def get_history(
    context: ContextDep,
    session: Annotated[AsyncSession, Depends(get_session)],
    status: Annotated[Optional[str], Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    """Get execution history for tenant.
    
    Args:
        context: Request context
        session: DB session
        status: Optional status filter
        limit: Pagination limit
        offset: Pagination offset
    
    Returns:
        {executions, total}
    """
    executions = await list_executions(
        session,
        tenant_id=context.tenant_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    
    return {
        "executions": executions,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{execution_id}")
async def get_execution_detail(
    context: ContextDep,
    session: Annotated[AsyncSession, Depends(get_session)],
    execution_id: str,
) -> dict:
    """Get execution record by ID.
    
    Args:
        context: Request context
        session: DB session
        execution_id: Execution ID
    
    Returns:
        Execution record
    
    Raises:
        HTTPException: 404 if not found or wrong tenant
    """
    execution = await get_execution(
        session,
        execution_id=execution_id,
        tenant_id=context.tenant_id,
    )
    
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    return execution


@router.get("/{execution_id}/lineage")
async def get_execution_lineage(
    context: ContextDep,
    session: Annotated[AsyncSession, Depends(get_session)],
    execution_id: str,
) -> dict:
    """Get lineage (parent/child relationships) for execution.
    
    Args:
        context: Request context
        session: DB session
        execution_id: Execution ID
    
    Returns:
        {lineage_records}
    """
    # Verify execution belongs to tenant
    execution = await get_execution(
        session,
        execution_id=execution_id,
        tenant_id=context.tenant_id,
    )
    
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    lineage = await get_lineage(session, execution_id=execution_id)
    
    return {"lineage": lineage}
