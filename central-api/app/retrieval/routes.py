"""Retrieval routes — code/doc search."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.auth.context import ContextDep
from app.db.engine import get_session
from app.retrieval.service import create_retrieval_service
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.post("/search")
async def search(
    session: Annotated[AsyncSession, Depends(get_session)],
    query: str,
    mode: str = "semantic",
    limit: int = 10,
    context: ContextDep = None,
) -> dict:
    """Search code and documents.
    
    Modes: semantic, keyword, hybrid
    """
    tenant_id = context.tenant_id if context else "default"
    try:
        svc = await create_retrieval_service(tenant_id, session)
        results = await svc.search(query, mode, limit)
        return {
            "query": query,
            "mode": mode,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/code")
async def index_code(
    session: Annotated[AsyncSession, Depends(get_session)],
    path: str,
    content: str,
    language: str = "python",
    context: ContextDep = None,
) -> dict:
    """Index code file."""
    tenant_id = context.tenant_id if context else "default"
    try:
        svc = await create_retrieval_service(tenant_id, session)
        result = await svc.index_code(path, content, language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/document")
async def index_document(
    session: Annotated[AsyncSession, Depends(get_session)],
    path: str,
    content: str,
    context: ContextDep = None,
) -> dict:
    """Index markdown/text document."""
    tenant_id = context.tenant_id if context else "default"
    try:
        svc = await create_retrieval_service(tenant_id, session)
        result = await svc.index_document(path, content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
