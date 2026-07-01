"""Retrieval service — semantic + keyword search over code/docs."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession


class RetrievalService:
    """Search code and documents."""

    def __init__(self, session: AsyncSession, tenant_id: str):
        self.session = session
        self.tenant_id = tenant_id

    async def search(self, query: str, mode: str = "semantic", limit: int = 10) -> list[dict]:
        """Search code/docs.
        
        Modes:
        - semantic: vector similarity (requires embeddings)
        - keyword: full-text search (SQL LIKE)
        - hybrid: semantic + keyword ranked
        """
        if mode == "keyword":
            return await self._keyword_search(query, limit)
        elif mode == "semantic":
            return await self._semantic_search(query, limit)
        else:  # hybrid
            semantic = await self._semantic_search(query, limit // 2)
            keyword = await self._keyword_search(query, limit // 2)
            return semantic + keyword

    async def _semantic_search(self, query: str, limit: int) -> list[dict]:
        """Vector similarity search (stub)."""
        return []

    async def _keyword_search(self, query: str, limit: int) -> list[dict]:
        """Full-text search (stub)."""
        return []

    async def index_code(self, path: str, content: str, language: str) -> dict:
        """Index code file."""
        return {
            "path": path,
            "language": language,
            "status": "indexed",
        }

    async def index_document(self, path: str, content: str) -> dict:
        """Index markdown/text document."""
        return {
            "path": path,
            "status": "indexed",
        }


async def create_retrieval_service(tenant_id: str, session: AsyncSession) -> RetrievalService:
    """Factory."""
    return RetrievalService(session, tenant_id)
