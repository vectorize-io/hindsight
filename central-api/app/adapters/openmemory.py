"""OpenMemoryAdapter — agent memory / MCP reference layer (stub).

Target: OpenMemory `/memory/*` (HSG sectors, temporal facts, embeddings) and MCP
memory tools. Scaffold: mocks.
"""

from __future__ import annotations

import uuid

from app.adapters.base import AdapterHit, BaseMemoryAdapter, EngineHealth


class OpenMemoryAdapter(BaseMemoryAdapter):
    backend = "openmemory"

    async def search(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        k: int = 10,
    ) -> list[AdapterHit]:
        return []  # stub: target = POST /memory/query (vector search)

    async def store(
        self,
        content: str,
        *,
        memory_type: str = "agent",
        workspace_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> AdapterHit:
        return AdapterHit(
            memory_id=str(uuid.uuid4()),
            backend=self.backend,
            content=content,
            memory_type=memory_type,
            metadata={"stub": True, **(metadata or {})},
        )

    async def health(self) -> EngineHealth:
        return EngineHealth(backend=self.backend, status="stub", detail={"api": "/memory/*"})
