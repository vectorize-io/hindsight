"""InternalAdapter — native memory store (stub).

Target: delegate embedding/vectors to the OpenMemory engine while the control
plane governs/maps/audits (see the Node prototype). Scaffold: controlled mocks.
"""

from __future__ import annotations

import uuid

from app.adapters.base import AdapterHit, BaseMemoryAdapter, EngineHealth


class InternalAdapter(BaseMemoryAdapter):
    backend = "internal"

    async def search(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        k: int = 10,
    ) -> list[AdapterHit]:
        return []  # stub: no engine call yet

    async def store(
        self,
        content: str,
        *,
        memory_type: str = "conversation",
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

    async def update(
        self, memory_id: str, content: str, metadata: dict | None = None
    ) -> AdapterHit:
        return AdapterHit(
            memory_id=memory_id, backend=self.backend, content=content,
            metadata={"stub": True, **(metadata or {})},
        )

    async def delete(self, memory_id: str) -> None:
        return None

    async def export(self, workspace_id: str | None = None) -> list[dict]:
        return []

    async def health(self) -> EngineHealth:
        return EngineHealth(
            backend=self.backend, status="stub", detail={"engine": "openmemory(target)"}
        )
