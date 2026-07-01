"""Adapter contract — every engine sits behind this interface.

The core never couples to a specific engine. Scaffold adapters return controlled
mocks (search) or raise NotImplementedError (writes on read-only engines). No real
engine calls are made yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class AdapterHit(BaseModel):
    memory_id: str
    backend: str
    content: str
    memory_type: str = "conversation"
    score: float | None = None
    citation: dict | None = None
    metadata: dict = Field(default_factory=dict)


class EngineHealth(BaseModel):
    backend: str
    status: str  # ok | degraded | stub | down
    detail: dict = Field(default_factory=dict)


class BaseMemoryAdapter(ABC):
    """Common contract for memory/retrieval engines."""

    backend: str = "base"
    read_only: bool = False

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        k: int = 10,
    ) -> list[AdapterHit]:
        ...

    async def store(
        self,
        content: str,
        *,
        memory_type: str = "conversation",
        workspace_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> AdapterHit:
        raise NotImplementedError(f"{self.backend} adapter does not support store()")

    async def update(
        self, memory_id: str, content: str, metadata: dict | None = None
    ) -> AdapterHit:
        raise NotImplementedError(f"{self.backend} adapter does not support update()")

    async def delete(self, memory_id: str) -> None:
        raise NotImplementedError(f"{self.backend} adapter does not support delete()")

    async def export(self, workspace_id: str | None = None) -> list[dict]:
        raise NotImplementedError(f"{self.backend} adapter does not support export()")

    async def health(self) -> EngineHealth:
        return EngineHealth(backend=self.backend, status="stub")
