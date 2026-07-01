"""CodeRAGAdapter — code/document retrieval backend (stub, read-only).

Target: CodeRAG `GET /search` (symbol-aware, hybrid+RRF, file:line citations).
Read-mostly: the repo is the source of truth; writes are index lifecycle, not
content writes. Scaffold: mocks.
"""

from __future__ import annotations

from app.adapters.base import AdapterHit, BaseMemoryAdapter, EngineHealth


class CodeRAGAdapter(BaseMemoryAdapter):
    backend = "coderag"
    read_only = True

    async def search(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        k: int = 10,
    ) -> list[AdapterHit]:
        return []  # stub: target = GET /search?q=&k= → file:line citations

    async def health(self) -> EngineHealth:
        return EngineHealth(backend=self.backend, status="stub", detail={"api": "GET /search"})
