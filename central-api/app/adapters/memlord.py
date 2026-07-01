"""MemlordAdapter — primary memory CRUD + retrieval engine.

Maps the canonical contract to MemLord's `/api-dev/*` REST surface (API-key
only): hybrid BM25+vector search with native RRF (`POST /api-dev/memories/search`,
returns `results[{id, content, memory_type, workspace_id, workspace, score}]`
where `score` is the rounded `rrf_score`), and paginated listing
(`POST /api-dev/memories`) used for export. MemLord's dedicated export
(`GET /api/workspaces/{id}/export`) is cookie/OAuth-only and unreachable via
the API key, so export is built by paginating the list endpoint instead.
Read-mostly in the federation: MemLord owns its own writes, so the controller
searches/lists it but never writes.

Network calls happen only when an API key is configured; otherwise the adapter
behaves as a safe stub (no connection attempts). A client can be injected for
mock-safe tests.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.adapters.base import AdapterHit, BaseMemoryAdapter, EngineHealth
from app.config import settings


class MemlordAdapter(BaseMemoryAdapter):
    backend = "memlord"
    read_only = True  # MemLord owns its own writes; the federation searches it
    export_page_size = 100  # rows per /api-dev/memories list page (server caps at 100)

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = (base_url or settings.memlord_url).rstrip("/")
        self._api_key = api_key if api_key is not None else settings.memlord_api_key
        self._client = client  # injected in tests (httpx.MockTransport)

    @property
    def configured(self) -> bool:
        """Real calls are made only when an API key is present (or a client injected)."""
        return bool(self._api_key) or self._client is not None

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        if self._client is not None:
            return await self._client.request(method, path, headers=headers, **kwargs)
        async with httpx.AsyncClient(base_url=self._base_url, timeout=10.0) as client:
            return await client.request(method, path, headers=headers, **kwargs)

    @staticmethod
    def _to_hit(row: dict[str, Any]) -> AdapterHit:
        rid = row.get("id")
        return AdapterHit(
            memory_id=f"memlord:{rid}",
            backend="memlord",
            content=row.get("content", ""),
            memory_type=row.get("memory_type") or "conversation",
            score=row.get("score"),  # MemLord returns the fused rrf_score
            citation={
                "backend": "memlord",
                "memory_id": str(rid) if rid is not None else None,
                "workspace": row.get("workspace"),
            },
            metadata={
                "workspace_id": row.get("workspace_id"),  # ≈ project scope
                "workspace": row.get("workspace"),
                "rrf_score": row.get("score"),
            },
        )

    async def search(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        k: int = 10,
    ) -> list[AdapterHit]:
        if not self.configured:
            return []
        resp = await self._request(
            "POST", "/api-dev/memories/search", json={"query": query, "limit": k}
        )
        resp.raise_for_status()
        rows = resp.json().get("results", [])
        return [self._to_hit(r) for r in rows]

    async def export(self, workspace_id: str | None = None) -> list[dict]:
        """Export memlord-resident memories via the API-key list surface.

        MemLord's *dedicated* export (``GET /api/workspaces/{id}/export``) is
        cookie/OAuth-protected and NOT exposed under ``/api-dev`` — so the
        service API key cannot reach it. But the API-key surface DOES expose
        paginated listing at ``POST /api-dev/memories`` (returns every memory
        the key can access: ``id, content, memory_type, workspace_id,
        created_at``). We paginate that to produce a full export.

        Note: that list endpoint does not yet honor server-side workspace
        filtering, so ``workspace_id`` is applied client-side here.
        """
        if not self.configured:
            return []
        out: list[dict] = []
        page = 1
        page_size = self.export_page_size
        max_pages = 1000  # safety bound against a misbehaving/looping backend
        while page <= max_pages:
            resp = await self._request(
                "POST", "/api-dev/memories", json={"page": page, "page_size": page_size}
            )
            resp.raise_for_status()
            rows = resp.json().get("memories", [])
            if not rows:
                break
            kept = rows
            if workspace_id is not None:
                kept = [r for r in rows if str(r.get("workspace_id")) == str(workspace_id)]
            out.extend(kept)
            if len(rows) < page_size:
                break  # short page → last page
            page += 1
        return out

    async def health(self) -> EngineHealth:
        if not self.configured:
            return EngineHealth(backend=self.backend, status="stub", detail={"api": "/api-dev"})
        try:
            resp = await self._request("GET", "/health")
            status = "ok" if resp.status_code == 200 else "degraded"
            return EngineHealth(
                backend=self.backend, status=status, detail={"code": resp.status_code}
            )
        except Exception as exc:  # noqa: BLE001
            return EngineHealth(backend=self.backend, status="down", detail={"error": str(exc)})
