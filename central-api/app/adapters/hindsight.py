"""HindsightAdapter — primary memory CRUD + retrieval engine.

Maps the canonical contract to Hindsight's REST API: hybrid semantic + full-text
search with native recall/reflect/retrieve operations. Full read-write support.

Key endpoints (Hindsight REST API at http://localhost:8888):
  GET  /v1/default/banks                   — list banks
  GET  /v1/default/banks/{bank_id}         — get bank profile
  POST /v1/default/banks/{bank_id}/memories/recall   — hybrid search
  POST /v1/default/banks/{bank_id}/memories          — create memory (retain)
  GET  /v1/default/banks/{bank_id}/memories/{id}     — get single memory
  GET  /v1/default/banks/{bank_id}/memories/list     — paginated list
  PATCH /v1/default/banks/{bank_id}/memories/{id}    — update memory
  DELETE /v1/default/banks/{bank_id}/memories/{id}   — delete memory
  POST /v1/default/banks/{bank_id}/reflect           — reflect/synthesize
  GET  /v1/default/banks/{bank_id}/stats             — bank statistics
  GET  /v1/default/banks/{bank_id}/tags              — list tags
  GET  /v1/default/banks/{bank_id}/operations        — list operations

Type Mapping:
  - Central API types: conversation, fact, preference, instruction, feedback, decision
  - Hindsight API types: experience, world, observation (= fact_type)
  - conversion: conversation → experience with "conversation" tag
"""

from __future__ import annotations

from typing import Any

import httpx

from app.adapters.base import AdapterHit, BaseMemoryAdapter, EngineHealth
from app.config import settings


class HindsightAdapter(BaseMemoryAdapter):
    backend = "hindsight"
    read_only = False  # Full read-write support

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        bank_id: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = (base_url or settings.hindsight_url).rstrip("/")
        self._api_key = api_key if api_key is not None else settings.hindsight_api_key
        self._bank_id = bank_id or settings.hindsight_bank_id
        self._client = client  # injected in tests (httpx.MockTransport)

    @property
    def configured(self) -> bool:
        """Real calls are made only when a URL is present (or a client injected)."""
        return bool(self._base_url) or self._client is not None

    @property
    def _api_base(self) -> str:
        """Hindsight REST API base path with the default project/version prefix."""
        return f"{self._base_url}/v1/default/banks/{self._bank_id}"

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers.setdefault("Content-Type", "application/json")

        url = f"{self._base_url}{path}"

        if self._client is not None:
            return await self._client.request(method, url, headers=headers, **kwargs)
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.request(method, url, headers=headers, **kwargs)

    @staticmethod
    def _to_hit(result: dict[str, Any], score: float | None = None) -> AdapterHit:
        """Convert Hindsight REST result to AdapterHit."""
        memory_id = result.get("id", "")
        text = result.get("text", result.get("content", ""))
        fact_type = result.get("fact_type", result.get("type", "experience"))
        tags = result.get("tags", [])
        metadata = result.get("metadata", {})
        entities = result.get("entities", [])

        # Convert Hindsight fact_type to Central API memory_type
        type_map = {
            "experience": "conversation",
            "world": "fact",
            "observation": "observation",
        }
        memory_type = type_map.get(fact_type, "conversation")

        # Extract workspace from tags
        workspace = None
        for tag in tags if isinstance(tags, list) else []:
            if isinstance(tag, str) and tag.startswith("workspace:"):
                workspace = tag.split(":", 1)[1]
                break

        return AdapterHit(
            memory_id=f"hindsight:{memory_id}",
            backend="hindsight",
            content=text,
            memory_type=memory_type,
            score=score,
            citation={
                "backend": "hindsight",
                "memory_id": memory_id,
                "workspace": workspace,
                "bank_id": None,  # caller fills this in
            },
            metadata={
                "tags": tags,
                "workspace": workspace,
                "entities": entities,
                **metadata,
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
        """Hybrid semantic + full-text search via memories/recall."""
        if not self.configured:
            return []

        req: dict[str, Any] = {"query": query, "limit": k}

        if memory_type:
            # Map Central API type to Hindsight fact_type
            reverse_map = {
                "conversation": "experience",
                "fact": "world",
                "observation": "observation",
                "preference": "experience",
                "instruction": "experience",
                "feedback": "experience",
                "decision": "experience",
            }
            req["type"] = reverse_map.get(memory_type, "experience")

        if workspace_id and not req.get("tags"):
            req["tags"] = [f"workspace:{workspace_id}"]
        elif workspace_id:
            req["tags"].append(f"workspace:{workspace_id}")

        try:
            resp = await self._request("POST", f"/v1/default/banks/{self._bank_id}/memories/recall", json=req)
            resp.raise_for_status()
            data = resp.json()

            # Hindsight recall returns {results: [{id, text, type, ...}]}
            results = data.get("results", [])

            hits: list[AdapterHit] = []
            for item in results:
                score = item.get("score") or item.get("similarity_score")
                hits.append(self._to_hit(item, score=score))

            return hits

        except Exception as e:
            print(f"Hindsight search error: {e}")
            return []

    async def store(
        self,
        content: str,
        *,
        memory_type: str = "conversation",
        workspace_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> AdapterHit:
        """Store a new memory in Hindsight (retain)."""
        if not self.configured:
            raise NotImplementedError(f"{self.backend} is not configured")

        # Map memory_type to Hindsight fact_type
        forward_map = {
            "conversation": "experience",
            "fact": "world",
            "observation": "observation",
            "preference": "experience",
            "instruction": "experience",
            "feedback": "experience",
            "decision": "experience",
        }
        fact_type = forward_map.get(memory_type, "experience")

        all_tags = list(tags) if tags else []
        if workspace_id:
            all_tags.append(f"workspace:{workspace_id}")

        req: dict[str, Any] = {
            "items": [
                {
                    "content": content,
                    "type": fact_type,
                }
            ],
        }
        if all_tags:
            req["items"][0]["tags"] = all_tags
        if metadata:
            req["items"][0]["metadata"] = metadata

        try:
            resp = await self._request(
                "POST", f"/v1/default/banks/{self._bank_id}/memories", json=req
            )
            resp.raise_for_status()
            data = resp.json()

            # Hindsight returns {status: ..., operation_id: ..., ...}
            # Memory creation is async; return metadata about the submission
            operation_id = data.get("operation_id", "")

            return AdapterHit(
                memory_id=f"hindsight:pending:{operation_id}" if operation_id else "hindsight:pending",
                backend="hindsight",
                content=content,
                memory_type=memory_type,
                citation={"backend": "hindsight", "operation_id": operation_id},
                metadata={"tags": all_tags or [], "operation_id": operation_id},
            )

        except Exception as e:
            print(f"Hindsight store error: {e}")
            raise

    async def update(
        self, memory_id: str, content: str, metadata: dict | None = None
    ) -> AdapterHit:
        """Update an existing memory."""
        if not self.configured:
            raise NotImplementedError(f"{self.backend} is not configured")

        # Extract UUID from memory_id (format: "hindsight:<uuid>")
        uuid = memory_id.split(":", 1)[1] if ":" in memory_id else memory_id

        req: dict[str, Any] = {"text": content}
        if metadata:
            req["metadata"] = metadata

        try:
            resp = await self._request(
                "PATCH", f"/v1/default/banks/{self._bank_id}/memories/{uuid}", json=req
            )
            resp.raise_for_status()
            data = resp.json()
            result = data if isinstance(data, dict) and "id" in data else data.get("memory", {})

            return self._to_hit({
                "id": result.get("id", uuid),
                "text": content,
                "type": result.get("fact_type", "experience"),
            })
        except Exception as e:
            print(f"Hindsight update error: {e}")
            raise

    async def delete(self, memory_id: str) -> None:
        """Delete a memory from Hindsight."""
        if not self.configured:
            raise NotImplementedError(f"{self.backend} is not configured")

        uuid = memory_id.split(":", 1)[1] if ":" in memory_id else memory_id

        try:
            # Hindsight REST API uses bulk delete via DELETE /memories
            resp = await self._request(
                "DELETE",
                f"/v1/default/banks/{self._bank_id}/memories",
                json={"memory_ids": [uuid]},
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"Hindsight delete error: {e}")
            raise

    async def export(self, workspace_id: str | None = None) -> list[dict]:
        """Export all memories using paginated memories/list."""
        if not self.configured:
            return []

        memories: list[dict] = []
        page = 1
        page_size = 100

        while True:
            try:
                params = f"?limit={page_size}&offset={(page - 1) * page_size}"
                if workspace_id:
                    params += f"&tag=workspace%3A{workspace_id}"

                resp = await self._request(
                    "GET", f"/v1/default/banks/{self._bank_id}/memories/list{params}"
                )
                resp.raise_for_status()
                data = resp.json()

                rows = data.get("items", [])
                if not rows:
                    break

                memories.extend(rows)

                if len(rows) < page_size:
                    break

                page += 1
            except Exception as e:
                print(f"Hindsight export error (page {page}): {e}")
                break

        return memories

    async def health(self) -> EngineHealth:
        """Check Hindsight health via banks list + get_bank."""
        if not self.configured:
            return EngineHealth(
                backend=self.backend, status="stub", detail={"reason": "not_configured"}
            )

        try:
            # Check with bank profile endpoint
            resp = await self._request(
                "GET", f"/v1/default/banks/{self._bank_id}/profile", timeout=5.0
            )

            if resp.status_code == 200:
                data = resp.json()
                return EngineHealth(
                    backend=self.backend,
                    status="ok",
                    detail={
                        "bank_id": data.get("bank_id"),
                        "name": data.get("name"),
                        "fact_count": data.get("fact_count"),
                    },
                )
            else:
                return EngineHealth(
                    backend=self.backend,
                    status="degraded",
                    detail={"status_code": resp.status_code},
                )

        except Exception as exc:
            return EngineHealth(
                backend=self.backend, status="down", detail={"error": str(exc)}
            )
