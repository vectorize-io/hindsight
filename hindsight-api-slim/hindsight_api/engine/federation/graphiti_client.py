"""Thin async HTTP client for the Graphiti world-graph service.

Implements the two endpoints the federation needs:

* ``POST /triplet`` — write a single (source_node, target_node, edge) tuple
  to a named ``group_id``. Returns the resolved edge plus any edges the
  write invalidated (deep-dive 2 §2.1 / deep-dive 4 §1.1). The forwarder
  uses ``invalidated_edges`` to drive channel A of C4 (re-consolidation of
  the local memory the invalidation points at via ``source_uri``).

* ``POST /search`` — basic mixed retrieval over a ``group_id`` set. Returns
  edges (facts) with their ``valid_at`` / ``invalid_at`` double timeline.
  Used by C3 ``search_world_graph`` (deep-dive 3 §2.1 — out of scope for
  this PR; the client method is here so the same breaker / timeout
  configuration is shared between the forwarder and the future reflect
  tool, per deep-dive 2 §2.6).

Timeout and circuit-breaker defaults match the C-track config (deep-dive 2
§2.6): 2s timeout, 5 failures → 30s open. Concurrent calls are bounded by
an asyncio.Semaphore (``max_concurrent``) so a reflect storm cannot drown
the forwarder (or vice versa) — the same per-op semaphore pattern
``llm_wrapper._build_per_op_semaphores`` uses.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import httpx

from .circuit_breaker import CircuitBreaker, CircuitOpenError

logger = logging.getLogger(__name__)


@dataclass
class NodePayload:
    uuid: UUID | None
    name: str
    group_id: str


@dataclass
class EdgePayload:
    uuid: UUID
    name: str  # SCREAMING_SNAKE_CASE predicate (C2 vocabulary)
    fact: str
    valid_at: str | None
    invalid_at: str | None
    group_id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TripletRequest:
    source_node: NodePayload
    target_node: NodePayload
    edge: EdgePayload


@dataclass
class EdgeResult:
    uuid: UUID
    name: str
    fact: str
    valid_at: str | None
    invalid_at: str | None
    source_uri: str | None  # from attributes, exposed for channel A
    group_id: str | None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeResult:
    uuid: UUID
    name: str
    group_id: str | None


@dataclass
class AddTripletResults:
    edges: list[EdgeResult]
    nodes: list[NodeResult]

    @property
    def invalidated_edges(self) -> list[EdgeResult]:
        return [e for e in self.edges if e.invalid_at is not None]


@dataclass
class FactResult:
    uuid: UUID
    name: str
    fact: str
    valid_at: str | None
    invalid_at: str | None
    created_at: str | None
    expired_at: str | None


class GraphitiClientError(Exception):
    """Wraps any non-recoverable HTTP / serialization failure."""


class GraphitiClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_ms: int = 2000,
        max_concurrent: int = 8,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_ms / 1000.0
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout_s),
            headers=self._auth_headers(api_key),
        )
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            reset_timeout_seconds=reset_timeout_seconds,
        )

    @staticmethod
    def _auth_headers(api_key: str | None) -> dict[str, str]:
        if not api_key:
            return {"Content-Type": "application/json"}
        return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "GraphitiClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def add_triplet(self, payload: TripletRequest) -> AddTripletResults:
        body = _triplet_to_json(payload)
        try:
            data = await self._breaker.call(lambda: self._post_json("/triplet", body))
        except CircuitOpenError:
            raise
        except httpx.HTTPError as e:
            raise GraphitiClientError(f"add_triplet HTTP error: {e}") from e
        try:
            return _parse_add_triplet_results(data)
        except Exception as e:
            raise GraphitiClientError(f"add_triplet response parse error: {e}") from e

    async def search(
        self,
        group_ids: list[str],
        query: str,
        max_facts: int = 10,
    ) -> list[FactResult]:
        body = {"group_ids": group_ids, "query": query, "max_facts": max_facts}
        try:
            data = await self._breaker.call(lambda: self._post_json("/search", body))
        except CircuitOpenError:
            raise
        except httpx.HTTPError as e:
            raise GraphitiClientError(f"search HTTP error: {e}") from e
        try:
            return _parse_fact_results(data)
        except Exception as e:
            raise GraphitiClientError(f"search response parse error: {e}") from e

    async def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        async with self._semaphore:
            resp = await self._client.post(f"{self._base_url}{path}", json=body)
            resp.raise_for_status()
            return resp.json()


# ---------------------------------------------------------------------------
# Serialization helpers — kept private to the client to avoid leaking the
# wire shape. Validated as part of the ``hs_llm_core`` stub-integration tests.
# ---------------------------------------------------------------------------


def _triplet_to_json(req: TripletRequest) -> dict[str, Any]:
    def node_dict(n: NodePayload) -> dict[str, Any]:
        return {
            "uuid": str(n.uuid) if n.uuid else None,
            "name": n.name,
            "group_id": n.group_id,
        }

    def edge_dict(e: EdgePayload) -> dict[str, Any]:
        return {
            "uuid": str(e.uuid),
            "name": e.name,
            "fact": e.fact,
            "valid_at": e.valid_at,
            "invalid_at": e.invalid_at,
            "group_id": e.group_id,
            "attributes": e.attributes,
        }

    return {
        "source_node": node_dict(req.source_node),
        "target_node": node_dict(req.target_node),
        "edge": edge_dict(req.edge),
    }


def _parse_add_triplet_results(data: dict[str, Any]) -> AddTripletResults:
    edges_raw = data.get("edges") or []
    nodes_raw = data.get("nodes") or []
    edges = [
        EdgeResult(
            uuid=UUID(e["uuid"]),
            name=e.get("name", ""),
            fact=e.get("fact", ""),
            valid_at=e.get("valid_at"),
            invalid_at=e.get("invalid_at"),
            source_uri=(e.get("attributes") or {}).get("source_uri"),
            group_id=e.get("group_id"),
            attributes=e.get("attributes") or {},
        )
        for e in edges_raw
    ]
    nodes = [
        NodeResult(
            uuid=UUID(n["uuid"]),
            name=n.get("name", ""),
            group_id=n.get("group_id"),
        )
        for n in nodes_raw
    ]
    return AddTripletResults(edges=edges, nodes=nodes)


def _parse_fact_results(data: Any) -> list[FactResult]:
    items = data if isinstance(data, list) else (data.get("facts") or [])
    out: list[FactResult] = []
    for it in items:
        out.append(
            FactResult(
                uuid=UUID(it["uuid"]),
                name=it.get("name", ""),
                fact=it.get("fact", ""),
                valid_at=it.get("valid_at"),
                invalid_at=it.get("invalid_at"),
                created_at=it.get("created_at"),
                expired_at=it.get("expired_at"),
            )
        )
    return out
