"""Tests for the ``graphiti_forward`` worker (C1 + C4-A).

The worker is the drain side of the graphiti_outbox → Graphiti pipeline. The
tests cover:

* Deterministic UUIDs — same (group_id, name) or (memory_id, rel) keys produce
  the same UUID across calls. This is the foundation for at-least-once
  retries landing on the same edge.
* Transient-vs-permanent error classification — the row is rescheduled only
  on transient failures, never on permanent ones (4xx other than 408/429).
* Channel A (C4-A) — ``invalidated_edges`` whose ``source_uri`` matches the
  bank's ``hindsight://bank/{bank_id}/memory/`` prefix are remembered for
  end-of-drain consolidation submit.
* End-to-end happy path via a mock Graphiti transport — one outbox row →
  one ``add_triplet`` call → ``forwarded``/``relations`` counts go up →
  ``reschedule_graphiti_outbox_rows`` is **not** called.
* Transient error path — the same drain surfaces a reschedule, not a drop.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pytest

from hindsight_api.engine.federation.circuit_breaker import CircuitOpenError
from hindsight_api.engine.federation.graphiti_client import (
    GraphitiClient,
    GraphitiClientError,
)
from hindsight_api.engine.retain.graphiti_forward import (
    _deterministic_edge_uuid,
    _deterministic_node_uuid,
    _handle_invalidated_edges,
    _is_transient_error,
    _trigger_backflow_actions,
)

# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_deterministic_node_uuid_stable_across_calls():
    """Same (group_id, name) → same UUID regardless of caller or process."""
    a = _deterministic_node_uuid("agent-shared-001", "Alice")
    b = _deterministic_node_uuid("agent-shared-001", "ALICE")  # case-folded
    assert a == b


def test_deterministic_node_uuid_differs_by_group_or_name():
    base = _deterministic_node_uuid("g1", "Alice")
    assert _deterministic_node_uuid("g2", "Alice") != base
    assert _deterministic_node_uuid("g1", "Bob") != base


def test_deterministic_edge_uuid_is_stable_per_relation():
    memory_id = uuid4()
    rel = {"source_entity_index": 0, "target_entity_index": 1, "predicate": "WORKS_AT"}
    a = _deterministic_edge_uuid(memory_id, rel)
    b = _deterministic_edge_uuid(memory_id, rel)
    assert a == b
    # Predicate change → different edge (no merge by accident).
    rel2 = dict(rel, predicate="LIVES_IN")
    assert _deterministic_edge_uuid(memory_id, rel2) != a


def test_is_transient_error_circuit_open_is_transient():
    assert _is_transient_error(CircuitOpenError("open")) is True


def test_is_transient_error_5xx_message_is_transient():
    err = GraphitiClientError("add_triplet HTTP error: 503 Service Unavailable")
    assert _is_transient_error(err) is True


def test_is_transient_error_timeout_message_is_transient():
    err = GraphitiClientError("add_triplet HTTP error: timeout")
    assert _is_transient_error(err) is True


def test_is_transient_error_4xx_is_permanent():
    """4xx other than 408/429 are configuration errors — retrying won't help."""
    err = GraphitiClientError("400 Bad Request: invalid group_id")
    assert _is_transient_error(err) is False


def test_is_transient_error_408_is_transient():
    err = GraphitiClientError("408 Request Timeout")
    assert _is_transient_error(err) is True


def test_is_transient_error_429_is_transient():
    err = GraphitiClientError("429 Too Many Requests")
    assert _is_transient_error(err) is True


# ---------------------------------------------------------------------------
# Channel A tests
# ---------------------------------------------------------------------------


class _StubEngine:
    """Minimal stand-in for ``MemoryEngine`` — the channel A handler only
    calls ``engine._get_backend()`` (for the write-back helper) and never
    actually touches the engine inside ``_handle_invalidated_edges`` itself.
    """


def _results(*uris: str | None, invalid_at: str | None = "2026-01-01T00:00:00Z"):
    """Build ``AddTripletResults`` with a single edge per source_uri."""
    from hindsight_api.engine.federation.graphiti_client import (
        AddTripletResults,
        EdgeResult,
    )

    edges = [
        EdgeResult(
            uuid=uuid4(),
            name="WORKS_AT",
            fact="Alice works at Acme",
            valid_at="2026-01-01T00:00:00Z",
            invalid_at=invalid_at,
            source_uri=uri,
            group_id="g1",
        )
        for uri in uris
    ]
    return AddTripletResults(edges=edges, nodes=[])


async def test_handle_invalidated_edges_marks_local_memory():
    """A source_uri matching the bank's prefix → EdgeResult stored in the
    memo keyed by edge_uuid, so end-of-drain can replay it through
    ``handle_graphiti_edge_invalidated``."""
    memo: dict[UUID, EdgeResult] = {}
    bank_id = "bank-A"
    memory_id = uuid4()
    uri = f"hindsight://bank/{bank_id}/memory/{memory_id}"
    results = _results(uri)

    count = await _handle_invalidated_edges(_StubEngine(), bank_id, results, memo)  # type: ignore[arg-type]

    assert count == 1
    assert len(memo) == 1
    stored = next(iter(memo.values()))
    assert stored.source_uri == uri
    assert stored.uuid in memo
    assert stored.invalid_at == "2026-01-01T00:00:00Z"


async def test_handle_invalidated_edges_ignores_other_banks():
    """Cross-bank source_uris are NOT memoized — owner bank is the only one
    that gets to re-consolidate its private memory (per main plan §6-5)."""
    memo: dict[UUID, EdgeResult] = {}
    foreign = _results(f"hindsight://bank/some-other-bank/memory/{uuid4()}")
    count = await _handle_invalidated_edges(_StubEngine(), "bank-A", foreign, memo)  # type: ignore[arg-type]
    assert count == 1  # the loop counts every invalidated edge, not just kept ones
    assert len(memo) == 0  # but the memo is bank-local


async def test_handle_invalidated_edges_unparseable_uri_logged_and_skipped():
    """Malformed URIs produce a warning, not a crash."""
    memo: dict[UUID, EdgeResult] = {}
    results = _results("hindsight://bank/bank-A/memory/not-a-uuid")
    count = await _handle_invalidated_edges(_StubEngine(), "bank-A", results, memo)  # type: ignore[arg-type]
    assert count == 1
    assert len(memo) == 0


async def test_trigger_backflow_actions_replays_through_engine_method():
    """End-of-drain: every memoized edge → engine.handle_graphiti_edge_invalidated
    is called once with the edge's uuid / source_uri / invalid_at. The
    engine method is the same primitive channel B uses, so the two paths
    produce identical DB writes + audit entries."""
    from hindsight_api.engine.federation.graphiti_client import EdgeResult

    edge1 = EdgeResult(
        uuid=uuid4(),
        name="WORKS_AT",
        fact="Alice works at Acme",
        valid_at=None,
        invalid_at="2026-06-01T00:00:00Z",
        source_uri=f"hindsight://bank/bank-A/memory/{uuid4()}",
        group_id="g1",
    )
    edge2 = EdgeResult(
        uuid=uuid4(),
        name="LIVES_IN",
        fact="Bob lives in Berlin",
        valid_at=None,
        invalid_at="2026-06-01T00:00:00Z",
        source_uri=f"hindsight://bank/bank-A/memory/{uuid4()}",
        group_id="g1",
    )
    memo: dict[UUID, EdgeResult] = {edge1.uuid: edge1, edge2.uuid: edge2}

    replays: list[dict[str, Any]] = []

    async def handle(*, bank_id, edge_uuid, source_uri, invalid_at, request_context):
        replays.append(
            {
                "bank_id": bank_id,
                "edge_uuid": edge_uuid,
                "source_uri": source_uri,
                "invalid_at": invalid_at,
                "internal": request_context.internal,
            }
        )

    class _Engine:
        async def handle_graphiti_edge_invalidated(self, **kw):
            await handle(**kw)

    await _trigger_backflow_actions(_Engine(), "bank-A", memo)  # type: ignore[arg-type]

    assert len(replays) == 2
    replayed_uuids = {r["edge_uuid"] for r in replays}
    assert replayed_uuids == {str(edge1.uuid), str(edge2.uuid)}
    for r in replays:
        assert r["bank_id"] == "bank-A"
        assert r["internal"] is True
        assert r["invalid_at"] is not None  # parsed to datetime
    # Memo is cleared so a follow-up drain with no invalidated edges is a no-op.
    assert len(memo) == 0


async def test_trigger_backflow_actions_no_op_when_memo_empty():
    """No invalidated edges → no replay. Keeps the cheap-path free of
    pointless engine calls."""
    memo: dict[UUID, EdgeResult] = {}

    class _Engine:
        async def handle_graphiti_edge_invalidated(self, **_):
            pytest.fail("should not replay when memo is empty")

    await _trigger_backflow_actions(_Engine(), "bank-A", memo)  # type: ignore[arg-type]


async def test_trigger_backflow_actions_continues_after_per_edge_failure():
    """A per-edge failure (e.g. transient DB blip) must not abort the
    rest of the batch — every other memoized edge still gets replayed.
    Per deep-dive 4 §1.3: at-least-once replay, best-effort delivery."""
    from hindsight_api.engine.federation.graphiti_client import EdgeResult

    good1 = EdgeResult(
        uuid=uuid4(),
        name="WORKS_AT",
        fact="Alice works at Acme",
        valid_at=None,
        invalid_at="2026-06-01T00:00:00Z",
        source_uri=f"hindsight://bank/bank-A/memory/{uuid4()}",
        group_id="g1",
    )
    good2 = EdgeResult(
        uuid=uuid4(),
        name="LIVES_IN",
        fact="Bob lives in Berlin",
        valid_at=None,
        invalid_at="2026-06-01T00:00:00Z",
        source_uri=f"hindsight://bank/bank-A/memory/{uuid4()}",
        group_id="g1",
    )
    memo: dict[UUID, EdgeResult] = {good1.uuid: good1, good2.uuid: good2}

    replays: list[str] = []

    class _Engine:
        async def handle_graphiti_edge_invalidated(self, *, edge_uuid, **_):
            replays.append(edge_uuid)
            if edge_uuid == str(good1.uuid):
                raise RuntimeError("simulated transient blip")

    await _trigger_backflow_actions(_Engine(), "bank-A", memo)  # type: ignore[arg-type]

    # Both edges attempted, in some order; the second one still replayed
    # despite the first one's failure. Sort both sides because UUID4
    # values are random — the literal list order is not stable across
    # runs, only the *set* of attempted edges is.
    assert sorted(replays) == sorted([str(good1.uuid), str(good2.uuid)])
    assert len(memo) == 0  # memo always cleared, even on partial failure


# ---------------------------------------------------------------------------
# End-to-end drain test with mocked Graphiti transport
# ---------------------------------------------------------------------------


def _row(memory_id: UUID, group_id: str, relations: list[dict], entities: list[dict] | None = None):
    """Build a fake outbox row as the worker would read it from
    ``claim_graphiti_outbox_batch``."""
    if entities is None:
        entities = [
            {"text": "Alice", "graphiti_uuid": None},
            {"text": "Acme", "graphiti_uuid": None},
        ]
    return {
        "id": 1,
        "bank_id": "bank-A",
        "memory_id": memory_id,
        "group_id": group_id,
        "fact_text": "Alice works at Acme",
        "entities": entities,
        "relations": relations,
        "tags": ["test"],
    }


class _FakeOps:
    """In-memory replacement for ``backend.ops``.

    Captures all claim/reschedule calls so the test can assert on them.
    Mirrors the real PG/Oracle semantics: ``claim_graphiti_outbox_batch``
    *removes* the claimed rows from the queue (real impl uses
    ``DELETE ... RETURNING``); the queue is empty when claim returns ``[]``.
    """

    def __init__(self, rows: list[dict]):
        self._rows = list(rows)
        self.rescheduled: list[dict] = []
        self.claimed_batches = 0

    async def claim_graphiti_outbox_batch(self, _conn, _table, bank_id, batch_size):
        self.claimed_batches += 1
        batch = self._rows[:batch_size]
        self._rows = self._rows[batch_size:]
        return batch

    async def reschedule_graphiti_outbox_rows(self, _conn, _table, ids, last_error):
        self.rescheduled.append({"ids": list(ids), "last_error": last_error})


class _FakeBackend:
    """Minimal asyncpg.Pool stand-in — exposes ``acquire()`` as a coroutine
    returning a connection whose ``execute`` returns an asyncpg-style
    ``UPDATE N`` tag. The write-back parser in graphiti_forward expects that
    exact string shape, so the fake must reproduce it."""

    def __init__(self, ops: _FakeOps):
        self.ops = ops
        self.executed: list[tuple[str, tuple]] = []

    async def acquire(self):
        return _FakeConn(self)

    async def release(self, _conn):
        return None


class _FakeConn:
    def __init__(self, backend: _FakeBackend):
        self._backend = backend

    async def execute(self, sql: str, *args):
        # Mirror asyncpg's "UPDATE N" return tag so the write-back parser is happy.
        self._backend.executed.append((sql, args))
        if sql.lstrip().upper().startswith("UPDATE"):
            return "UPDATE 0"
        return "SELECT 0"

    def transaction(self):
        return _FakeTx()


class _FakeTx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConfigResolver:
    def __init__(self, config):
        self._config = config

    async def resolve_full_config(self, _bank_id, _ctx):
        return self._config


class _FakeEngine:
    def __init__(
        self,
        *,
        ops: _FakeOps,
        config,
        submit_consolidation=None,
        handle_backflow=None,
    ):
        self._backend = _FakeBackend(ops)
        self._config_resolver = _FakeConfigResolver(config)
        self._submit_consolidation = submit_consolidation
        self._handle_backflow = handle_backflow

    async def _get_backend(self):
        return self._backend

    async def submit_async_consolidation(self, *, bank_id, request_context):
        if self._submit_consolidation is not None:
            await self._submit_consolidation(bank_id=bank_id, request_context=request_context)

    async def handle_graphiti_edge_invalidated(
        self,
        *,
        bank_id: str,
        edge_uuid: str,
        source_uri: str,
        invalid_at,  # datetime
        superseded_by_fact: str | None = None,
        request_context,
    ):
        # Default fake: just record the call. Tests that don't care about
        # the backflow path can ignore the recorder; tests that do care
        # pass ``handle_backflow=`` to inspect the args. We return a
        # minimal ``GraphitiBackflowResult`` (not_found=False) so the
        # production code's ``result.not_found`` check has a real
        # attribute to read — mirrors the contract the real engine
        # method guarantees.
        if self._handle_backflow is not None:
            await self._handle_backflow(
                bank_id=bank_id,
                edge_uuid=edge_uuid,
                source_uri=source_uri,
                invalid_at=invalid_at,
                request_context=request_context,
            )
        from hindsight_api.engine.response_models import GraphitiBackflowResult

        return GraphitiBackflowResult(edge_uuid=edge_uuid, memory_id=None, not_found=False)


def _config(group_id: str = "g1", base_url: str = "http://graphiti.test"):
    return SimpleNamespace(
        graphiti_group_id=group_id,
        graphiti_base_url=base_url,
        graphiti_api_key=None,
    )


def _build_mocked_client(handler, *, base_url: str = "http://graphiti.test") -> GraphitiClient:
    """Return a GraphitiClient that routes ``add_triplet`` to a fake handler.

    We hand-build a custom httpx.AsyncClient and pass it in so the Graphiti
    client doesn't take ownership of its own transport.
    """
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url=base_url)
    return GraphitiClient(base_url=base_url, api_key=None, client=http)


def _request_context():
    from hindsight_api.models import RequestContext

    return RequestContext(internal=True)


async def test_run_graphiti_forward_job_happy_path():
    """One row with one relation → one add_triplet call, no reschedule."""
    memory_id = uuid4()
    row = _row(
        memory_id=memory_id,
        group_id="g1",
        relations=[
            {
                "source_entity_index": 0,
                "target_entity_index": 1,
                "predicate": "WORKS_AT",
                "rel_valid_at": None,
                "rel_invalid_at": None,
            }
        ],
    )
    ops = _FakeOps([row])
    engine = _FakeEngine(ops=ops, config=_config())

    seen_requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_requests.append(body)
        return httpx.Response(
            200,
            json={
                "edges": [
                    {
                        "uuid": str(uuid4()),
                        "name": "WORKS_AT",
                        "fact": "Alice works at Acme",
                        "valid_at": "2026-01-01T00:00:00Z",
                        "invalid_at": None,
                        "attributes": {"source_uri": f"hindsight://bank/bank-A/memory/{memory_id}"},
                        "group_id": "g1",
                    }
                ],
                "nodes": [
                    {"uuid": str(uuid4()), "name": "Alice", "group_id": "g1"},
                    {"uuid": str(uuid4()), "name": "Acme", "group_id": "g1"},
                ],
            },
        )

    client = _build_mocked_client(handler)
    from hindsight_api.engine.retain.graphiti_forward import run_graphiti_forward_job

    result = await run_graphiti_forward_job(
        memory_engine=engine,  # type: ignore[arg-type]
        bank_id="bank-A",
        request_context=_request_context(),
        client_factory=lambda _cfg: client,
    )

    assert result["drained"] == 1
    assert result["forwarded"] == 1
    assert result["relations"] == 1
    assert result["invalidated_edges"] == 0
    assert result["rescheduled"] == 0
    assert result["dropped"] == 0
    assert result["errors"] == 0
    # The single add_triplet was sent with the deterministic edge UUID
    assert len(seen_requests) == 1
    assert seen_requests[0]["edge"]["uuid"] == str(_deterministic_edge_uuid(memory_id, row["relations"][0]))
    assert seen_requests[0]["edge"]["attributes"]["source_uri"] == (f"hindsight://bank/bank-A/memory/{memory_id}")
    # No failures → reschedule was never called
    assert ops.rescheduled == []


async def test_run_graphiti_forward_job_transient_failure_reschedules():
    """Server returns 503 → row is rescheduled (not dropped)."""
    memory_id = uuid4()
    row = _row(
        memory_id=memory_id,
        group_id="g1",
        relations=[
            {
                "source_entity_index": 0,
                "target_entity_index": 1,
                "predicate": "WORKS_AT",
                "rel_valid_at": None,
                "rel_invalid_at": None,
            }
        ],
    )
    ops = _FakeOps([row])
    engine = _FakeEngine(ops=ops, config=_config())

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "upstream overloaded"})

    client = _build_mocked_client(handler)
    from hindsight_api.engine.retain.graphiti_forward import run_graphiti_forward_job

    result = await run_graphiti_forward_job(
        memory_engine=engine,  # type: ignore[arg-type]
        bank_id="bank-A",
        request_context=_request_context(),
        client_factory=lambda _cfg: client,
    )

    assert result["drained"] == 1
    assert result["forwarded"] == 0
    assert result["rescheduled"] == 1
    assert result["dropped"] == 0
    # Reschedule was called once with the row's id and a 503-derived message.
    assert len(ops.rescheduled) == 1
    assert ops.rescheduled[0]["ids"] == [1]
    assert "503" in ops.rescheduled[0]["last_error"]


async def test_run_graphiti_forward_job_permanent_failure_drops():
    """Server returns 400 → row is dropped (not rescheduled). The bank's
    queue must not be poisoned by a misconfigured graphiti backend."""
    row = _row(
        memory_id=uuid4(),
        group_id="g1",
        relations=[
            {
                "source_entity_index": 0,
                "target_entity_index": 1,
                "predicate": "WORKS_AT",
                "rel_valid_at": None,
                "rel_invalid_at": None,
            }
        ],
    )
    ops = _FakeOps([row])
    engine = _FakeEngine(ops=ops, config=_config())

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad group_id"})

    client = _build_mocked_client(handler)
    from hindsight_api.engine.retain.graphiti_forward import run_graphiti_forward_job

    result = await run_graphiti_forward_job(
        memory_engine=engine,  # type: ignore[arg-type]
        bank_id="bank-A",
        request_context=_request_context(),
        client_factory=lambda _cfg: client,
    )

    assert result["drained"] == 1
    assert result["forwarded"] == 0
    assert result["rescheduled"] == 0
    assert result["dropped"] == 1
    assert ops.rescheduled == []


async def test_run_graphiti_forward_job_channel_a_replays_through_engine_method():
    """An invalidated edge in the response → end-of-drain replay through
    ``engine.handle_graphiti_edge_invalidated``, exactly once per edge.
    The engine method is the same primitive channel B uses, so a
    forwarder-driven invalidation and an overlay-driven invalidation
    produce identical writes + audit entries."""
    bank_id = "bank-A"
    memory_id = uuid4()
    row = _row(
        memory_id=memory_id,
        group_id="g1",
        relations=[
            {
                "source_entity_index": 0,
                "target_entity_index": 1,
                "predicate": "WORKS_AT",
                "rel_valid_at": None,
                "rel_invalid_at": None,
            }
        ],
    )
    ops = _FakeOps([row])

    backflow_calls: list[dict] = []

    async def record(**kwargs):
        backflow_calls.append(kwargs)

    engine = _FakeEngine(ops=ops, config=_config(), handle_backflow=record)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "edges": [
                    {
                        "uuid": str(uuid4()),
                        "name": "WORKS_AT",
                        "fact": "Alice works at Acme",
                        "valid_at": "2026-01-01T00:00:00Z",
                        "invalid_at": "2026-06-01T00:00:00Z",  # ← invalidated
                        "attributes": {"source_uri": f"hindsight://bank/{bank_id}/memory/{memory_id}"},
                        "group_id": "g1",
                    }
                ],
                "nodes": [
                    {"uuid": str(uuid4()), "name": "Alice", "group_id": "g1"},
                    {"uuid": str(uuid4()), "name": "Acme", "group_id": "g1"},
                ],
            },
        )

    client = _build_mocked_client(handler)
    from hindsight_api.engine.retain.graphiti_forward import run_graphiti_forward_job

    result = await run_graphiti_forward_job(
        memory_engine=engine,  # type: ignore[arg-type]
        bank_id=bank_id,
        request_context=_request_context(),
        client_factory=lambda _cfg: client,
    )

    assert result["invalidated_edges"] == 1
    assert result["forwarded"] == 1
    # Exactly one end-of-drain replay, with the edge's uuid / source_uri
    # / invalid_at threaded through to the shared engine method. The
    # context is internal (forwarder, not an overlay call).
    assert len(backflow_calls) == 1
    call = backflow_calls[0]
    assert call["bank_id"] == bank_id
    assert call["source_uri"] == f"hindsight://bank/{bank_id}/memory/{memory_id}"
    assert call["invalid_at"] is not None  # parsed to a tz-aware datetime
    assert call["request_context"].internal is True


async def test_run_graphiti_forward_job_skips_when_bank_lost_federation():
    """Bank config no longer has graphiti_group_id → drain is a no-op and
    the outbox rows stay queued for a future submission (we never delete
    rows in this branch)."""
    row = _row(
        memory_id=uuid4(),
        group_id="g1",
        relations=[
            {
                "source_entity_index": 0,
                "target_entity_index": 1,
                "predicate": "WORKS_AT",
                "rel_valid_at": None,
                "rel_invalid_at": None,
            }
        ],
    )
    ops = _FakeOps([row])
    # Bank no longer federated
    engine = _FakeEngine(ops=ops, config=_config(group_id=""))
    # No HTTP handler needed — the worker should bail out before calling.
    client = _build_mocked_client(lambda _req: httpx.Response(500))
    from hindsight_api.engine.retain.graphiti_forward import run_graphiti_forward_job

    result = await run_graphiti_forward_job(
        memory_engine=engine,  # type: ignore[arg-type]
        bank_id="bank-A",
        request_context=_request_context(),
        client_factory=lambda _cfg: client,
    )

    assert result == {
        "drained": 0,
        "forwarded": 0,
        "relations": 0,
        "invalidated_edges": 0,
        "rescheduled": 0,
        "dropped": 0,
        "write_backs": 0,
        "errors": 0,
    }
    # claim() was never called → rows are still in the outbox.
    assert ops.claimed_batches == 0


# ---------------------------------------------------------------------------
# Env-var fallback (cross-commit fix surfaced by C3 docs)
# ---------------------------------------------------------------------------


def test_build_client_prefers_hindsight_api_prefixed_env():
    """HINDSIGHT_API_GRAPHITI_BASE_URL is the documented env name; it must
    win over the bare alias when both are set (otherwise the user's
    .env.example value would be ignored).
    """
    from hindsight_api.engine.retain import graphiti_forward as gf
    from hindsight_api.engine.retain.graphiti_forward import _build_client

    captured: dict[str, str | None] = {}

    class _RecordingClient:
        def __init__(self, base_url: str, api_key: str | None = None) -> None:
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    orig = gf.GraphitiClient
    gf.GraphitiClient = _RecordingClient  # type: ignore[assignment]
    try:
        env = {
            "HINDSIGHT_API_GRAPHITI_BASE_URL": "http://primary:1234",
            "HINDSIGHT_API_GRAPHITI_API_KEY": "primary-key",
            "GRAPHITI_BASE_URL": "http://legacy:5678",
            "GRAPHITI_API_KEY": "legacy-key",
        }
        with patch.dict(os.environ, env, clear=False):
            _build_client(SimpleNamespace(graphiti_base_url=None, graphiti_api_key=None))
        assert captured["base_url"] == "http://primary:1234"
        assert captured["api_key"] == "primary-key"
    finally:
        gf.GraphitiClient = orig  # type: ignore[assignment]


def test_build_client_falls_back_to_bare_env_alias():
    """When the HINDSIGHT_API_-prefixed form is not set, the bare
    GRAPHITI_BASE_URL is honored for backward compat.
    """
    from hindsight_api.engine.retain import graphiti_forward as gf
    from hindsight_api.engine.retain.graphiti_forward import _build_client

    captured: dict[str, str | None] = {}

    class _RecordingClient:
        def __init__(self, base_url: str, api_key: str | None = None) -> None:
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    orig = gf.GraphitiClient
    gf.GraphitiClient = _RecordingClient  # type: ignore[assignment]
    try:
        with patch.dict(
            os.environ,
            {
                "GRAPHITI_BASE_URL": "http://legacy:5678",
                "GRAPHITI_API_KEY": "legacy-key",
            },
            clear=False,
        ):
            os.environ.pop("HINDSIGHT_API_GRAPHITI_BASE_URL", None)
            os.environ.pop("HINDSIGHT_API_GRAPHITI_API_KEY", None)
            _build_client(SimpleNamespace(graphiti_base_url=None, graphiti_api_key=None))
        assert captured["base_url"] == "http://legacy:5678"
        assert captured["api_key"] == "legacy-key"
    finally:
        gf.GraphitiClient = orig  # type: ignore[assignment]
