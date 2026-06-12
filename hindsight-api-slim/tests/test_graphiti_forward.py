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
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from hindsight_api.engine.federation.circuit_breaker import CircuitOpenError
from hindsight_api.engine.federation.graphiti_client import (
    GraphitiClient,
    GraphitiClientError,
)
from hindsight_api.engine.retain.graphiti_forward import (
    _INVALIDATED_MEMO,
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
    """A source_uri matching the bank's prefix → memo populated."""
    _INVALIDATED_MEMO.clear()
    bank_id = "bank-A"
    memory_id = uuid4()
    uri = f"hindsight://bank/{bank_id}/memory/{memory_id}"
    results = _results(uri)

    count = await _handle_invalidated_edges(_StubEngine(), bank_id, results)  # type: ignore[arg-type]

    assert count == 1
    assert memory_id in _INVALIDATED_MEMO


async def test_handle_invalidated_edges_ignores_other_banks():
    """Cross-bank source_uris are NOT memoized — owner bank is the only one
    that gets to re-consolidate its private memory (per main plan §6-5)."""
    _INVALIDATED_MEMO.clear()
    foreign = _results(f"hindsight://bank/some-other-bank/memory/{uuid4()}")
    count = await _handle_invalidated_edges(_StubEngine(), "bank-A", foreign)  # type: ignore[arg-type]
    assert count == 1  # the loop counts every invalidated edge, not just kept ones
    assert len(_INVALIDATED_MEMO) == 0  # but the memo is bank-local


async def test_handle_invalidated_edges_unparseable_uri_logged_and_skipped():
    """Malformed URIs produce a warning, not a crash."""
    _INVALIDATED_MEMO.clear()
    results = _results("hindsight://bank/bank-A/memory/not-a-uuid")
    count = await _handle_invalidated_edges(_StubEngine(), "bank-A", results)  # type: ignore[arg-type]
    assert count == 1
    assert len(_INVALIDATED_MEMO) == 0


async def test_trigger_backflow_actions_submits_consolidation():
    """End-of-drain: at least one invalidated memory → engine.submit_async_consolidation
    is called once with the bank_id and an internal request context."""
    _INVALIDATED_MEMO.clear()
    _INVALIDATED_MEMO[uuid4()] = None
    _INVALIDATED_MEMO[uuid4()] = None

    submitted: list[dict[str, Any]] = []

    class _Engine:
        async def submit_async_consolidation(self, *, bank_id, request_context):
            submitted.append({"bank_id": bank_id, "internal": request_context.internal})

    await _trigger_backflow_actions(_Engine(), "bank-A")  # type: ignore[arg-type]

    assert len(submitted) == 1
    assert submitted[0]["bank_id"] == "bank-A"
    assert submitted[0]["internal"] is True
    # Memo is cleared so a follow-up drain with no invalidated edges is a no-op.
    assert len(_INVALIDATED_MEMO) == 0


async def test_trigger_backflow_actions_no_op_when_memo_empty():
    """No invalidated edges → no submit. Keeps the cheap-path free of
    pointless job submissions."""
    _INVALIDATED_MEMO.clear()

    class _Engine:
        async def submit_async_consolidation(self, **_):
            pytest.fail("should not submit when memo is empty")

    await _trigger_backflow_actions(_Engine(), "bank-A")  # type: ignore[arg-type]


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
    def __init__(self, *, ops: _FakeOps, config, submit_consolidation=None):
        self._backend = _FakeBackend(ops)
        self._config_resolver = _FakeConfigResolver(config)
        self._submit_consolidation = submit_consolidation

    async def _get_backend(self):
        return self._backend

    async def submit_async_consolidation(self, *, bank_id, request_context):
        if self._submit_consolidation is not None:
            await self._submit_consolidation(bank_id=bank_id, request_context=request_context)


def _config(group_id: str = "g1", base_url: str = "http://graphiti.test"):
    from types import SimpleNamespace

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
    _INVALIDATED_MEMO.clear()
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
    _INVALIDATED_MEMO.clear()
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
    _INVALIDATED_MEMO.clear()
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


async def test_run_graphiti_forward_job_channel_a_triggers_consolidation():
    """An invalidated edge in the response → end-of-drain consolidation
    submit, exactly once per drain."""
    _INVALIDATED_MEMO.clear()
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

    submissions: list[dict] = []

    async def submit(*, bank_id, request_context):
        submissions.append({"bank_id": bank_id, "internal": request_context.internal})

    engine = _FakeEngine(ops=ops, config=_config(), submit_consolidation=submit)

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
    # Exactly one end-of-drain consolidation submit, internal context
    assert len(submissions) == 1
    assert submissions[0]["bank_id"] == bank_id
    assert submissions[0]["internal"] is True


async def test_run_graphiti_forward_job_skips_when_bank_lost_federation():
    """Bank config no longer has graphiti_group_id → drain is a no-op and
    the outbox rows stay queued for a future submission (we never delete
    rows in this branch)."""
    _INVALIDATED_MEMO.clear()
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
