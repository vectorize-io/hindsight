"""Tests for the channel-C backflow polling worker (deep-dive 5).

The worker is the time-driven complement to ``graphiti_forward`` (channel
A) and the HTTP endpoint (channel B). Tests cover:

* Wire layer — ``GraphitiClient.list_invalidated_edges`` parses success
  responses, surfaces ``truncated=True``, and wraps HTTP / parse errors.
* Pure helpers — ``_is_local_edge`` source_uri prefix filter,
  ``_compute_new_since`` cursor arithmetic (max + clamp + empty).
* Stub integration — ``_poll_one_bank`` replays local edges through
  ``MemoryEngine.handle_graphiti_edge_invalidated`` (the same primitive
  channels A and B use), filters out cross-bank / unparseable edges,
  handles truncated responses by NOT advancing the cursor, and handles
  per-edge engine-method failures without poisoning the batch.
* Worker loop — ``run_graphiti_backflow_poller`` skips banks without
  ``graphiti_backflow_polling_enabled`` or without
  ``graphiti_group_id``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest

from hindsight_api.engine.federation.graphiti_client import (
    EdgeResult,
    GraphitiClient,
    GraphitiClientError,
    InvalidatedEdgesResponse,
)
from hindsight_api.engine.retain.graphiti_backflow_poller import (
    PollTickResult,
    _compute_new_since,
    _is_local_edge,
    _poll_one_bank,
    _read_poller_state,
    _replay_local_edges,
    _write_poller_state,
    run_graphiti_backflow_poller,
)


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_is_local_edge_matches_bank_prefix():
    bank_id = "agent-7"
    edge = EdgeResult(
        uuid=uuid4(),
        name="WORKS_AT",
        fact="x",
        valid_at=None,
        invalid_at="2024-01-01T00:00:00Z",
        source_uri=f"hindsight://bank/{bank_id}/memory/{uuid4()}",
        group_id=bank_id,
    )
    assert _is_local_edge(bank_id, edge) is True


def test_is_local_edge_rejects_other_bank():
    edge = EdgeResult(
        uuid=uuid4(),
        name="WORKS_AT",
        fact="x",
        valid_at=None,
        invalid_at="2024-01-01T00:00:00Z",
        source_uri=f"hindsight://bank/agent-OTHER/memory/{uuid4()}",
        group_id="agent-OTHER",
    )
    assert _is_local_edge("agent-7", edge) is False


def test_is_local_edge_rejects_missing_uri():
    edge = EdgeResult(
        uuid=uuid4(),
        name="WORKS_AT",
        fact="x",
        valid_at=None,
        invalid_at="2024-01-01T00:00:00Z",
        source_uri=None,
        group_id="agent-7",
    )
    assert _is_local_edge("agent-7", edge) is False


def test_compute_new_since_picks_max():
    e1 = EdgeResult(
        uuid=uuid4(),
        name="X",
        fact="",
        valid_at=None,
        invalid_at="2024-01-01T00:00:00+00:00",
        source_uri=None,
        group_id="b",
    )
    e2 = EdgeResult(
        uuid=uuid4(),
        name="X",
        fact="",
        valid_at=None,
        invalid_at="2024-02-01T00:00:00+00:00",
        source_uri=None,
        group_id="b",
    )
    e3 = EdgeResult(
        uuid=uuid4(),
        name="X",
        fact="",
        valid_at=None,
        invalid_at="2024-01-15T00:00:00+00:00",
        source_uri=None,
        group_id="b",
    )
    floor = datetime(1970, 1, 1, tzinfo=timezone.utc)
    result = _compute_new_since([e1, e2, e3], floor)
    assert result == datetime(2024, 2, 1, tzinfo=timezone.utc)


def test_compute_new_since_clamps_to_floor():
    """A parseable edge earlier than the current cursor must not move
    the cursor backward (deep-dive 5 §3.7 — Graphiti filters ``>= since``
    but a contract regression allowing ``>`` is not worth a re-poll loop).
    """
    floor = datetime(2024, 6, 1, tzinfo=timezone.utc)
    edge = EdgeResult(
        uuid=uuid4(),
        name="X",
        fact="",
        valid_at=None,
        invalid_at="2024-01-01T00:00:00+00:00",
        source_uri=None,
        group_id="b",
    )
    assert _compute_new_since([edge], floor) == floor


def test_compute_new_since_handles_empty():
    floor = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert _compute_new_since([], floor) is None


def test_compute_new_since_skips_unparseable_and_none():
    floor = datetime(1970, 1, 1, tzinfo=timezone.utc)
    bad = EdgeResult(
        uuid=uuid4(),
        name="X",
        fact="",
        valid_at=None,
        invalid_at="not-a-date",
        source_uri=None,
        group_id="b",
    )
    none_dt = EdgeResult(
        uuid=uuid4(),
        name="X",
        fact="",
        valid_at=None,
        invalid_at=None,
        source_uri=None,
        group_id="b",
    )
    good = EdgeResult(
        uuid=uuid4(),
        name="X",
        fact="",
        valid_at=None,
        invalid_at="2024-03-01T00:00:00+00:00",
        source_uri=None,
        group_id="b",
    )
    assert _compute_new_since([bad, none_dt, good], floor) == datetime(2024, 3, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Wire-layer tests for ``GraphitiClient.list_invalidated_edges``
# ---------------------------------------------------------------------------


def _stub_client(handler) -> GraphitiClient:
    """Build a GraphitiClient that routes calls through a sync handler.

    We don't want the real circuit-breaker / timeout machinery in unit
    tests — the breaker is unit-tested elsewhere. Bypass it by handing
    the client a pre-built httpx.AsyncClient and letting our handler
    intercept transport.
    """

    async def _send(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        return handler(request, body)

    transport = httpx.MockTransport(_send)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://graphiti")
    return GraphitiClient(base_url="http://graphiti", api_key=None, client=http_client)


def test_list_invalidated_edges_parses_response():
    def handler(request, body):
        return httpx.Response(
            200,
            json={
                "edges": [
                    {
                        "uuid": str(uuid4()),
                        "name": "WORKS_AT",
                        "fact": "old fact",
                        "valid_at": None,
                        "invalid_at": "2024-01-01T00:00:00+00:00",
                        "group_id": "agent-7",
                        "attributes": {
                            "source_uri": "hindsight://bank/agent-7/memory/00000000-0000-0000-0000-000000000001"
                        },
                    }
                ],
                "truncated": False,
            },
        )

    async def go():
        client = _stub_client(handler)
        try:
            resp = await client.list_invalidated_edges(
                group_ids=["agent-7"],
                since=datetime(2024, 1, 1, tzinfo=timezone.utc),
                max_edges=50,
            )
        finally:
            await client.aclose()
        return resp

    resp = asyncio.run(go())
    assert len(resp.edges) == 1
    assert resp.edges[0].source_uri.startswith("hindsight://bank/agent-7/")
    assert resp.truncated is False


def test_list_invalidated_edges_handles_truncated_flag():
    def handler(request, body):
        return httpx.Response(200, json={"edges": [], "truncated": True})

    async def go():
        client = _stub_client(handler)
        try:
            return await client.list_invalidated_edges(group_ids=["agent-7"])
        finally:
            await client.aclose()

    resp = asyncio.run(go())
    assert resp.truncated is True
    assert resp.edges == []


def test_list_invalidated_edges_wraps_http_error():
    def handler(request, body):
        return httpx.Response(500, text="boom")

    async def go():
        client = _stub_client(handler)
        try:
            return await client.list_invalidated_edges(group_ids=["agent-7"])
        finally:
            await client.aclose()

    with pytest.raises(GraphitiClientError, match="HTTP error"):
        asyncio.run(go())


def test_list_invalidated_edges_naive_since_assumed_utc():
    """The wire format requires tz-aware ISO-8601. We coerce naive
    datetimes to UTC silently (matches the engine-side convention;
    see graphiti_forward._parse_iso_to_utc)."""

    captured: dict[str, Any] = {}

    def handler(request, body):
        captured["body"] = body
        return httpx.Response(200, json={"edges": [], "truncated": False})

    async def go():
        client = _stub_client(handler)
        try:
            await client.list_invalidated_edges(
                group_ids=["agent-7"],
                since=datetime(2024, 1, 1),  # naive
            )
        finally:
            await client.aclose()

    asyncio.run(go())
    assert captured["body"]["since"].endswith("+00:00")


# ---------------------------------------------------------------------------
# Stub-integration tests for ``_poll_one_bank``
# ---------------------------------------------------------------------------


def _make_engine_mock(*, since_row: datetime | None, edges: list[EdgeResult], truncated: bool = False):
    """Build a MemoryEngine double with just enough surface for _poll_one_bank.

    The engine method has a lot of internal collaborators (_get_backend,
    _config_resolver, list_banks, etc.) — for these tests we only need
    _get_backend (returns a context manager that yields a conn) plus
    handle_graphiti_edge_invalidated (replay target).
    """
    engine = MagicMock()
    backend = MagicMock()
    conn = AsyncMock()

    async def _fetchrow(query, *args):
        if "SELECT last_seen_invalid_at" in query:
            return {"last_seen_invalid_at": since_row} if since_row is not None else None
        return None

    async def _execute(query, *args):
        return "INSERT 0 1"

    conn.fetchrow = _fetchrow
    conn.execute = _execute
    backend.acquire = MagicMock()
    backend.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    backend.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    async def _get_backend():
        return backend

    engine._get_backend = _get_backend
    return engine, backend, conn


def _make_client_mock(*, edges: list[EdgeResult] | None = None, truncated: bool = False):
    """Build a GraphitiClient double with ``list_invalidated_edges`` as
    an AsyncMock returning a canned ``InvalidatedEdgesResponse``.

    Tests that need different behavior (truncated responses, errors)
    override ``client.list_invalidated_edges`` after construction.
    """
    client = MagicMock()
    client.list_invalidated_edges = AsyncMock(
        return_value=InvalidatedEdgesResponse(edges=edges or [], truncated=truncated)
    )
    return client


def _wire_engine_backend(engine) -> None:
    """Wire ``engine._get_backend`` to a no-op backend for tests that
    build their own MagicMock engine (i.e. don't use ``_make_engine_mock``).

    The poller worker calls ``_get_backend`` inside ``_poll_one_bank``
    to read the poller state. For tests that don't care about the
    state row we hand back a backend whose ``acquire()`` returns
    ``None`` — ``acquire_with_retry`` treats that as "no row" and the
    worker proceeds to the Graphiti call.

    Without this helper the worker's ``async with backend.acquire() as
    conn`` hangs forever on a bare ``MagicMock`` (the default
    ``__aenter__`` returns a MagicMock, not ``None``).
    """
    backend = MagicMock()
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    backend.acquire = MagicMock()
    backend.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    backend.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    engine._get_backend = AsyncMock(return_value=backend)


def test_poll_one_bank_empty_response_does_not_advance_cursor():
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    engine, _backend, _conn = _make_engine_mock(since_row=since, edges=[])

    client = MagicMock()
    client.list_invalidated_edges = AsyncMock(return_value=InvalidatedEdgesResponse(edges=[], truncated=False))
    engine.handle_graphiti_edge_invalidated = AsyncMock()

    async def go():
        return await _poll_one_bank(engine, client, "agent-7", internal=SimpleNamespace(internal=True))

    result = asyncio.run(go())
    assert result.polled is True
    assert result.edges_seen == 0
    assert result.edges_replayed == 0
    # The cursor stays put when there's nothing to advance over.
    assert result.new_since == since
    engine.handle_graphiti_edge_invalidated.assert_not_awaited()


def test_poll_one_bank_replays_local_edges_and_advances_cursor():
    bank_id = "agent-7"
    mem_id = uuid4()
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    edges = [
        EdgeResult(
            uuid=uuid4(),
            name="WORKS_AT",
            fact="old",
            valid_at=None,
            invalid_at="2024-02-01T00:00:00+00:00",
            source_uri=f"hindsight://bank/{bank_id}/memory/{mem_id}",
            group_id=bank_id,
        ),
        EdgeResult(
            uuid=uuid4(),
            name="LIVES_IN",
            fact="older",
            valid_at=None,
            invalid_at="2024-01-15T00:00:00+00:00",
            source_uri=f"hindsight://bank/{bank_id}/memory/{uuid4()}",
            group_id=bank_id,
        ),
    ]
    engine, _backend, _conn = _make_engine_mock(since_row=since, edges=edges)
    replay_result = SimpleNamespace(not_found=False, edges_replayed=2)
    engine.handle_graphiti_edge_invalidated = AsyncMock(return_value=replay_result)

    async def go():
        return await _poll_one_bank(
            engine, _make_client_mock(edges=edges), bank_id, internal=SimpleNamespace(internal=True)
        )

    result = asyncio.run(go())
    assert result.edges_replayed == 2
    assert engine.handle_graphiti_edge_invalidated.await_count == 2
    # Cursor advances to the max invalid_at in the page.
    assert result.new_since == datetime(2024, 2, 1, tzinfo=timezone.utc)


def test_poll_one_bank_filters_out_other_bank_edges():
    bank_id = "agent-7"
    edges = [
        # Local — should be replayed.
        EdgeResult(
            uuid=uuid4(),
            name="X",
            fact="",
            valid_at=None,
            invalid_at="2024-02-01T00:00:00+00:00",
            source_uri=f"hindsight://bank/{bank_id}/memory/{uuid4()}",
            group_id=bank_id,
        ),
        # Cross-bank — should be filtered out before replay.
        EdgeResult(
            uuid=uuid4(),
            name="X",
            fact="",
            valid_at=None,
            invalid_at="2024-02-15T00:00:00+00:00",
            source_uri=f"hindsight://bank/agent-OTHER/memory/{uuid4()}",
            group_id="agent-OTHER",
        ),
    ]
    engine, _backend, _conn = _make_engine_mock(since_row=None, edges=edges)
    engine.handle_graphiti_edge_invalidated = AsyncMock(return_value=SimpleNamespace(not_found=False))

    async def go():
        return await _poll_one_bank(
            engine, _make_client_mock(edges=edges), bank_id, internal=SimpleNamespace(internal=True)
        )

    result = asyncio.run(go())
    assert result.edges_replayed == 1
    assert engine.handle_graphiti_edge_invalidated.await_count == 1


def test_poll_one_bank_truncated_response_does_not_advance_cursor():
    """Deep-dive 5 §3.3 invariant: truncated=True means Graphiti is
    paging; keeping ``since`` fixed is the only way to resume.
    """
    bank_id = "agent-7"
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    edges = [
        EdgeResult(
            uuid=uuid4(),
            name="X",
            fact="",
            valid_at=None,
            invalid_at="2024-02-01T00:00:00+00:00",
            source_uri=f"hindsight://bank/{bank_id}/memory/{uuid4()}",
            group_id=bank_id,
        ),
    ]
    engine, _backend, _conn = _make_engine_mock(since_row=since, edges=edges)
    engine.handle_graphiti_edge_invalidated = AsyncMock(return_value=SimpleNamespace(not_found=False))

    client = MagicMock()
    client.list_invalidated_edges = AsyncMock(return_value=InvalidatedEdgesResponse(edges=edges, truncated=True))

    async def go():
        return await _poll_one_bank(engine, client, bank_id, internal=SimpleNamespace(internal=True))

    result = asyncio.run(go())
    assert result.truncated is True
    assert result.edges_replayed == 1
    # Cursor must NOT advance when truncated.
    assert result.new_since == since


def test_poll_one_bank_first_run_uses_epoch():
    """No row in the poller state table → since=epoch (deep-dive 5 §3.3)."""
    engine, _backend, _conn = _make_engine_mock(since_row=None, edges=[])
    engine.handle_graphiti_edge_invalidated = AsyncMock()

    client = MagicMock()
    client.list_invalidated_edges = AsyncMock(return_value=InvalidatedEdgesResponse(edges=[], truncated=False))

    captured: dict[str, Any] = {}

    original = client.list_invalidated_edges.side_effect

    async def capturing_call(*args, **kwargs):
        captured["kwargs"] = kwargs
        return await original(*args, **kwargs)

    client.list_invalidated_edges = AsyncMock(side_effect=capturing_call)

    async def go():
        return await _poll_one_bank(engine, client, "agent-7", internal=SimpleNamespace(internal=True))

    asyncio.run(go())
    assert captured["kwargs"]["since"] == datetime(1970, 1, 1, tzinfo=timezone.utc)


def test_poll_one_bank_engine_method_failure_does_not_abort_batch():
    bank_id = "agent-7"
    edges = [
        EdgeResult(
            uuid=uuid4(),
            name="X",
            fact="",
            valid_at=None,
            invalid_at="2024-02-01T00:00:00+00:00",
            source_uri=f"hindsight://bank/{bank_id}/memory/{uuid4()}",
            group_id=bank_id,
        ),
        EdgeResult(
            uuid=uuid4(),
            name="X",
            fact="",
            valid_at=None,
            invalid_at="2024-02-15T00:00:00+00:00",
            source_uri=f"hindsight://bank/{bank_id}/memory/{uuid4()}",
            group_id=bank_id,
        ),
    ]
    engine, _backend, _conn = _make_engine_mock(since_row=None, edges=edges)

    call_count = 0

    async def flaky(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("DB went away")
        return SimpleNamespace(not_found=False)

    engine.handle_graphiti_edge_invalidated = AsyncMock(side_effect=flaky)

    async def go():
        return await _poll_one_bank(
            engine, _make_client_mock(edges=edges), bank_id, internal=SimpleNamespace(internal=True)
        )

    result = asyncio.run(go())
    assert result.edges_replayed == 1
    assert result.errors == 1


def test_poll_one_bank_client_error_keeps_cursor_and_writes_diagnostic():
    engine, backend, _conn = _make_engine_mock(since_row=None, edges=[])
    engine.handle_graphiti_edge_invalidated = AsyncMock()

    client = MagicMock()
    client.list_invalidated_edges = AsyncMock(side_effect=GraphitiClientError("HTTP error: 503 Service Unavailable"))

    async def go():
        return await _poll_one_bank(engine, client, "agent-7", internal=SimpleNamespace(internal=True))

    result = asyncio.run(go())
    assert result.errors == 1
    assert "503" in (result.error_message or "")
    engine.handle_graphiti_edge_invalidated.assert_not_awaited()


def test_poll_one_bank_unparseable_invalid_at_counted_as_error():
    bank_id = "agent-7"
    edges = [
        EdgeResult(
            uuid=uuid4(),
            name="X",
            fact="",
            valid_at=None,
            invalid_at="not-a-date",
            source_uri=f"hindsight://bank/{bank_id}/memory/{uuid4()}",
            group_id=bank_id,
        ),
    ]
    engine, _backend, _conn = _make_engine_mock(since_row=None, edges=edges)
    engine.handle_graphiti_edge_invalidated = AsyncMock()

    async def go():
        return await _poll_one_bank(
            engine, _make_client_mock(edges=edges), bank_id, internal=SimpleNamespace(internal=True)
        )

    result = asyncio.run(go())
    assert result.edges_replayed == 0
    assert result.errors == 1
    engine.handle_graphiti_edge_invalidated.assert_not_awaited()


# ---------------------------------------------------------------------------
# Worker loop tests
# ---------------------------------------------------------------------------


def test_run_graphiti_backflow_poller_skips_banks_without_polling_flag():
    """A bank with the flag off must not be queried at all — saving the
    roundtrip cost when only one of N banks needs the fallback.
    """
    engine = MagicMock()
    _wire_engine_backend(engine)
    internal = SimpleNamespace(internal=True)
    engine.list_banks = AsyncMock(
        return_value=[
            {"bank_id": "bank-A"},
            {"bank_id": "bank-B"},
        ]
    )

    async def resolve(bank_id, ctx):
        if bank_id == "bank-A":
            return SimpleNamespace(
                graphiti_backflow_polling_enabled=False,
                graphiti_group_id="bank-A",
            )
        return SimpleNamespace(
            graphiti_backflow_polling_enabled=True,
            graphiti_group_id="bank-B",
        )

    engine._config_resolver = MagicMock()
    engine._config_resolver.resolve_full_config = AsyncMock(side_effect=resolve)

    client_factory_calls: list[str] = []
    captured_banks: list[str] = []
    stop = asyncio.Event()

    async def client_factory(cfg):
        client = MagicMock()
        client_factory_calls.append(cfg.graphiti_group_id)

        async def failing(**kwargs):
            captured_banks.append(cfg.graphiti_group_id)
            stop.set()  # stop after the first call
            raise RuntimeError("stop here")

        client.list_invalidated_edges = AsyncMock(side_effect=failing)
        return client

    async def go():
        await run_graphiti_backflow_poller(
            engine,
            poll_interval_seconds=60,
            client_factory=client_factory,
            stop_event=stop,
        )

    asyncio.run(go())
    assert client_factory_calls == ["bank-B"]
    assert captured_banks == ["bank-B"]


def test_run_graphiti_backflow_poller_skips_banks_without_group_id():
    engine = MagicMock()
    _wire_engine_backend(engine)
    engine.list_banks = AsyncMock(return_value=[{"bank_id": "bank-C"}])
    engine._config_resolver = MagicMock()
    engine._config_resolver.resolve_full_config = AsyncMock(
        return_value=SimpleNamespace(
            graphiti_backflow_polling_enabled=True,
            graphiti_group_id="",  # not federated
        )
    )

    stop = asyncio.Event()
    resolve_calls: list[str] = []

    original_resolve = engine._config_resolver.resolve_full_config

    async def resolve_capture(bank_id, ctx):
        resolve_calls.append(bank_id)
        stop.set()  # stop after the first config resolve
        return await original_resolve(bank_id, ctx)

    engine._config_resolver.resolve_full_config = AsyncMock(side_effect=resolve_capture)

    async def go():
        await run_graphiti_backflow_poller(engine, poll_interval_seconds=60, stop_event=stop)

    asyncio.run(go())
    # resolve called once for bank-C, but the worker skipped it
    # (no group_id) so no client_factory / no client.list_invalidated_edges.
    assert resolve_calls == ["bank-C"]


def test_run_graphiti_backflow_poller_rejects_nonpositive_interval():
    with pytest.raises(ValueError, match="poll_interval_seconds must be > 0"):
        asyncio.run(run_graphiti_backflow_poller(MagicMock(), poll_interval_seconds=0))
