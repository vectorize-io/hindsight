"""Tests for cooperative recall/reflect cancellation on client disconnect (#2122).

Layers covered:
- the ``CancellationToken`` primitive,
- ``RequestContext`` integration (the carrier the engine checks at boundaries),
- ``run_cancellable_on_disconnect`` (reads the scope token, maps cancel -> 499),
- ``ClientDisconnectCancellationMiddleware``, including the critical regression
  test that it still fires **behind a BaseHTTPMiddleware** — the exact condition
  under which ``Request.is_disconnected()`` silently never fires and the original
  #2127 implementation did nothing.
"""

import asyncio

import pytest
from fastapi import FastAPI, HTTPException, Request

from hindsight_api.api.disconnect import (
    SCOPE_CANCELLATION_TOKEN,
    ClientDisconnectCancellationMiddleware,
    _should_monitor,
    get_scope_cancellation_token,
)
from hindsight_api.api.http import (
    _CLIENT_CLOSED_REQUEST_STATUS_CODE,
    run_cancellable_on_disconnect,
    run_task_cancellable_on_disconnect,
)
from hindsight_api.cancellation import CancellationToken, OperationCancelledError
from hindsight_api.models import RequestContext

_TEST_TIMEOUT_SECONDS = 3.0


# --- CancellationToken primitive ------------------------------------------------


def test_token_starts_uncancelled():
    token = CancellationToken()
    assert token.cancelled is False
    token.raise_if_cancelled()  # no-op


def test_token_raises_after_cancel():
    token = CancellationToken()
    token.cancel("client disconnected")
    assert token.cancelled is True
    assert token.reason == "client disconnected"
    with pytest.raises(OperationCancelledError) as exc:
        token.raise_if_cancelled()
    assert exc.value.reason == "client disconnected"


def test_token_cancel_is_idempotent_first_reason_wins():
    token = CancellationToken()
    token.cancel("first")
    token.cancel("second")
    assert token.reason == "first"


async def test_token_wait_unblocks_on_cancel():
    token = CancellationToken()

    async def cancel_soon():
        await asyncio.sleep(0)
        token.cancel("done")

    asyncio.create_task(cancel_soon())
    await asyncio.wait_for(token.wait(), timeout=_TEST_TIMEOUT_SECONDS)
    assert token.cancelled is True


# --- RequestContext integration -------------------------------------------------


def test_request_context_check_is_noop_without_token():
    ctx = RequestContext()
    assert ctx.cancellation is None
    ctx.raise_if_cancelled()  # must not raise


def test_request_context_raises_when_token_fired():
    token = CancellationToken()
    token.cancel("client disconnected")
    ctx = RequestContext(cancellation=token)
    with pytest.raises(OperationCancelledError):
        ctx.raise_if_cancelled()


# --- _should_monitor path gating ------------------------------------------------


def _scope(path: str, method: str = "POST") -> dict:
    return {"type": "http", "path": path, "method": method}


def test_should_monitor_recall_reflect_and_retain():
    assert _should_monitor(_scope("/v1/default/banks/b/memories/recall")) is True
    assert _should_monitor(_scope("/v1/default/banks/b/reflect")) is True
    # The retain POST: a sync retain runs inline and must be cancellable (#4526).
    assert _should_monitor(_scope("/v1/default/banks/b/memories")) is True
    # Same path, different operation: DELETE clears a bank and is not a retain.
    assert _should_monitor(_scope("/v1/default/banks/b/memories", method="DELETE")) is False
    assert _should_monitor(_scope("/health", method="GET")) is False
    assert _should_monitor(_scope("/v1/default/banks/b/memories/recall/extra")) is False
    # The upload path stays out: its body is streamed, not a small JSON doc.
    assert _should_monitor(_scope("/v1/default/banks/b/files/retain")) is False


# --- run_cancellable_on_disconnect ----------------------------------------------


class _ScopeRequest:
    """Minimal Request stand-in exposing a .scope dict."""

    def __init__(self, scope: dict) -> None:
        self.scope = scope


async def test_run_cancellable_returns_result_when_no_token():
    ctx = RequestContext()
    req = _ScopeRequest({})  # middleware didn't attach a token

    async def work() -> str:
        ctx.raise_if_cancelled()
        return "ok"

    result = await run_cancellable_on_disconnect(req, ctx, work(), operation="recall", bank_id="b1")
    assert result == "ok"


async def test_run_cancellable_wires_scope_token_and_maps_to_499():
    token = CancellationToken()
    ctx = RequestContext()
    req = _ScopeRequest({SCOPE_CANCELLATION_TOKEN: token})

    async def work() -> str:
        # token fires mid-flight; engine checkpoint raises
        for _ in range(1000):
            ctx.raise_if_cancelled()
            await asyncio.sleep(0.005)
        return "done"

    async def fire_soon():
        await asyncio.sleep(0.02)
        token.cancel("client disconnected")

    asyncio.create_task(fire_soon())
    with pytest.raises(HTTPException) as exc:
        await asyncio.wait_for(
            run_cancellable_on_disconnect(req, ctx, work(), operation="reflect", bank_id="b1"),
            timeout=_TEST_TIMEOUT_SECONDS,
        )
    assert exc.value.status_code == _CLIENT_CLOSED_REQUEST_STATUS_CODE
    assert exc.value.detail == "client disconnected"
    # the engine's carrier was wired to the scope token
    assert ctx.cancellation is token


# --- Middleware ASGI integration (the regression that matters) ------------------


def _build_app(*, with_base_http_middleware: bool) -> tuple[FastAPI, asyncio.Event]:
    """Reflect-like app guarded by the disconnect middleware.

    with_base_http_middleware reproduces the production setup where a
    @app.middleware("http") (BaseHTTPMiddleware) sits between uvicorn and the
    route and breaks Request.is_disconnected().
    """
    app = FastAPI()
    cancelled = asyncio.Event()

    @app.post("/v1/default/banks/{bank_id}/reflect")
    async def reflect(bank_id: str, http_request: Request):
        ctx = RequestContext()

        async def work():
            try:
                while True:
                    ctx.raise_if_cancelled()
                    await asyncio.sleep(0.005)
            except OperationCancelledError:
                cancelled.set()
                raise

        return await run_cancellable_on_disconnect(http_request, ctx, work(), operation="reflect", bank_id=bank_id)

    if with_base_http_middleware:

        @app.middleware("http")
        async def noop(request, call_next):
            return await call_next(request)

    # Installed last -> outermost -> owns the raw receive (as in create_app).
    app.add_middleware(ClientDisconnectCancellationMiddleware)
    return app, cancelled


async def _drive_disconnect(app: FastAPI) -> list:
    """Send a request, then http.disconnect once the handler is running."""
    started = asyncio.Event()
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            started.set()
            return {"type": "http.request", "body": b"{}", "more_body": False}
        await started.wait()
        await asyncio.sleep(0.05)
        return {"type": "http.disconnect"}

    messages = []

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/default/banks/b1/reflect",
        "raw_path": b"/v1/default/banks/b1/reflect",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=_TEST_TIMEOUT_SECONDS)
    return messages


async def test_middleware_cancels_behind_base_http_middleware():
    """THE regression test: disconnect cancellation must fire even with a
    BaseHTTPMiddleware in the stack (where is_disconnected() is broken)."""
    app, cancelled = _build_app(with_base_http_middleware=True)
    messages = await _drive_disconnect(app)
    assert cancelled.is_set(), "work was not cancelled behind BaseHTTPMiddleware"
    start = next(m for m in messages if m["type"] == "http.response.start")
    assert start["status"] == _CLIENT_CLOSED_REQUEST_STATUS_CODE


async def test_middleware_cancels_without_base_http_middleware():
    app, cancelled = _build_app(with_base_http_middleware=False)
    messages = await _drive_disconnect(app)
    assert cancelled.is_set()
    start = next(m for m in messages if m["type"] == "http.response.start")
    assert start["status"] == _CLIENT_CLOSED_REQUEST_STATUS_CODE


async def test_middleware_passes_through_unmonitored_paths():
    """Non-recall/reflect paths get no token and no receive proxying."""
    seen_scope = {}

    async def app(scope, receive, send):
        seen_scope.update(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = ClientDisconnectCancellationMiddleware(app)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(m):
        sent.append(m)

    await mw({"type": "http", "path": "/health"}, receive, send)
    assert SCOPE_CANCELLATION_TOKEN not in seen_scope
    assert any(m["type"] == "http.response.start" for m in sent)


async def test_middleware_completes_normally_when_no_disconnect():
    app, cancelled = _build_app(with_base_http_middleware=True)
    # never disconnects; the route would loop forever, so give it a token and
    # trip it via a normal completion path instead: hit an unmonitored no-op.
    token_seen = {}

    async def inner(scope, receive, send):
        token_seen["t"] = get_scope_cancellation_token(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    mw = ClientDisconnectCancellationMiddleware(inner)

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    sent = []

    async def send(m):
        sent.append(m)

    scope = {"type": "http", "path": "/v1/default/banks/b1/reflect"}
    await asyncio.wait_for(mw(scope, receive, send), timeout=_TEST_TIMEOUT_SECONDS)
    # monitored path => token attached, request completed 200
    assert isinstance(token_seen["t"], CancellationToken)
    assert any(m.get("status") == 200 for m in sent if m["type"] == "http.response.start")


# --- run_task_cancellable_on_disconnect (sync retain, #4526) --------------------


async def test_run_task_cancellable_returns_result_when_no_token():
    async def work() -> str:
        return "ok"

    result = await run_task_cancellable_on_disconnect(_ScopeRequest({}), work(), operation="retain", bank_id="b1")
    assert result == "ok"


async def test_run_task_cancellable_kills_an_unco_operative_await():
    """The point of the hard variant: no checkpoint is reached, yet the work stops.

    A sync retain spends its wall time parked on the LLM semaphore and the provider
    response, so a cooperative checkpoint never gets a turn. The task must be
    cancelled outright, and its ``finally`` must still run.
    """
    token = CancellationToken()
    req = _ScopeRequest({SCOPE_CANCELLATION_TOKEN: token})
    unwound = asyncio.Event()

    async def work() -> str:
        try:
            await asyncio.sleep(3600)  # one long await, no checkpoints
            return "committed"
        finally:
            unwound.set()

    async def fire_soon():
        await asyncio.sleep(0.02)
        token.cancel("client disconnected")

    asyncio.create_task(fire_soon())
    with pytest.raises(HTTPException) as exc:
        await asyncio.wait_for(
            run_task_cancellable_on_disconnect(req, work(), operation="retain", bank_id="b1"),
            timeout=_TEST_TIMEOUT_SECONDS,
        )
    assert exc.value.status_code == _CLIENT_CLOSED_REQUEST_STATUS_CODE
    assert exc.value.detail == "client disconnected"
    assert unwound.is_set(), "the cancelled task was abandoned instead of awaited"


async def test_run_task_cancellable_propagates_work_errors():
    token = CancellationToken()
    req = _ScopeRequest({SCOPE_CANCELLATION_TOKEN: token})

    async def work() -> str:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await run_task_cancellable_on_disconnect(req, work(), operation="retain", bank_id="b1")


# --- The whole route, end to end (#4526) ----------------------------------------


async def test_sync_retain_route_answers_499_and_stops_the_work(memory, monkeypatch):
    """A sync retain whose client hangs up must stop, not run to completion.

    Drives the real app at the ASGI layer (the only level where a disconnect can be
    delivered) with the extraction stubbed out as one long await — the shape of a
    retain parked on the LLM semaphore, which is where the 1880s run in #4526 sat.
    """
    from hindsight_api.api import create_app

    app = create_app(memory, initialize_memory=False)
    started = asyncio.Event()
    unwound = asyncio.Event()

    async def never_finishes(*args, **kwargs):
        started.set()
        try:
            await asyncio.sleep(3600)
        finally:
            unwound.set()

    monkeypatch.setattr(memory, "retain_batch_async", never_finishes)

    body = b'{"items": [{"content": "Alice works at Google"}], "async": false}'
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await started.wait()
        return {"type": "http.disconnect"}

    messages = []

    async def send(message):
        messages.append(message)

    path = "/v1/default/banks/disconnect-probe/memories"
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10.0)

    start = next(m for m in messages if m["type"] == "http.response.start")
    assert start["status"] == _CLIENT_CLOSED_REQUEST_STATUS_CODE
    assert unwound.is_set(), "the abandoned retain kept running after the client left"
