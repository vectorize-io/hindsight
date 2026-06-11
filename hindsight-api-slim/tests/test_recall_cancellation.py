"""Tests for cooperative recall cancellation on client disconnect (issue #2122).

Covers three layers:
- the ``CancellationToken`` primitive,
- ``RequestContext`` integration (the carrier the engine checks at stage
  boundaries),
- the HTTP ``_cancel_on_client_disconnect`` driver, including an ASGI-level
  end-to-end that proves a disconnect aborts staged work and surfaces 499.
"""

import asyncio

import pytest
from fastapi import FastAPI, HTTPException, Request

from hindsight_api.api.http import (
    _CLIENT_CLOSED_REQUEST_STATUS_CODE,
    _cancel_on_client_disconnect,
)
from hindsight_api.cancellation import CancellationToken, OperationCancelledError
from hindsight_api.models import RequestContext

_TEST_TIMEOUT_SECONDS = 3.0


class _FakeRequest:
    """Minimal Request stand-in whose disconnect state is driven by an Event."""

    def __init__(self, disconnected: asyncio.Event) -> None:
        self._disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self._disconnected.is_set()


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


# --- HTTP disconnect driver -----------------------------------------------------


async def test_driver_attaches_and_restores_token():
    ctx = RequestContext()
    disconnected = asyncio.Event()

    async with _cancel_on_client_disconnect(_FakeRequest(disconnected), ctx, poll_interval=0) as token:
        assert ctx.cancellation is token

    # Restored to the previous (None) token once the scope exits.
    assert ctx.cancellation is None


async def test_driver_restores_previous_token_when_nested():
    outer = CancellationToken()
    ctx = RequestContext(cancellation=outer)
    disconnected = asyncio.Event()

    async with _cancel_on_client_disconnect(_FakeRequest(disconnected), ctx, poll_interval=0):
        assert ctx.cancellation is not outer

    assert ctx.cancellation is outer


async def test_driver_cancels_staged_work_on_disconnect():
    """A worker that checks the token between stages aborts once the client leaves."""
    ctx = RequestContext()
    disconnected = asyncio.Event()

    async def staged_work() -> str:
        # Stand-in for the recall pipeline: checkpoint, do a slice of work, repeat.
        for _ in range(1000):
            ctx.raise_if_cancelled()
            await asyncio.sleep(0.005)
        return "done"

    async def run() -> None:
        async with _cancel_on_client_disconnect(_FakeRequest(disconnected), ctx, poll_interval=0):
            work = asyncio.create_task(staged_work())
            await asyncio.sleep(0.02)
            disconnected.set()
            await work

    with pytest.raises(OperationCancelledError) as exc:
        await asyncio.wait_for(run(), timeout=_TEST_TIMEOUT_SECONDS)
    assert exc.value.reason == "client disconnected"
    assert ctx.cancellation is None


async def test_driver_returns_work_result_when_client_stays():
    ctx = RequestContext()
    disconnected = asyncio.Event()  # never set

    async def staged_work() -> str:
        for _ in range(3):
            ctx.raise_if_cancelled()
            await asyncio.sleep(0)
        return "done"

    async with _cancel_on_client_disconnect(_FakeRequest(disconnected), ctx, poll_interval=0):
        result = await staged_work()

    assert result == "done"


# --- ASGI end-to-end: disconnect -> aborted work -> 499 -------------------------


async def test_asgi_disconnect_aborts_work_and_returns_499():
    app = FastAPI()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @app.post("/recall")
    async def recall(http_request: Request):
        ctx = RequestContext()
        try:
            async with _cancel_on_client_disconnect(http_request, ctx, poll_interval=0):
                started.set()
                while True:  # staged pipeline: checkpoint then a slice of work
                    ctx.raise_if_cancelled()
                    await asyncio.sleep(0.005)
        except OperationCancelledError as e:
            cancelled.set()
            raise HTTPException(status_code=_CLIENT_CLOSED_REQUEST_STATUS_CODE, detail=e.reason)
        return {"ok": True}

    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await started.wait()
        return {"type": "http.disconnect"}

    messages = []

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/recall",
        "query_string": b"",
        "headers": [],
    }

    await asyncio.wait_for(app(scope, receive, send), timeout=_TEST_TIMEOUT_SECONDS)

    assert cancelled.is_set()
    response_start = next(m for m in messages if m["type"] == "http.response.start")
    assert response_start["status"] == _CLIENT_CLOSED_REQUEST_STATUS_CODE
