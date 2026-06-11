import asyncio

import pytest
from fastapi import FastAPI, Request

from hindsight_api.api.http import _CLIENT_CLOSED_REQUEST_STATUS_CODE, _run_until_client_disconnect

_TEST_TIMEOUT_SECONDS = 3.0


class _ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_run_until_client_disconnect_returns_completed_result():
    async def work() -> str:
        return "done"

    result = await _run_until_client_disconnect(_ConnectedRequest(), work(), poll_interval=0)

    assert result == "done"


@pytest.mark.asyncio
async def test_fastapi_asgi_disconnect_cancels_wrapped_work():
    app = FastAPI()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def work() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    @app.post("/recall")
    async def recall(request: Request):
        await _run_until_client_disconnect(request, work(), poll_interval=0)
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
    response_start = next(message for message in messages if message["type"] == "http.response.start")
    assert response_start["status"] == _CLIENT_CLOSED_REQUEST_STATUS_CODE
