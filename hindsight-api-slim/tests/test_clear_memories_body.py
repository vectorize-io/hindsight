"""A selection body must never turn into an unfiltered bank clear (#4337)."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from hindsight_api.api import create_app
from hindsight_api.engine.memory_engine import MemoryEngine


@pytest.fixture
def app() -> FastAPI:
    memory = MagicMock(spec=MemoryEngine)
    memory.audit_logger = None
    memory.delete_bank = AsyncMock()
    return create_app(memory, initialize_memory=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b'{"ids":["one"]}', b"{}", b"[]", b"null", b" ", b"invalid", b"\xff"])
@pytest.mark.parametrize("query", ["", "?type=world"])
async def test_nonempty_body_never_calls_delete(app, body: bytes, query: str) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.request("DELETE", "/v1/default/banks/bank/memories" + query, content=body)
    assert response.status_code == 400
    assert "does not accept a request body" in response.json()["detail"]
    app.state.memory.delete_bank.assert_not_awaited()


@pytest.mark.asyncio
async def test_streamed_body_is_rejected_without_reading_the_rest(app) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield b""
        yield b'{"ids":'
        pytest.fail("The rejected request must not be buffered or drained by the handler")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.request("DELETE", "/v1/default/banks/bank/memories", content=chunks())
    assert response.status_code == 400
    app.state.memory.delete_bank.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fact_type", [None, "world", "observation"])
async def test_empty_body_preserves_clear_semantics(app, fact_type: str | None) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.request(
            "DELETE",
            "/v1/default/banks/bank/memories",
            params={"type": fact_type} if fact_type else {},
            content=b"",
        )
    assert response.status_code == 200
    assert response.json()["success"] is True
    app.state.memory.delete_bank.assert_awaited_once()
    call = app.state.memory.delete_bank.await_args
    assert call.args == ("bank",)
    assert call.kwargs["fact_type"] == fact_type
    assert call.kwargs["delete_bank_profile"] is False
