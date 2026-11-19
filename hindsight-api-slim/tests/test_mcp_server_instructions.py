"""Per-bank MCP server instructions, sent in the initialize response.

Each test drives a real MCP ``initialize`` over streamable HTTP, through the same
ASGI middleware the API mounts. That matters: the instructions are written onto a
private attribute of the SDK's session (see ``BankServerInstructionsMiddleware``),
so only a real handshake proves they reach the client.
"""

import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from unittest.mock import patch

import httpx
import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from hindsight_api.api.mcp import MCPMiddleware, create_mcp_server
from hindsight_api.config import MCP_SERVER_INSTRUCTIONS_MAX_CHARS


async def _not_found(scope, receive, send):
    await send({"type": "http.response.start", "status": 404, "headers": []})
    await send({"type": "http.response.body", "body": b""})


@asynccontextmanager
async def _mcp_client(memory, *, stateless: bool):
    """An httpx client routed to MCP servers built in the requested transport mode."""
    single = create_mcp_server(memory, multi_bank=False).http_app(path="/", stateless_http=stateless)
    multi = create_mcp_server(memory, multi_bank=True).http_app(path="/", stateless_http=stateless)
    app = MCPMiddleware(_not_found, memory, multi_bank_app=multi, single_bank_app=single)
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(single.router.lifespan_context(single))
        await stack.enter_async_context(multi.router.lifespan_context(multi))
        client = await stack.enter_async_context(
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        )
        yield client


async def _initialize(http_client: httpx.AsyncClient, url: str):
    async with streamable_http_client(url, http_client=http_client) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            return await session.initialize()


@asynccontextmanager
async def _bank(memory, request_context, instructions: str | None):
    bank_id = f"mcp-instructions-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    try:
        if instructions is not None:
            await memory._config_resolver.update_bank_config(
                bank_id, {"mcp_server_instructions": instructions}, request_context
            )
        yield bank_id
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.parametrize("stateless", [True, False], ids=["stateless", "stateful"])
async def test_single_bank_initialize_carries_the_bank_instructions(memory, request_context, stateless):
    text = "Read the document 'house-rules' first. Tag new facts with 'inbox'."
    async with _bank(memory, request_context, text) as bank_id, _mcp_client(memory, stateless=stateless) as client:
        result = await _initialize(client, f"http://test/mcp/{bank_id}/")

    assert result.instructions == text


@pytest.mark.asyncio
@pytest.mark.parametrize("stateless", [True, False], ids=["stateless", "stateful"])
async def test_each_bank_session_gets_only_its_own_instructions(memory, request_context, stateless):
    async with (
        _bank(memory, request_context, "Instructions for bank A.") as bank_a,
        _bank(memory, request_context, "Instructions for bank B.") as bank_b,
        _bank(memory, request_context, None) as bank_c,
        _mcp_client(memory, stateless=stateless) as client,
    ):
        a = await _initialize(client, f"http://test/mcp/{bank_a}/")
        b = await _initialize(client, f"http://test/mcp/{bank_b}/")
        c = await _initialize(client, f"http://test/mcp/{bank_c}/")
        a_again = await _initialize(client, f"http://test/mcp/{bank_a}/")

    assert a.instructions == "Instructions for bank A."
    assert b.instructions == "Instructions for bank B."
    assert c.instructions is None
    assert a_again.instructions == "Instructions for bank A."


@pytest.mark.asyncio
async def test_instructions_follow_a_config_change_on_the_next_connection(memory, request_context):
    async with (
        _bank(memory, request_context, "First version.") as bank_id,
        _mcp_client(memory, stateless=True) as client,
    ):
        first = await _initialize(client, f"http://test/mcp/{bank_id}/")
        await memory._config_resolver.update_bank_config(
            bank_id, {"mcp_server_instructions": "Second version."}, request_context
        )
        second = await _initialize(client, f"http://test/mcp/{bank_id}/")
        await memory._config_resolver.update_bank_config(bank_id, {"mcp_server_instructions": None}, request_context)
        cleared = await _initialize(client, f"http://test/mcp/{bank_id}/")

    assert first.instructions == "First version."
    assert second.instructions == "Second version."
    assert cleared.instructions is None


@pytest.mark.asyncio
async def test_blank_instructions_send_none(memory, request_context):
    async with _bank(memory, request_context, "   \n  ") as bank_id, _mcp_client(memory, stateless=True) as client:
        result = await _initialize(client, f"http://test/mcp/{bank_id}/")

    assert result.instructions is None


@pytest.mark.asyncio
async def test_multi_bank_endpoint_sends_no_bank_instructions(memory, request_context):
    """The multi-bank endpoint reaches every bank, so no one bank's text describes it."""
    async with _bank(memory, request_context, "Only for the single-bank endpoint.") as bank_id:
        async with _mcp_client(memory, stateless=True) as client:
            client.headers["X-Bank-Id"] = bank_id
            result = await _initialize(client, "http://test/mcp/")

    assert result.instructions is None


@pytest.mark.asyncio
async def test_failed_config_lookup_still_connects(memory, request_context):
    async with _bank(memory, request_context, "Unreachable.") as bank_id, _mcp_client(memory, stateless=True) as client:
        with patch.object(memory._config_resolver, "get_bank_config", side_effect=RuntimeError("db down")):
            result = await _initialize(client, f"http://test/mcp/{bank_id}/")

    assert result.instructions is None
    assert result.serverInfo.name == "hindsight-mcp-server"


@pytest.mark.asyncio
async def test_instructions_longer_than_the_cap_are_rejected(memory, request_context):
    async with _bank(memory, request_context, None) as bank_id:
        await memory._config_resolver.update_bank_config(
            bank_id, {"mcp_server_instructions": "x" * MCP_SERVER_INSTRUCTIONS_MAX_CHARS}, request_context
        )
        with pytest.raises(ValueError, match="mcp_server_instructions must be at most"):
            await memory._config_resolver.update_bank_config(
                bank_id, {"mcp_server_instructions": "x" * (MCP_SERVER_INSTRUCTIONS_MAX_CHARS + 1)}, request_context
            )


@pytest.mark.asyncio
async def test_non_string_instructions_are_rejected(memory, request_context):
    async with _bank(memory, request_context, None) as bank_id:
        with pytest.raises(ValueError, match="mcp_server_instructions must be a string"):
            await memory._config_resolver.update_bank_config(
                bank_id, {"mcp_server_instructions": {"text": "not a string"}}, request_context
            )
