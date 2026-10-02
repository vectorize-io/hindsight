"""Classify real interrupted HTTP responses, not only nested socket exceptions."""

import asyncio
import errno
import json

import aiohttp
import pytest

from hindsight_api.engine.tei_retry import is_retryable_tei_transport_error


@pytest.mark.asyncio
@pytest.mark.parametrize("response_kind", ["truncated-body", "disconnect-before-headers"])
async def test_interrupted_responses_are_retryable(response_kind: str) -> None:
    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            if response_kind == "truncated-body":
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b"Content-Length: 100\r\nConnection: close\r\n\r\n[0.1]"
                )
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
            with pytest.raises(aiohttp.ClientError) as caught:
                # TEI inference uses POST. aiohttp's built-in stale-connection
                # retry applies only to idempotent HTTP methods, so it cannot
                # recover this response for the provider's own retry loop.
                async with session.post(f"http://127.0.0.1:{port}/embed", json={"inputs": ["test"]}) as response:
                    await response.json()
        assert is_retryable_tei_transport_error(caught.value)
    finally:
        server.close()
        await server.wait_closed()


def test_permanent_transport_and_json_errors_still_fail_fast() -> None:
    assert not is_retryable_tei_transport_error(OSError(errno.EACCES, "permission denied"))
    assert not is_retryable_tei_transport_error(aiohttp.InvalidURL("not an HTTP URL"))
    assert not is_retryable_tei_transport_error(json.JSONDecodeError("invalid JSON", "bad", 0))
