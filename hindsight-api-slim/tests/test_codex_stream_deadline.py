"""Regression tests for the Codex per-request deadline and stream size guard (issue #3898).

Before the fix ``CodexLLM`` never read the configured timeout at all — the factory
did not pass one and the class did not have the attribute — and it fetched the SSE
body with a buffering ``client.post()`` under a hardcoded 120 s httpx timeout. That
timeout is per socket read, so a backend that wedges into runaway generation and
keeps emitting deltas never trips it: the client read one such response for ~830 s
(~12 MB) until the backend itself closed the connection, holding the consolidation
slot for the whole time.

These tests run a real local SSE server rather than mocking the client, because the whole
defect lives in the interaction between a still-flowing socket and the timeout that
was supposed to bound it — a mocked response cannot express "bytes keep arriving".
"""

import asyncio
import time
from unittest.mock import patch

import aiohttp
import pytest

from hindsight_api.engine.aiohttp_session import UpstreamHTTPError
from hindsight_api.engine.providers.codex_llm import (
    _MAX_SSE_BODY_CHARS,
    CodexLLM,
    CodexRunawayStreamError,
)
from hindsight_api.engine.remote_retry import is_transient_remote_error

pytestmark = pytest.mark.asyncio


def _delta_chunk(payload: str) -> bytes:
    return b'event: response.text.delta\ndata: {"delta": "' + payload.encode() + b'"}\n\n'


async def _sse_server(chunks_per_second: float, body: bytes | None = None, *, finite: bool = False):
    """Serve one endless (or fixed) SSE body; returns (server, base_url)."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while await reader.readline() not in (b"\r\n", b"\n", b""):
            pass
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
        try:
            if finite:
                assert body is not None
                writer.write(b"%x\r\n" % len(body) + body + b"\r\n0\r\n\r\n")
                await writer.drain()
                return
            chunk = _delta_chunk("x" * 4000)
            while True:
                writer.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                await writer.drain()
                await asyncio.sleep(1 / chunks_per_second)
        except (ConnectionResetError, BrokenPipeError):
            return
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"


def _build_llm(base_url: str, timeout: float | None) -> CodexLLM:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url=base_url,
            model="gpt-5.4-mini",
            timeout=timeout,
        )


async def test_configured_timeout_bounds_a_runaway_stream():
    """A stream that never stops delivering bytes is abandoned at the deadline.

    The failure this guards is silent: with a per-read timeout the call simply
    never returns, so assert on elapsed wall time, not just on the exception.
    """
    server, base_url = await _sse_server(chunks_per_second=4)
    llm = _build_llm(base_url, timeout=2.0)
    try:
        started = time.monotonic()
        with pytest.raises(CodexRunawayStreamError) as excinfo:
            await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
        elapsed = time.monotonic() - started
        assert "2s deadline" in str(excinfo.value)
        assert elapsed < 10.0, f"call ran {elapsed:.1f}s despite a 2s deadline"
    finally:
        await llm.cleanup()
        server.close()


async def test_deadline_also_bounds_the_tool_call_path():
    """``call_with_tools`` is a second request path — reflect runs through it."""
    server, base_url = await _sse_server(chunks_per_second=4)
    llm = _build_llm(base_url, timeout=2.0)
    try:
        started = time.monotonic()
        with pytest.raises(CodexRunawayStreamError):
            await llm.call_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                max_retries=0,
            )
        elapsed = time.monotonic() - started
        assert elapsed < 10.0, f"tool call ran {elapsed:.1f}s despite a 2s deadline"
    finally:
        await llm.cleanup()
        server.close()


async def test_runaway_is_retryable_transport_error():
    """The deadline must land in the existing retry path, not escape as a hard failure."""
    assert issubclass(CodexRunawayStreamError, aiohttp.ClientError)
    assert is_transient_remote_error(CodexRunawayStreamError("runaway"))


async def test_oversized_body_is_abandoned_before_the_deadline():
    """The size guard cuts off a fast runaway stream without waiting out the clock."""
    server, base_url = await _sse_server(chunks_per_second=2000)
    # Generous deadline: only the byte ceiling can end this call.
    llm = _build_llm(base_url, timeout=120.0)
    try:
        started = time.monotonic()
        with pytest.raises(CodexRunawayStreamError) as excinfo:
            await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
        elapsed = time.monotonic() - started
        assert str(_MAX_SSE_BODY_CHARS) in str(excinfo.value)
        assert elapsed < 60.0, f"size guard did not fire ({elapsed:.1f}s)"
    finally:
        await llm.cleanup()
        server.close()


async def test_normal_response_still_parses():
    """The stream rewrite must not change what an ordinary short response returns."""
    body = _delta_chunk("hello ") + _delta_chunk("world") + b"data: [DONE]\n\n"
    server, base_url = await _sse_server(chunks_per_second=1, body=body, finite=True)
    llm = _build_llm(base_url, timeout=30.0)
    try:
        assert (await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)).content == "hello world"
    finally:
        await llm.cleanup()
        server.close()


async def test_error_body_is_readable_after_streaming():
    """Non-200 responses still expose ``.text`` — callers log it and classify on it."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while await reader.readline() not in (b"\r\n", b"\n", b""):
            pass
        payload = b'{"error": "bad request detail"}'
        writer.write(
            b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(payload)
        )
        writer.write(payload)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    base_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    llm = _build_llm(base_url, timeout=30.0)
    try:
        with pytest.raises(UpstreamHTTPError) as excinfo:
            await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
        assert excinfo.value.status_code == 400
        assert "bad request detail" in excinfo.value.body
    finally:
        await llm.cleanup()
        server.close()


async def test_lines_split_across_chunks_are_reassembled():
    """SSE lines arrive in arbitrary chunks; the reader must rejoin them exactly.

    Covers a ``\\r\\n`` split between two chunks, a multi-byte character split
    mid-sequence, and one line far longer than aiohttp's own line-reader buffer
    (which would raise rather than yield it).
    """
    long_delta = "y" * 200_000
    body = (
        b'event: response.text.delta\r\ndata: {"delta": "caf\xc3\xa9 "}\r\n\r\n'
        + b'event: response.text.delta\ndata: {"delta": "'
        + long_delta.encode()
        + b'"}\n\n'
        + b"data: [DONE]\n\n"
    )
    # Cut points: inside the first "\r\n", inside the "é" (0xc3 | 0xa9), inside the long line.
    first_crlf = body.index(b"\r\n") + 1
    inside_e_acute = body.index(b"\xc3") + 1
    inside_long_line = body.index(b"yyyy") + 50_000
    cuts = [0, first_crlf, inside_e_acute, inside_long_line, len(body)]

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while await reader.readline() not in (b"\r\n", b"\n", b""):
            pass
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
        for start, end in zip(cuts, cuts[1:]):
            piece = body[start:end]
            writer.write(b"%x\r\n" % len(piece) + piece + b"\r\n")
            await writer.drain()
            await asyncio.sleep(0.02)  # separate reads on the client side
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    base_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    llm = _build_llm(base_url, timeout=30.0)
    try:
        result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
        assert result.content == "café " + long_delta
    finally:
        await llm.cleanup()
        server.close()
