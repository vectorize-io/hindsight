"""Regression tests for transient HTTP handling in the remote TEI embeddings client.

The upstream is a real aiohttp server (``tests/aiohttp_stub.py``), so the retries below
are driven by what actually reaches the client's transport: dropped connections, 429s,
4xx bodies.
"""

import errno
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from hindsight_api.engine import embeddings as embeddings_module
from hindsight_api.engine.aiohttp_session import UpstreamHTTPError
from hindsight_api.engine.embeddings import RemoteTEIEmbeddings
from hindsight_api.engine.tei_retry import TEI_KEEPALIVE_EXPIRY_SECONDS
from tests.aiohttp_stub import Handler, stub_server


@asynccontextmanager
async def _tei(handler: Handler, **kwargs: Any) -> AsyncIterator[RemoteTEIEmbeddings]:
    """A ready-to-encode provider pointed at ``handler``; its session is closed afterwards."""
    async with stub_server(handler) as base_url:
        embeddings = RemoteTEIEmbeddings(base_url=base_url, **kwargs)
        # Skip initialize(): these tests count /embed attempts, not the startup probe.
        embeddings._initialized = True
        embeddings._dimension = 2
        try:
            yield embeddings
        finally:
            await embeddings._session.close()


def _drop_connection(request: web.Request) -> web.Response:
    """Close the socket without answering — the client sees a dropped connection."""
    assert request.transport is not None
    request.transport.close()
    return web.Response()


async def test_dropped_connection_retries_then_succeeds() -> None:
    attempts = 0

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return _drop_connection(request)
        return web.json_response([[0.1, 0.2]])

    async with _tei(handler, max_retries=3, retry_delay=0) as embeddings:
        assert await embeddings.encode(["text"]) == [[0.1, 0.2]]

    assert attempts == 2


class _FlakySession:
    """Raises ``error`` from the first ``failures`` requests, then defers to the real session."""

    def __init__(self, real: aiohttp.ClientSession, error: BaseException, failures: int = 1) -> None:
        self._real = real
        self._error = error
        self._failures = failures
        self.calls = 0

    def request(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls <= self._failures:
            raise self._error
        return self._real.request(*args, **kwargs)


async def test_bad_file_descriptor_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare OSError from a pooled socket that died while idle is retried."""

    async def handler(request: web.Request) -> web.StreamResponse:
        return web.json_response([[0.3, 0.4]])

    async with _tei(handler, max_retries=3, retry_delay=0) as embeddings:
        flaky = _FlakySession(embeddings._session.get(), OSError(errno.EBADF, "Bad file descriptor"))
        monkeypatch.setattr(embeddings._session, "get", lambda: flaky)

        assert await embeddings.encode(["text"]) == [[0.3, 0.4]]

    assert flaky.calls == 2


async def test_non_retryable_os_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: web.Request) -> web.StreamResponse:
        return web.json_response([[0.3, 0.4]])

    async with _tei(handler, max_retries=3, retry_delay=0) as embeddings:
        flaky = _FlakySession(embeddings._session.get(), OSError(errno.EACCES, "Permission denied"))
        monkeypatch.setattr(embeddings._session, "get", lambda: flaky)

        with pytest.raises(OSError):
            await embeddings.encode(["text"])

    assert flaky.calls == 1


async def test_persistent_dropped_connection_exhausts_retry_budget() -> None:
    attempts = 0

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal attempts
        attempts += 1
        return _drop_connection(request)

    async with _tei(handler, max_retries=2, retry_delay=0) as embeddings:
        with pytest.raises(RuntimeError, match="TEI embedding request failed") as exc_info:
            await embeddings.encode(["text"])

    assert attempts == 3
    assert isinstance(exc_info.value.__context__, aiohttp.ClientConnectionError)


async def test_retry_on_too_many_requests() -> None:
    """TEI's 429 overload response should use the transient retry budget."""
    attempts = 0

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return web.json_response({"error": "Model is overloaded"}, status=429)
        return web.json_response([[0.1, 0.2]])

    async with _tei(handler, max_retries=3, retry_delay=0) as embeddings:
        assert await embeddings.encode(["text"]) == [[0.1, 0.2]]

    assert attempts == 2


async def test_retry_after_header_reaches_the_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 429's own headers, not a guess, decide how long to wait."""
    seen: list[str | None] = []

    def spy_delay(headers: Mapping[str, str], fallback_delay: float, *, request_timeout: float) -> float:
        seen.append(headers.get("Retry-After"))
        return 0.0

    monkeypatch.setattr(embeddings_module, "tei_retry_delay", spy_delay)
    attempts = 0

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return web.json_response({"error": "busy"}, status=429, headers={"Retry-After": "7"})
        return web.json_response([[0.1, 0.2]])

    async with _tei(handler, max_retries=3, retry_delay=0) as embeddings:
        assert await embeddings.encode(["text"]) == [[0.1, 0.2]]

    assert seen == ["7"]


async def test_other_client_errors_fail_fast() -> None:
    """Non-429 4xx responses should not consume the retry budget."""
    attempts = 0

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal attempts
        attempts += 1
        return web.json_response({"error": "invalid input"}, status=400)

    async with _tei(handler, max_retries=3, retry_delay=0) as embeddings:
        with pytest.raises(RuntimeError, match="TEI embedding request failed") as exc_info:
            await embeddings.encode(["text"])

    assert attempts == 1
    # The upstream's reason survives into the error the caller sees.
    assert "invalid input" in str(exc_info.value)


async def test_persistent_too_many_requests_exhausts_retry_budget() -> None:
    """A persistent overload should make exactly max_retries + 1 attempts."""
    attempts = 0

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal attempts
        attempts += 1
        return web.json_response({"error": "Model is overloaded"}, status=429, headers={"Retry-After": "Infinity"})

    async with _tei(handler, max_retries=2, retry_delay=0) as embeddings:
        with pytest.raises(RuntimeError, match="TEI embedding request failed") as exc_info:
            await embeddings.encode(["text"])

    assert attempts == 3
    assert isinstance(exc_info.value.__context__, UpstreamHTTPError)
    assert exc_info.value.__context__.status_code == 429


async def test_initialize_probes_info_and_detects_dimension() -> None:
    paths: list[str] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        paths.append(request.path)
        if request.path == "/info":
            return web.json_response({"model_id": "BAAI/bge-small-en-v1.5"})
        return web.json_response([[0.0, 0.1, 0.2]])

    async with stub_server(handler) as base_url:
        embeddings = RemoteTEIEmbeddings(base_url=base_url)
        try:
            await embeddings.initialize()
            assert embeddings.dimension == 3
            assert embeddings._model_id == "BAAI/bge-small-en-v1.5"
        finally:
            await embeddings._session.close()

    assert paths == ["/info", "/embed"]


async def test_initialize_failure_is_reported_as_a_connection_error() -> None:
    async def handler(request: web.Request) -> web.StreamResponse:
        return web.json_response({"error": "forbidden"}, status=403)

    async with stub_server(handler) as base_url:
        embeddings = RemoteTEIEmbeddings(base_url=base_url)
        try:
            with pytest.raises(RuntimeError, match="Failed to connect to TEI server"):
                await embeddings.initialize()
        finally:
            await embeddings._session.close()


async def test_encode_before_initialize_is_rejected() -> None:
    embeddings = RemoteTEIEmbeddings(base_url="http://tei.invalid")
    with pytest.raises(RuntimeError, match="not initialized"):
        await embeddings.encode(["text"])


def test_default_tei_batch_size_is_32() -> None:
    """Unset env keeps 32 texts per /embed request — TEI's own --max-client-batch-size."""
    import os

    from hindsight_api.config import HindsightConfig

    saved_provider = os.environ.get("HINDSIGHT_API_LLM_PROVIDER")
    saved_batch = os.environ.pop("HINDSIGHT_API_EMBEDDINGS_TEI_BATCH_SIZE", None)
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    try:
        assert HindsightConfig.from_env().embeddings_tei_batch_size == 32
    finally:
        if saved_batch is not None:
            os.environ["HINDSIGHT_API_EMBEDDINGS_TEI_BATCH_SIZE"] = saved_batch
        if saved_provider is None:
            os.environ.pop("HINDSIGHT_API_LLM_PROVIDER", None)
        else:
            os.environ["HINDSIGHT_API_LLM_PROVIDER"] = saved_provider


def test_tei_batch_size_env_var_reaches_the_client() -> None:
    """The configured batch size is what encode() splits on, not the hardcoded 32."""
    import os

    from hindsight_api.config import HindsightConfig, clear_config_cache
    from hindsight_api.engine.embeddings import create_embeddings_from_env

    saved = {
        key: os.environ.get(key)
        for key in (
            "HINDSIGHT_API_LLM_PROVIDER",
            "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
            "HINDSIGHT_API_EMBEDDINGS_TEI_URL",
            "HINDSIGHT_API_EMBEDDINGS_TEI_BATCH_SIZE",
        )
    }
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "tei"
    os.environ["HINDSIGHT_API_EMBEDDINGS_TEI_URL"] = "http://localhost:8080"
    os.environ["HINDSIGHT_API_EMBEDDINGS_TEI_BATCH_SIZE"] = "128"
    clear_config_cache()
    try:
        assert HindsightConfig.from_env().embeddings_tei_batch_size == 128
        assert create_embeddings_from_env().batch_size == 128
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_config_cache()


@pytest.mark.parametrize(
    ("timeout", "expected_keepalive"),
    [(5.0, 5.0), (60.0, TEI_KEEPALIVE_EXPIRY_SECONDS)],
)
async def test_session_drops_idle_sockets_before_a_proxy_would(timeout: float, expected_keepalive: float) -> None:
    # Idle pooled sockets are retired after min(timeout, TEI_KEEPALIVE_EXPIRY_SECONDS), so a
    # load balancer that silently closes idle connections does not hand us a dead one.
    embeddings = RemoteTEIEmbeddings(base_url="http://tei.invalid", timeout=timeout)
    session = embeddings._session.get()
    try:
        assert session.connector is not None
        assert session.connector._keepalive_timeout == expected_keepalive
        # Per-phase, not whole-request: a long body must not be cut off by `total`.
        assert session.timeout.total is None
        assert session.timeout.sock_read == timeout
    finally:
        await embeddings._session.close()
