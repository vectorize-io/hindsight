"""Regression tests for transient HTTP handling in the remote TEI embeddings client."""

import httpx
import pytest

from hindsight_api.engine.embeddings import RemoteTEIEmbeddings


def test_connect_timeout_retries_then_succeeds() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectTimeout("connection timed out")
        return httpx.Response(200, json=[[0.1, 0.2]])

    embeddings = RemoteTEIEmbeddings(
        base_url="http://localhost:8080",
        max_retries=3,
        retry_delay=0,
    )
    embeddings._client = httpx.Client(transport=httpx.MockTransport(handler))

    assert embeddings.encode(["text"]) == [[0.1, 0.2]]
    assert attempts == 2


def test_persistent_connect_timeout_exhausts_retry_budget() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectTimeout("connection timed out")

    embeddings = RemoteTEIEmbeddings(
        base_url="http://localhost:8080",
        max_retries=2,
        retry_delay=0,
    )
    embeddings._client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(RuntimeError, match="TEI embedding request failed") as exc_info:
        embeddings.encode(["text"])

    assert attempts == 3
    assert isinstance(exc_info.value.__context__, httpx.ConnectTimeout)


def test_retry_on_too_many_requests() -> None:
    """TEI's 429 overload response should use the transient retry budget."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": "Model is overloaded"})
        return httpx.Response(200, json=[[0.1, 0.2]])

    embeddings = RemoteTEIEmbeddings(
        base_url="http://localhost:8080",
        max_retries=3,
        retry_delay=0,
    )
    embeddings._client = httpx.Client(transport=httpx.MockTransport(handler))

    assert embeddings.encode(["text"]) == [[0.1, 0.2]]
    assert attempts == 2


def test_other_client_errors_fail_fast() -> None:
    """Non-429 4xx responses should not consume the retry budget."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"error": "invalid input"})

    embeddings = RemoteTEIEmbeddings(
        base_url="http://localhost:8080",
        max_retries=3,
        retry_delay=0,
    )
    embeddings._client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(RuntimeError, match="TEI embedding request failed"):
        embeddings.encode(["text"])

    assert attempts == 1


def test_persistent_too_many_requests_exhausts_retry_budget() -> None:
    """A persistent overload should make exactly max_retries + 1 attempts."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            429,
            request=request,
            headers={"Retry-After": "Infinity"},
            json={"error": "Model is overloaded"},
        )

    embeddings = RemoteTEIEmbeddings(
        base_url="http://localhost:8080",
        max_retries=2,
        retry_delay=0,
    )
    embeddings._client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(RuntimeError, match="TEI embedding request failed") as exc_info:
        embeddings.encode(["text"])

    assert attempts == 3
    assert isinstance(exc_info.value.__context__, httpx.HTTPStatusError)
