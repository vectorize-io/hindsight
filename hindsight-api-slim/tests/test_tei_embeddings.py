"""Regression tests for transient HTTP handling in the remote TEI embeddings client."""

import errno

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


def test_bad_file_descriptor_retries_then_succeeds() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EBADF, "Bad file descriptor")
        return httpx.Response(200, json=[[0.3, 0.4]])

    embeddings = RemoteTEIEmbeddings(
        base_url="http://localhost:8080",
        max_retries=3,
        retry_delay=0,
    )
    embeddings._client = httpx.Client(transport=httpx.MockTransport(handler))

    assert embeddings.encode(["text"]) == [[0.3, 0.4]]
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


def test_default_tei_batch_size_is_32() -> None:
    """Unset env keeps the historical 32 texts per /embed request."""
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
