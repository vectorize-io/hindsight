"""Tests for LiteLLM embeddings retry mechanism, error recovery, and dimensions configuration."""

from unittest.mock import patch

import httpx
import pytest

from hindsight_api.config import (
    ENV_EMBEDDINGS_LITELLM_DIMENSIONS,
    ENV_EMBEDDINGS_PROVIDER,
    clear_config_cache,
)
from hindsight_api.engine.embeddings import (
    EmbeddingRetryPolicy,
    LiteLLMEmbeddings,
    create_embeddings_from_env,
)


@pytest.mark.asyncio
async def test_litellm_embeddings_dimension_override_skips_probe() -> None:
    """When dimensions is configured, initialize() must skip the network probe entirely."""
    embeddings = LiteLLMEmbeddings(
        api_base="http://test-litellm:4000",
        model="custom-model",
        dimensions=768,
    )

    with patch.object(httpx.Client, "post") as mock_post:
        await embeddings.initialize()
        mock_post.assert_not_called()

    assert embeddings.dimension == 768


@pytest.mark.asyncio
async def test_litellm_embeddings_initialize_probe_success() -> None:
    """When dimensions is not set, initialize() probes the endpoint and detects vector dimension."""
    embeddings = LiteLLMEmbeddings(
        api_base="http://test-litellm:4000",
        model="text-embedding-3-small",
        retry_policy=EmbeddingRetryPolicy(initial_backoff=0.01),
    )

    mock_resp = httpx.Response(
        200,
        json={"data": [{"embedding": [0.1] * 1024, "index": 0}]},
        request=httpx.Request("POST", "http://test-litellm:4000/embeddings"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp) as mock_post:
        await embeddings.initialize()
        mock_post.assert_called_once()

    assert embeddings.dimension == 1024


@pytest.mark.asyncio
async def test_litellm_embeddings_probe_retries_on_500_and_recovers() -> None:
    """Probe should retry on transient 500 errors and succeed once proxy is ready."""
    embeddings = LiteLLMEmbeddings(
        api_base="http://test-litellm:4000",
        model="text-embedding-3-small",
        retry_policy=EmbeddingRetryPolicy(max_retries=3, initial_backoff=0.01),
    )

    req = httpx.Request("POST", "http://test-litellm:4000/embeddings")
    err_resp = httpx.Response(500, request=req)
    ok_resp = httpx.Response(
        200,
        json={"data": [{"embedding": [0.1] * 1536, "index": 0}]},
        request=req,
    )

    with patch.object(httpx.Client, "post", side_effect=[err_resp, err_resp, ok_resp]) as mock_post:
        await embeddings.initialize()
        assert mock_post.call_count == 3

    assert embeddings.dimension == 1536


@pytest.mark.asyncio
async def test_litellm_embeddings_probe_retries_on_connect_error_and_recovers() -> None:
    """Probe should retry on connection error (e.g. proxy starting up) and succeed."""
    embeddings = LiteLLMEmbeddings(
        api_base="http://test-litellm:4000",
        model="text-embedding-3-small",
        retry_policy=EmbeddingRetryPolicy(max_retries=3, initial_backoff=0.01),
    )

    req = httpx.Request("POST", "http://test-litellm:4000/embeddings")
    ok_resp = httpx.Response(
        200,
        json={"data": [{"embedding": [0.1] * 1536, "index": 0}]},
        request=req,
    )

    with patch.object(
        httpx.Client,
        "post",
        side_effect=[httpx.ConnectError("Connection refused", request=req), ok_resp],
    ) as mock_post:
        await embeddings.initialize()
        assert mock_post.call_count == 2

    assert embeddings.dimension == 1536


@pytest.mark.asyncio
async def test_litellm_embeddings_probe_exhausts_retries_and_raises() -> None:
    """Probe should raise RuntimeError after exhausting max_retries."""
    embeddings = LiteLLMEmbeddings(
        api_base="http://test-litellm:4000",
        model="text-embedding-3-small",
        retry_policy=EmbeddingRetryPolicy(max_retries=2, initial_backoff=0.01),
    )

    req = httpx.Request("POST", "http://test-litellm:4000/embeddings")
    err_resp = httpx.Response(500, request=req)

    with patch.object(httpx.Client, "post", side_effect=[err_resp, err_resp, err_resp]) as mock_post:
        with pytest.raises(RuntimeError, match="Failed to connect to LiteLLM proxy"):
            await embeddings.initialize()
        assert mock_post.call_count == 3


def _ready_embeddings(dimensions: int | None, dim: int) -> LiteLLMEmbeddings:
    """A LiteLLMEmbeddings past initialize(), without touching the network."""
    embeddings = LiteLLMEmbeddings(
        api_base="http://test-litellm:4000",
        model="text-embedding-3-small",
        dimensions=dimensions,
        retry_policy=EmbeddingRetryPolicy(max_retries=2, initial_backoff=0.01),
    )
    embeddings._client = httpx.Client()
    embeddings._dimension = dim
    return embeddings


def _embed_response(dim: int, count: int) -> httpx.Response:
    req = httpx.Request("POST", "http://test-litellm:4000/embeddings")
    return httpx.Response(
        200,
        json={"data": [{"embedding": [0.1] * dim, "index": i} for i in range(count)]},
        request=req,
    )


def test_litellm_embeddings_encode_retries_on_503_and_succeeds() -> None:
    """encode() should retry transient 503 errors and recover."""
    embeddings = _ready_embeddings(dimensions=512, dim=512)

    req = httpx.Request("POST", "http://test-litellm:4000/embeddings")
    err_resp = httpx.Response(503, request=req)

    with patch.object(httpx.Client, "post", side_effect=[err_resp, _embed_response(512, 2)]) as mock_post:
        res = embeddings.encode(["hello", "world"])
        assert len(res) == 2
        assert len(res[0]) == 512
        assert mock_post.call_count == 2


def test_litellm_embeddings_encode_never_forwards_dimensions() -> None:
    """A declared width must not be sent upstream: proxied backends reject the field."""
    embeddings = _ready_embeddings(dimensions=512, dim=512)

    with patch.object(httpx.Client, "post", return_value=_embed_response(512, 1)) as mock_post:
        embeddings.encode(["hello"])

    payload = mock_post.call_args[1]["json"]
    assert "dimensions" not in payload
    assert payload == {"model": "text-embedding-3-small", "input": ["hello"]}


def test_litellm_embeddings_encode_rejects_wrong_declared_dimension() -> None:
    """A declared width that the proxy contradicts must fail loudly, not corrupt vectors."""
    embeddings = _ready_embeddings(dimensions=512, dim=512)

    with patch.object(httpx.Client, "post", return_value=_embed_response(1536, 1)):
        with pytest.raises(RuntimeError, match="declares 512 dimensions but"):
            embeddings.encode(["hello"])


def test_litellm_embeddings_encode_without_declared_dimension_is_unchecked() -> None:
    """Probe-detected dimensions need no second-guessing on the encode path."""
    embeddings = _ready_embeddings(dimensions=None, dim=1536)

    with patch.object(httpx.Client, "post", return_value=_embed_response(1536, 1)) as mock_post:
        res = embeddings.encode(["hello"])

    assert len(res[0]) == 1536
    assert "dimensions" not in mock_post.call_args[1]["json"]


@pytest.mark.asyncio
async def test_litellm_embeddings_probe_does_not_retry_client_error() -> None:
    """A 400 is a misconfiguration, not a cold proxy: fail fast without burning retries."""
    embeddings = LiteLLMEmbeddings(
        api_base="http://test-litellm:4000",
        model="text-embedding-3-small",
        retry_policy=EmbeddingRetryPolicy(max_retries=3, initial_backoff=0.01),
    )

    req = httpx.Request("POST", "http://test-litellm:4000/embeddings")
    bad_resp = httpx.Response(400, request=req)

    with patch.object(httpx.Client, "post", return_value=bad_resp) as mock_post:
        with pytest.raises(RuntimeError, match="Failed to connect to LiteLLM proxy"):
            await embeddings.initialize()
        assert mock_post.call_count == 1


@pytest.mark.asyncio
async def test_litellm_embeddings_probe_empty_data_names_the_env_var() -> None:
    """An empty data array is unrecoverable; the error must point at the escape hatch."""
    embeddings = LiteLLMEmbeddings(
        api_base="http://test-litellm:4000",
        model="text-embedding-3-small",
        retry_policy=EmbeddingRetryPolicy(initial_backoff=0.01),
    )

    req = httpx.Request("POST", "http://test-litellm:4000/embeddings")
    empty_resp = httpx.Response(200, json={"data": []}, request=req)

    with patch.object(httpx.Client, "post", return_value=empty_resp):
        with pytest.raises(RuntimeError, match=ENV_EMBEDDINGS_LITELLM_DIMENSIONS):
            await embeddings.initialize()


def test_create_embeddings_from_env_with_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_embeddings_from_env() should parse HINDSIGHT_API_EMBEDDINGS_LITELLM_DIMENSIONS."""
    monkeypatch.setenv(ENV_EMBEDDINGS_PROVIDER, "litellm")
    monkeypatch.setenv(ENV_EMBEDDINGS_LITELLM_DIMENSIONS, "384")

    clear_config_cache()
    try:
        embeddings = create_embeddings_from_env()
        assert isinstance(embeddings, LiteLLMEmbeddings)
        assert embeddings.dimensions == 384
    finally:
        clear_config_cache()
