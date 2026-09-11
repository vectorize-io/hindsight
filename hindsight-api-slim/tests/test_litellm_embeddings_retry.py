"""Tests for LiteLLM embeddings retry mechanism, error recovery, and dimensions configuration."""

from typing import Any

import pytest
from aiohttp import web

from hindsight_api.config import (
    ENV_EMBEDDINGS_LITELLM_DIMENSIONS,
    ENV_EMBEDDINGS_PROVIDER,
    clear_config_cache,
)
from hindsight_api.engine.embeddings import (
    LiteLLMEmbeddings,
    RetryPolicy,
    create_embeddings_from_env,
)
from tests.aiohttp_stub import stub_server

# A step in a scripted proxy: an HTTP status with no body, a JSON body served with 200,
# or DISCONNECT to drop the connection without answering (a proxy still starting up).
DISCONNECT = "disconnect"


class _ScriptedProxy:
    """Answers each POST /embeddings with the next scripted step; repeats the last one."""

    def __init__(self, *steps: Any) -> None:
        self._steps = list(steps)
        self.payloads: list[dict[str, Any]] = []

    @property
    def calls(self) -> int:
        return len(self.payloads)

    async def handle(self, request: web.Request) -> web.StreamResponse:
        assert request.path == "/embeddings"
        self.payloads.append(await request.json())
        step = self._steps[min(len(self.payloads) - 1, len(self._steps) - 1)]
        if step == DISCONNECT:
            assert request.transport is not None
            request.transport.close()
            return web.Response(status=200)
        if isinstance(step, int):
            return web.Response(status=step)
        return web.json_response(step)


def _embed_body(dim: int, count: int) -> dict[str, Any]:
    return {"data": [{"embedding": [0.1] * dim, "index": i} for i in range(count)]}


async def test_litellm_embeddings_dimension_override_skips_probe() -> None:
    """When dimensions is configured, initialize() must skip the network probe entirely."""
    proxy = _ScriptedProxy(_embed_body(768, 1))
    async with stub_server(proxy.handle) as base_url:
        embeddings = LiteLLMEmbeddings(api_base=base_url, model="custom-model", dimensions=768)
        await embeddings.initialize()

    assert proxy.calls == 0
    assert embeddings.dimension == 768


async def test_litellm_embeddings_initialize_probe_success() -> None:
    """When dimensions is not set, initialize() probes the endpoint and detects vector dimension."""
    proxy = _ScriptedProxy(_embed_body(1024, 1))
    async with stub_server(proxy.handle) as base_url:
        embeddings = LiteLLMEmbeddings(
            api_base=base_url,
            model="text-embedding-3-small",
            retry_policy=RetryPolicy(initial_backoff=0.01),
        )
        await embeddings.initialize()

    assert proxy.calls == 1
    assert embeddings.dimension == 1024


async def test_litellm_embeddings_probe_retries_on_500_and_recovers() -> None:
    """Probe should retry on transient 500 errors and succeed once proxy is ready."""
    proxy = _ScriptedProxy(500, 500, _embed_body(1536, 1))
    async with stub_server(proxy.handle) as base_url:
        embeddings = LiteLLMEmbeddings(
            api_base=base_url,
            model="text-embedding-3-small",
            retry_policy=RetryPolicy(max_retries=3, initial_backoff=0.01),
        )
        await embeddings.initialize()

    assert proxy.calls == 3
    assert embeddings.dimension == 1536


async def test_litellm_embeddings_probe_retries_on_connect_error_and_recovers() -> None:
    """Probe should retry on a dropped connection (e.g. proxy starting up) and succeed."""
    proxy = _ScriptedProxy(DISCONNECT, _embed_body(1536, 1))
    async with stub_server(proxy.handle) as base_url:
        embeddings = LiteLLMEmbeddings(
            api_base=base_url,
            model="text-embedding-3-small",
            retry_policy=RetryPolicy(max_retries=3, initial_backoff=0.01),
        )
        await embeddings.initialize()

    assert proxy.calls == 2
    assert embeddings.dimension == 1536


async def test_litellm_embeddings_probe_exhausts_retries_and_raises() -> None:
    """Probe should raise RuntimeError after exhausting max_retries."""
    proxy = _ScriptedProxy(500)
    async with stub_server(proxy.handle) as base_url:
        embeddings = LiteLLMEmbeddings(
            api_base=base_url,
            model="text-embedding-3-small",
            retry_policy=RetryPolicy(max_retries=2, initial_backoff=0.01),
        )
        with pytest.raises(RuntimeError, match="Failed to connect to LiteLLM proxy"):
            await embeddings.initialize()

    assert proxy.calls == 3


async def _ready_embeddings(base_url: str, dimensions: int | None, dim: int) -> LiteLLMEmbeddings:
    """A LiteLLMEmbeddings past initialize(), without spending a probe request."""
    embeddings = LiteLLMEmbeddings(
        api_base=base_url,
        model="text-embedding-3-small",
        dimensions=dimensions if dimensions is not None else dim,
        retry_policy=RetryPolicy(max_retries=2, initial_backoff=0.01),
    )
    # A declared width skips the probe; restore the configured value afterwards.
    await embeddings.initialize()
    embeddings.dimensions = dimensions
    embeddings._dimension = dim
    return embeddings


async def test_litellm_embeddings_encode_retries_on_503_and_succeeds() -> None:
    """encode() should retry transient 503 errors and recover."""
    proxy = _ScriptedProxy(503, _embed_body(512, 2))
    async with stub_server(proxy.handle) as base_url:
        embeddings = await _ready_embeddings(base_url, dimensions=512, dim=512)
        res = await embeddings.encode(["hello", "world"])

    assert len(res) == 2
    assert len(res[0]) == 512
    assert proxy.calls == 2


async def test_litellm_embeddings_encode_never_forwards_dimensions() -> None:
    """A declared width must not be sent upstream: proxied backends reject the field."""
    proxy = _ScriptedProxy(_embed_body(512, 1))
    async with stub_server(proxy.handle) as base_url:
        embeddings = await _ready_embeddings(base_url, dimensions=512, dim=512)
        await embeddings.encode(["hello"])

    payload = proxy.payloads[-1]
    assert "dimensions" not in payload
    assert payload == {"model": "text-embedding-3-small", "input": ["hello"]}


async def test_litellm_embeddings_encode_rejects_wrong_declared_dimension() -> None:
    """A declared width that the proxy contradicts must fail loudly, not corrupt vectors."""
    proxy = _ScriptedProxy(_embed_body(1536, 1))
    async with stub_server(proxy.handle) as base_url:
        embeddings = await _ready_embeddings(base_url, dimensions=512, dim=512)
        with pytest.raises(RuntimeError, match="declares 512 dimensions but"):
            await embeddings.encode(["hello"])


async def test_litellm_embeddings_encode_without_declared_dimension_is_unchecked() -> None:
    """Probe-detected dimensions need no second-guessing on the encode path."""
    proxy = _ScriptedProxy(_embed_body(1536, 1))
    async with stub_server(proxy.handle) as base_url:
        embeddings = await _ready_embeddings(base_url, dimensions=None, dim=1536)
        res = await embeddings.encode(["hello"])

    assert len(res[0]) == 1536
    assert "dimensions" not in proxy.payloads[-1]


async def test_litellm_embeddings_probe_does_not_retry_client_error() -> None:
    """A 400 is a misconfiguration, not a cold proxy: fail fast without burning retries."""
    proxy = _ScriptedProxy(400)
    async with stub_server(proxy.handle) as base_url:
        embeddings = LiteLLMEmbeddings(
            api_base=base_url,
            model="text-embedding-3-small",
            retry_policy=RetryPolicy(max_retries=3, initial_backoff=0.01),
        )
        with pytest.raises(RuntimeError, match="Failed to connect to LiteLLM proxy"):
            await embeddings.initialize()

    assert proxy.calls == 1


async def test_litellm_embeddings_probe_empty_data_names_the_env_var() -> None:
    """An empty data array is unrecoverable; the error must point at the escape hatch."""
    proxy = _ScriptedProxy({"data": []})
    async with stub_server(proxy.handle) as base_url:
        embeddings = LiteLLMEmbeddings(
            api_base=base_url,
            model="text-embedding-3-small",
            retry_policy=RetryPolicy(initial_backoff=0.01),
        )
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
