"""NVIDIA NIM embeddings: OpenAI-shaped /embeddings plus a native input_type.

The stub tests run everywhere; the live test hits the real NVIDIA API and is skipped
unless NVIDIA_LIVE_API_KEY is set.
"""

import math
import os

import pytest
from aiohttp import web

from hindsight_api.engine.embeddings import NvidiaEmbeddings
from tests.aiohttp_stub import stub_server

LIVE_API_KEY = os.environ.get("NVIDIA_LIVE_API_KEY")


@pytest.mark.asyncio
async def test_nvidia_embeddings_wire_format_and_batching():
    """Documents go out as input_type=passage, queries as query; batches are split and re-ordered by index."""
    bodies: list[dict] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        assert request.path == "/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer nvapi-test"
        body = await request.json()
        bodies.append(body)
        # Reply out of order: the provider must restore input order from `index`.
        data = [{"index": i, "embedding": [float(text.split()[-1]), 0.0]} for i, text in enumerate(body["input"])]
        return web.json_response({"data": list(reversed(data))})

    async with stub_server(handler) as base_url:
        emb = NvidiaEmbeddings(
            api_key="nvapi-test", model="nvidia/m", base_url=base_url + "/v1/", dimensions=2, batch_size=2
        )
        await emb.initialize()
        docs = await emb.encode_documents(["doc 0", "doc 1", "doc 2"])
        queries = await emb.encode_query(["q 7"])

    assert emb.dimension == 2
    assert docs == [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    assert queries == [[7.0, 0.0]]
    doc_bodies = [b for b in bodies if b["input_type"] == "passage"]
    assert sorted(len(b["input"]) for b in doc_bodies) == [1, 2]
    assert [b for b in bodies if b["input_type"] == "query"] == [
        {
            "model": "nvidia/m",
            "input": ["q 7"],
            "input_type": "query",
            "dimensions": 2,
            "encoding_format": "float",
            "truncate": "END",
        }
    ]


@pytest.mark.asyncio
async def test_nvidia_embeddings_rejects_misaligned_response():
    async def handler(request: web.Request) -> web.StreamResponse:
        return web.json_response({"data": [{"index": 0, "embedding": [1.0]}]})

    async with stub_server(handler) as base_url:
        emb = NvidiaEmbeddings(api_key="k", base_url=base_url, dimensions=1)
        await emb.initialize()
        with pytest.raises(RuntimeError, match="1:1 alignment"):
            await emb.encode_documents(["a", "b"])


def test_factory_builds_nvidia_and_skips_generic_prefixes(monkeypatch):
    """`nvidia` sends a native input_type, so the generic asymmetric prefixes must not apply."""
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.embeddings import create_embeddings_from_env

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "nvidia")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_NVIDIA_DIMENSIONS", "2048")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_NVIDIA_BATCH_SIZE", "7")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX", "query: ")
    clear_config_cache()
    try:
        emb = create_embeddings_from_env()
    finally:
        clear_config_cache()
    assert isinstance(emb, NvidiaEmbeddings)
    assert (emb.dimensions, emb.batch_size, emb.query_prefix) == (2048, 7, "")


def test_factory_requires_api_key(monkeypatch):
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.embeddings import create_embeddings_from_env

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "nvidia")
    monkeypatch.delenv("HINDSIGHT_API_EMBEDDINGS_NVIDIA_API_KEY", raising=False)
    clear_config_cache()
    try:
        with pytest.raises(ValueError, match="HINDSIGHT_API_EMBEDDINGS_NVIDIA_API_KEY"):
            create_embeddings_from_env()
    finally:
        clear_config_cache()


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.mark.skipif(not LIVE_API_KEY, reason="NVIDIA_LIVE_API_KEY not set")
@pytest.mark.asyncio
async def test_live_nvidia_embeddings_query_matches_relevant_passage():
    assert LIVE_API_KEY is not None
    emb = NvidiaEmbeddings(api_key=LIVE_API_KEY)
    await emb.initialize()
    docs = await emb.encode_documents(["Paris is the capital of France.", "Bananas are yellow."])
    (query,) = await emb.encode_query(["What is the capital of France?"])
    assert all(len(v) == emb.dimension == 1024 for v in [*docs, query])
    assert _cosine(query, docs[0]) > _cosine(query, docs[1])
