"""The on-loop query path of the TEI embedder: same vectors as the thread path, and it steps
aside (returns None) whenever the thread path must handle the request."""

import asyncio

import httpx
import pytest
from aiohttp import web

from hindsight_api.engine.embeddings import RemoteTEIEmbeddings
from hindsight_api.engine.retain import embedding_utils


def _vector_for(text: str) -> list[float]:
    return [float(len(text)), float(sum(map(ord, text)) % 997), 0.5]


async def _start_tei(status: int = 200):
    seen: list[list[str]] = []

    async def embed(request: web.Request) -> web.Response:
        inputs = (await request.json())["inputs"]
        seen.append(inputs)
        if status != 200:
            return web.Response(status=status)
        return web.json_response([_vector_for(t) for t in inputs])

    app = web.Application()
    app.router.add_post("/embed", embed)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}", seen


def _embedder(base_url: str, *, query_prefix: str = "") -> RemoteTEIEmbeddings:
    e = RemoteTEIEmbeddings(base_url=base_url, query_prefix=query_prefix)
    e._initialized = True
    return e


@pytest.mark.asyncio
async def test_async_query_matches_the_thread_path_and_applies_the_query_prefix():
    runner, url, seen = await _start_tei()
    try:
        e = _embedder(url, query_prefix="query: ")
        on_loop = await e.aencode_query(["where is the cache"])
        threaded = await asyncio.get_running_loop().run_in_executor(None, e.encode_query, ["where is the cache"])
        assert on_loop == threaded == [_vector_for("query: where is the cache")]
        assert seen == [["query: where is the cache"], ["query: where is the cache"]]
        await e._aio_session.close()
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_failed_attempt_declines_so_the_thread_path_can_retry():
    runner, url, _ = await _start_tei(status=503)
    try:
        e = _embedder(url)
        assert await e.aencode_query(["q"]) is None
        await e._aio_session.close()
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_declines_when_uninitialized_or_a_client_is_injected():
    e = RemoteTEIEmbeddings(base_url="http://127.0.0.1:9")
    assert await e.aencode_query(["q"]) is None  # not initialized
    e._initialized = True
    e._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[[1.0]])))
    assert await e.aencode_query(["q"]) is None  # tests inject a transport; honour it


@pytest.mark.asyncio
async def test_generate_embeddings_batch_uses_the_on_loop_path_only_for_queries(monkeypatch):
    calls: list[str] = []

    class Backend:
        dimension = 1

        async def aencode_query(self, texts):
            calls.append("async")
            return [[9.0] for _ in texts]

        def encode_query(self, texts):
            calls.append("thread-query")
            return [[1.0] for _ in texts]

        def encode_documents(self, texts):
            calls.append("thread-doc")
            return [[2.0] for _ in texts]

    assert await embedding_utils.generate_embeddings_batch(Backend(), ["q"], input_type="query") == [[9.0]]
    assert await embedding_utils.generate_embeddings_batch(Backend(), ["d"]) == [[2.0]]
    assert calls == ["async", "thread-doc"]


@pytest.mark.asyncio
async def test_on_loop_vectors_get_the_same_validation_as_the_thread_path():
    class WrongDimension:
        dimension = 3

        async def aencode_query(self, texts):
            return [[1.0] for _ in texts]

    with pytest.raises(Exception):
        await embedding_utils.generate_embeddings_batch(WrongDimension(), ["q"], input_type="query")
