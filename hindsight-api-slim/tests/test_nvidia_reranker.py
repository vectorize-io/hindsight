"""NVIDIA NIM reranker: native wire format (per-model path, nested query, rankings[].logit).

The stub tests run everywhere; the live test hits the real NVIDIA API and is skipped
unless NVIDIA_LIVE_API_KEY is set.
"""

import os

import pytest
from aiohttp import web

from hindsight_api.engine.cross_encoder import NvidiaCrossEncoder
from tests.aiohttp_stub import stub_server

LIVE_API_KEY = os.environ.get("NVIDIA_LIVE_API_KEY")


@pytest.mark.asyncio
async def test_nvidia_reranker_wire_format():
    """One request per distinct query, on the per-model path, scores mapped back by index."""
    requests: list[tuple[str, dict]] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        assert request.headers["Authorization"] == "Bearer nvapi-test"
        body = await request.json()
        requests.append((request.path, body))
        # Rank in reverse order so the index mapping is exercised.
        n = len(body["passages"])
        return web.json_response({"rankings": [{"index": i, "logit": float(i) - 5} for i in reversed(range(n))]})

    async with stub_server(handler) as base_url:
        encoder = NvidiaCrossEncoder(api_key="nvapi-test", model="nvidia/some-rerank", base_url=base_url + "/v1/")
        await encoder.initialize()
        scores = await encoder.predict([("q1", "a"), ("q2", "b"), ("q1", "c")])

    assert scores == [-5.0, -5.0, -4.0]
    assert sorted(path for path, _ in requests) == ["/v1/retrieval/nvidia/some-rerank/reranking"] * 2
    bodies = {body["query"]["text"]: body for _, body in requests}
    assert bodies["q1"] == {
        "model": "nvidia/some-rerank",
        "query": {"text": "q1"},
        "passages": [{"text": "a"}, {"text": "c"}],
    }
    assert bodies["q2"]["passages"] == [{"text": "b"}]


@pytest.mark.asyncio
async def test_nvidia_reranker_splits_a_query_into_batch_size_requests():
    """The API caps passages per request, so one query's passages go out in batch_size slices."""
    batch_lengths: list[int] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        batch_lengths.append(len(body["passages"]))
        # Score is the passage's own number, so the mapping back across slices is checkable.
        return web.json_response(
            {"rankings": [{"index": i, "logit": float(p["text"])} for i, p in enumerate(body["passages"])]}
        )

    async with stub_server(handler) as base_url:
        encoder = NvidiaCrossEncoder(api_key="k", model="nvidia/m", base_url=base_url, batch_size=2, max_concurrent=2)
        await encoder.initialize()
        scores = await encoder.predict([("q", str(i)) for i in range(5)])

    assert sorted(batch_lengths) == [1, 2, 2]
    assert scores == [0.0, 1.0, 2.0, 3.0, 4.0]


@pytest.mark.skipif(not LIVE_API_KEY, reason="NVIDIA_LIVE_API_KEY not set")
@pytest.mark.asyncio
async def test_live_nvidia_reranker_ranks_relevant_passage_first():
    assert LIVE_API_KEY is not None
    # batch_size=2 so the live call also exercises the split across requests.
    encoder = NvidiaCrossEncoder(api_key=LIVE_API_KEY, batch_size=2)
    await encoder.initialize()
    query = "What is the capital of France?"
    scores = await encoder.predict(
        [(query, "Bananas are yellow."), (query, "Paris is the capital of France."), (query, "Python is a language.")]
    )
    assert len(scores) == 3
    assert scores[1] == max(scores)


@pytest.mark.asyncio
async def test_nvidia_reranker_failing_slice_raises_upstream_error():
    """A 4xx on any slice surfaces as UpstreamHTTPError (what the shared retry classifies on)."""
    from hindsight_api.engine.aiohttp_session import UpstreamHTTPError

    async def handler(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if body["passages"][0]["text"] == "bad":
            return web.json_response({"error": "too_long"}, status=422)
        return web.json_response({"rankings": [{"index": i, "logit": 0.0} for i in range(len(body["passages"]))]})

    async with stub_server(handler) as base_url:
        encoder = NvidiaCrossEncoder(api_key="k", model="nvidia/m", base_url=base_url, batch_size=1)
        await encoder.initialize()
        with pytest.raises(UpstreamHTTPError) as exc_info:
            await encoder.predict([("q", "ok"), ("q", "bad"), ("q", "ok2")])
    assert exc_info.value.status_code == 422
