"""Cohere's native retrieval task belongs to each request, including v2 embeds."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.config import DEFAULT_EMBEDDINGS_MAX_CONCURRENT_REQUESTS
from hindsight_api.engine.aiohttp_session import LoopLocal
from hindsight_api.engine.embeddings import CohereEmbeddings


def _backend(output_dimensions: int | None, input_type: str = "search_document") -> CohereEmbeddings:
    backend = CohereEmbeddings(api_key="test", output_dimensions=output_dimensions, input_type=input_type)
    # What `_with_request_concurrency` gives every remote provider. Constructing the backend
    # directly skips the factory and leaves the base class's sequential default of 1, which is a
    # semaphore of one in-flight request — the concurrency test below then deadlocks waiting for a
    # second request that can never start, and it would be testing a bound it set itself rather
    # than how a deployed Cohere backend behaves.
    backend.max_concurrent_requests = DEFAULT_EMBEDDINGS_MAX_CONCURRENT_REQUESTS
    response = SimpleNamespace(embeddings=[[0.1, 0.2]])
    if output_dimensions is not None:
        response.embeddings = SimpleNamespace(float_=[[0.1, 0.2]])
    client = MagicMock()
    client.embed = AsyncMock(return_value=response)
    client.v2.embed = AsyncMock(return_value=response)
    backend._clients = LoopLocal(lambda: client)
    return backend


def _embed_call(backend: CohereEmbeddings) -> AsyncMock:
    assert backend._clients is not None
    client = backend._clients.get()
    return client.embed if backend.output_dimensions is None else client.v2.embed


@pytest.mark.asyncio
@pytest.mark.parametrize("output_dimensions", [None, 256])
async def test_recall_queries_use_cohere_search_query(output_dimensions: int | None) -> None:
    backend = _backend(output_dimensions)

    assert await backend.encode_query(["where are my keys?"]) == [[0.1, 0.2]]

    request = _embed_call(backend).await_args
    assert request is not None
    assert request.kwargs["input_type"] == "search_query"
    assert request.kwargs["texts"] == ["where are my keys?"]
    assert backend.input_type == "search_document"


@pytest.mark.asyncio
@pytest.mark.parametrize("output_dimensions", [None, 256])
async def test_concurrent_query_and_document_tasks_stay_request_local(output_dimensions: int | None) -> None:
    backend = _backend(output_dimensions)
    embed = _embed_call(backend)
    response = embed.return_value
    query_started = asyncio.Event()
    document_started = asyncio.Event()

    async def respond(**kwargs) -> SimpleNamespace:
        if kwargs["texts"] == ["query"]:
            query_started.set()
            await asyncio.wait_for(document_started.wait(), timeout=1.0)
        else:
            document_started.set()
        return response

    embed.side_effect = respond
    query = asyncio.create_task(backend.encode_query(["query"]))
    await asyncio.wait_for(query_started.wait(), timeout=1.0)
    document = asyncio.create_task(backend.encode_documents(["document"]))
    await asyncio.gather(query, document)

    assert [call.kwargs["input_type"] for call in embed.await_args_list] == ["search_query", "search_document"]
    assert backend.input_type == "search_document"


@pytest.mark.asyncio
@pytest.mark.parametrize("output_dimensions", [None, 256])
async def test_direct_encode_keeps_explicit_input_type(output_dimensions: int | None) -> None:
    backend = _backend(output_dimensions, input_type="classification")

    assert await backend.encode(["classify me"]) == [[0.1, 0.2]]

    request = _embed_call(backend).await_args
    assert request is not None
    assert request.kwargs["input_type"] == "classification"
