"""Convenience memory listings retain the filters available in the API and TS SDK."""

import inspect

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from hindsight_client import Hindsight
from hindsight_client_api.models.list_memory_units_response import ListMemoryUnitsResponse


@pytest.mark.parametrize("method", ["list_memories", "alist_memories"])
def test_memory_listing_family_exposes_existing_filters(method: str) -> None:
    parameters = inspect.signature(getattr(Hindsight, method)).parameters
    assert {"document_id", "state", "consolidation_state"} <= parameters.keys()


@pytest.mark.parametrize("state", ["valid", "invalidated"])
@pytest.mark.parametrize("consolidation_state", ["failed", "pending", "done"])
async def test_memory_filters_reach_the_http_query(state: str, consolidation_state: str) -> None:
    requests: list[web.Request] = []
    response = ListMemoryUnitsResponse(items=[], total=0, limit=100, offset=0)

    async def memories(request: web.Request) -> web.Response:
        requests.append(request)
        return web.json_response(response.to_dict())

    app = web.Application()
    app.router.add_get("/v1/default/banks/test/memories/list", memories)
    async with TestServer(app) as server:
        client = Hindsight(base_url=str(server.make_url("")))
        try:
            result = await client.alist_memories(
                "test", document_id="notes + revision&2", state=state, consolidation_state=consolidation_state
            )
        finally:
            await client.aclose()

    assert result.total == 0
    assert len(requests) == 1
    assert requests[0].query["document_id"] == "notes + revision&2"
    assert requests[0].query["state"] == state
    assert requests[0].query["consolidation_state"] == consolidation_state


async def test_omitted_memory_filters_are_not_sent() -> None:
    requests: list[web.Request] = []
    response = ListMemoryUnitsResponse(items=[], total=0, limit=100, offset=0)

    async def memories(request: web.Request) -> web.Response:
        requests.append(request)
        return web.json_response(response.to_dict())

    app = web.Application()
    app.router.add_get("/v1/default/banks/test/memories/list", memories)
    async with TestServer(app) as server:
        client = Hindsight(base_url=str(server.make_url("")))
        try:
            await client.alist_memories("test")
        finally:
            await client.aclose()

    assert len(requests) == 1
    assert {"document_id", "state", "consolidation_state"}.isdisjoint(requests[0].query)
