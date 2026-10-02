"""Single-item retain must preserve the strategy/scope options already supported by batches."""

import asyncio
from typing import Literal

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from hindsight_client import Hindsight
from hindsight_client_api.models.memory_item import MemoryItem
from hindsight_client_api.models.retain_request import RetainRequest
from hindsight_client_api.models.retain_response import RetainResponse


@pytest.mark.parametrize("method", ["retain", "aretain"])
@pytest.mark.parametrize(
    ("scopes", "strategy"),
    [(value, "meeting") for value in ["per_tag", "combined", "all_combinations", "shared", [["project:p1"], []], None]]
    + [(None, None)],
)
async def test_single_retain_forwards_existing_options(
    method: str,
    scopes: Literal["per_tag", "combined", "all_combinations", "shared"] | list[list[str]] | None,
    strategy: str | None,
) -> None:
    captured: list[MemoryItem] = []

    async def retain(request: web.Request) -> web.Response:
        payload = RetainRequest.from_dict(await request.json())
        assert payload is not None
        captured.extend(payload.items)
        result = RetainResponse(success=True, bank_id="test", items_count=1, var_async=False)
        return web.json_response(result.to_dict())

    app = web.Application()
    app.router.add_post("/v1/default/banks/test/memories", retain)
    async with TestServer(app) as server:
        url = str(server.make_url(""))
        if method == "retain":

            def sync_retain() -> RetainResponse:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                client = Hindsight(base_url=url)
                try:
                    if strategy is None:
                        return client.retain("test", "Meeting notes")
                    return client.retain("test", "Meeting notes", observation_scopes=scopes, strategy=strategy)
                finally:
                    client.close()
                    loop.close()
                    asyncio.set_event_loop(None)

            result = await asyncio.to_thread(sync_retain)
        else:
            client = Hindsight(base_url=url)
            try:
                if strategy is None:
                    result = await client.aretain("test", "Meeting notes")
                else:
                    result = await client.aretain("test", "Meeting notes", observation_scopes=scopes, strategy=strategy)
            finally:
                await client.aclose()

    assert result.items_count == 1
    assert len(captured) == 1
    item = captured[0]
    assert item.strategy == strategy
    assert (item.observation_scopes.actual_instance if item.observation_scopes is not None else None) == scopes
