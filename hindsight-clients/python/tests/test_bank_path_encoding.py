"""Handwritten bank routes must preserve bank IDs as a single URL component."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from hindsight_client import Hindsight

METHODS = ["acreate_bank", "aget_bank_config", "aupdate_bank_config", "areset_bank_config"]


def profile(bank_id: str) -> dict:
    return {
        "bank_id": bank_id,
        "name": bank_id,
        "mission": "test",
        "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
    }


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bank_id", ["team?draft#notes", "team%2Farchive", "team + café"])
async def test_bank_routes_encode_reserved_characters(method: str, bank_id: str) -> None:
    seen_paths: list[str] = []
    seen_queries: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        seen_paths.append(request.path)
        seen_queries.append(request.query_string)
        return web.json_response(profile(bank_id))

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    async with TestServer(app) as server:
        client = Hindsight(base_url=str(server.make_url("")).rstrip("/"))
        try:
            await getattr(client, method)(bank_id)
        finally:
            await client.aclose()
    suffix = "" if method == "acreate_bank" else "/config"
    assert seen_paths == [f"/v1/default/banks/{bank_id}{suffix}"]
    assert seen_queries == [""]
