"""Handwritten bank requests should use the same environment proxy policy as the SDK."""

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
async def test_bank_routes_use_http_proxy(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def proxy(request: web.Request) -> web.Response:
        seen.append(request.raw_path)
        assert request.headers["Authorization"] == "Bearer test-key"
        return web.json_response(profile("test"))

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", proxy)
    async with TestServer(app) as server:
        proxy_url = str(server.make_url(""))
        for name in ("HTTP_PROXY", "http_proxy"):
            monkeypatch.setenv(name, proxy_url)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.setenv(name, "")
        client = Hindsight(base_url="http://hindsight.invalid", api_key="test-key", timeout=2)
        try:
            await getattr(client, method)("test")
        finally:
            await client.aclose()
    suffix = "" if method == "acreate_bank" else "/config"
    assert seen == [f"http://hindsight.invalid/v1/default/banks/test{suffix}"]
