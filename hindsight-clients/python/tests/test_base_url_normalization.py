"""A base URL's trailing slash must not become part of generated route paths."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from hindsight_client import Hindsight
from hindsight_client_api.models.features_info import FeaturesInfo
from hindsight_client_api.models.version_response import VersionResponse


@pytest.mark.parametrize("suffix", ["", "/", "///"])
@pytest.mark.parametrize("prefix", ["", "/service"])
async def test_generated_routes_preserve_base_prefix_without_double_slashes(prefix: str, suffix: str) -> None:
    seen: list[str] = []
    expected = VersionResponse(
        api_version="test-version",
        features=FeaturesInfo(**dict.fromkeys(FeaturesInfo.model_fields, False)),
    )

    async def version(request: web.Request) -> web.Response:
        seen.append(request.path)
        return web.json_response(expected.to_dict())

    app = web.Application()
    app.router.add_get(f"{prefix}/version", version)
    async with TestServer(app) as server:
        client = Hindsight(base_url=f"{str(server.make_url(prefix)).rstrip('/')}{suffix}")
        try:
            response = await client.aget_version()
        finally:
            await client.aclose()

    assert response.api_version == "test-version"
    assert seen == [f"{prefix}/version"]
