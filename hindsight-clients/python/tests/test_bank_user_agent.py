"""Bank helpers preserve the SDK identity advertised by the rest of the client."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from hindsight_client import Hindsight
from hindsight_client.hindsight_client import DEFAULT_USER_AGENT
from hindsight_client_api.models.bank_profile_response import BankProfileResponse
from hindsight_client_api.models.disposition_traits import DispositionTraits


@pytest.mark.parametrize("method", ["acreate_bank", "aget_bank_config", "aupdate_bank_config", "areset_bank_config"])
@pytest.mark.parametrize("user_agent", [None, "hindsight-integration-test/1.2.3"])
async def test_bank_helper_uses_configured_user_agent(method: str, user_agent: str | None) -> None:
    observed: list[str] = []
    profile = BankProfileResponse(
        bank_id="test",
        name="test",
        mission="test",
        disposition=DispositionTraits(skepticism=3, literalism=3, empathy=3),
    )

    async def bank_route(request: web.Request) -> web.Response:
        observed.append(request.headers["User-Agent"])
        assert request.headers["Authorization"] == "Bearer test-key"
        return web.json_response(profile.to_dict())

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", bank_route)
    async with TestServer(app) as server:
        client = Hindsight(base_url=str(server.make_url("")), api_key="test-key", user_agent=user_agent)
        try:
            await getattr(client, method)("test")
        finally:
            await client.aclose()

    assert observed == [user_agent or DEFAULT_USER_AGENT]
