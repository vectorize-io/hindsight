"""A transient mission setup failure must not become permanent for the ADK service."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from google.adk.events import Event
from google.adk.sessions import Session
from google.genai import types
from hindsight_client import Hindsight
from hindsight_client_api.models.bank_profile_response import BankProfileResponse
from hindsight_client_api.models.disposition_traits import DispositionTraits
from hindsight_client_api.models.retain_response import RetainResponse
from hindsight_google_adk import HindsightMemoryService


async def test_later_session_retries_failed_mission_setup_then_caches_success() -> None:
    bank_setups: list[str] = []
    retained_documents: list[str] = []

    async def setup(request: web.Request) -> web.Response:
        body = await request.json()
        bank_setups.append(body["mission"])
        if len(bank_setups) == 1:
            return web.Response(status=503, text="Temporary startup failure")
        profile = BankProfileResponse(
            bank_id="app::user",
            name="app::user",
            mission=body["mission"],
            disposition=DispositionTraits(skepticism=3, literalism=3, empathy=3),
        )
        return web.json_response(profile.to_dict())

    async def retain(request: web.Request) -> web.Response:
        body = await request.json()
        retained_documents.append(body["items"][0]["document_id"])
        result = RetainResponse(success=True, bank_id="app::user", items_count=1, var_async=False)
        return web.json_response(result.to_dict())

    app = web.Application()
    app.router.add_put("/v1/default/banks/app::user", setup)
    app.router.add_post("/v1/default/banks/app::user/memories", retain)
    async with TestServer(app) as server:
        client = Hindsight(base_url=str(server.make_url("")))
        service = HindsightMemoryService(client=client, mission="Track user preferences")
        try:
            for session_id in ["first", "second", "third"]:
                session = Session(
                    id=session_id,
                    app_name="app",
                    user_id="user",
                    events=[Event(author="user", content=types.Content(parts=[types.Part(text="Likes tea")]))],
                )
                await service.add_session_to_memory(session)
        finally:
            await client.aclose()

    assert retained_documents == ["first", "second", "third"]
    assert bank_setups == ["Track user preferences", "Track user preferences"]
