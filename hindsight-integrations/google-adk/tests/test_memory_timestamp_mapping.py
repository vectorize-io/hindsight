"""Explicit ADK memory timestamps survive the adapter and SDK boundary."""

from datetime import datetime

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types
from hindsight_client import Hindsight
from hindsight_client_api.models.memory_item import MemoryItem
from hindsight_client_api.models.retain_request import RetainRequest
from hindsight_client_api.models.retain_response import RetainResponse
from hindsight_google_adk import HindsightMemoryService


@pytest.mark.parametrize("timestamp", ["2023-03-10T12:45:00Z", "2023-03-10T12:45:00+05:30", "unset", None])
async def test_explicit_memory_preserves_timestamp(timestamp: str | None) -> None:
    retained: list[MemoryItem] = []

    async def retain(request: web.Request) -> web.Response:
        payload = RetainRequest.from_dict(await request.json())
        assert payload is not None
        retained.extend(payload.items)
        response = RetainResponse(success=True, bank_id="app::user", items_count=1, var_async=False)
        return web.json_response(response.to_dict())

    app = web.Application()
    app.router.add_post("/v1/default/banks/app::user/memories", retain)
    async with TestServer(app) as server:
        client = Hindsight(base_url=str(server.make_url("")))
        service = HindsightMemoryService(client=client)
        try:
            entry = MemoryEntry(
                id="historical-memory",
                author="user",
                content=types.Content(parts=[types.Part(text="Moved to Paris")]),
                timestamp=timestamp,
            )
            await service.add_memory(app_name="app", user_id="user", memories=[entry])
        finally:
            await client.aclose()

    assert len(retained) == 1
    item = retained[0]
    assert item.document_id == "historical-memory"
    assert item.metadata["author"] == "user"
    expected = datetime.fromisoformat(timestamp) if timestamp is not None and timestamp != "unset" else timestamp
    assert (item.timestamp.actual_instance if item.timestamp is not None else None) == expected
