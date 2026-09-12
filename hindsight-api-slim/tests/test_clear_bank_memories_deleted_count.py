"""DELETE /v1/{tenant}/banks/{bank}/memories must report how many memory units it removed.

The engine already counts the units inside the delete transaction; the endpoint used to
drop that number and answer with a bare ``{"success": true}``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from hindsight_api.api import create_app


def _app(delete_bank: AsyncMock):
    return create_app(SimpleNamespace(audit_logger=None, delete_bank=delete_bank), initialize_memory=False)


@pytest.mark.asyncio
async def test_clear_bank_memories_reports_deleted_count():
    delete_bank = AsyncMock(return_value={"memory_units_deleted": 6, "entities_deleted": 0, "documents_deleted": 0})
    transport = httpx.ASGITransport(app=_app(delete_bank))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/v1/default/banks/repro-bank/memories")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["deleted_count"] == 6
    assert "6" in body["message"]
    delete_bank.assert_awaited_once()
    kwargs = delete_bank.await_args.kwargs
    assert kwargs["fact_type"] is None
    assert kwargs["delete_bank_profile"] is False


@pytest.mark.asyncio
async def test_clear_bank_memories_type_filter_reports_deleted_count():
    delete_bank = AsyncMock(return_value={"memory_units_deleted": 2, "entities_deleted": 0})
    transport = httpx.ASGITransport(app=_app(delete_bank))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/v1/default/banks/repro-bank/memories", params={"type": "world"})

    assert resp.status_code == 200
    assert resp.json()["deleted_count"] == 2
    assert delete_bank.await_args.kwargs["fact_type"] == "world"


@pytest.mark.asyncio
async def test_clear_bank_memories_on_empty_bank_reports_zero_not_null():
    delete_bank = AsyncMock(return_value={"memory_units_deleted": 0, "entities_deleted": 0})
    transport = httpx.ASGITransport(app=_app(delete_bank))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/v1/default/banks/empty-bank/memories")

    assert resp.status_code == 200
    assert resp.json()["deleted_count"] == 0
