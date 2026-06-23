"""Regression: user-facing GET list endpoints must reject negative limit/offset
with a clean 422 at the FastAPI boundary instead of letting the value reach
Postgres (``LIMIT/OFFSET must not be negative``) and surfacing as an opaque 500
that also leaks the raw Postgres error string.

This makes pagination validation consistent with the sibling list endpoints in
the same router (document-chunks / directives / async-ops / audit) that already
declare ``Query(..., ge=...)``. The engine emits ``LIMIT $n OFFSET $n`` with no
``max(0, ...)`` clamp, so the guard has to live at the request boundary.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.api import create_app


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _url(bank_id: str, suffix: str) -> str:
    return f"/v1/default/banks/{bank_id}/{suffix}"


# Endpoints that accept a ``limit`` query param.
LIMIT_ENDPOINTS = [
    "graph",
    "memories/list",
    "entities",
    "entities/graph",
    "documents",
    "tags",
]

# Subset that also accept an ``offset`` query param.
OFFSET_ENDPOINTS = [
    "memories/list",
    "entities",
    "documents",
    "tags",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", LIMIT_ENDPOINTS)
async def test_negative_limit_returns_422_not_500(api_client, suffix):
    bank_id = f"pag-{uuid.uuid4().hex[:8]}"
    resp = await api_client.get(_url(bank_id, suffix), params={"limit": -1})
    # FastAPI validation runs before the handler / DB, so a bad pagination input
    # is a clean 422 — never a 500 leaking the raw Postgres error.
    assert resp.status_code == 422, resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", OFFSET_ENDPOINTS)
async def test_negative_offset_returns_422_not_500(api_client, suffix):
    bank_id = f"pag-{uuid.uuid4().hex[:8]}"
    resp = await api_client.get(_url(bank_id, suffix), params={"offset": -1})
    assert resp.status_code == 422, resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 1, 100])
async def test_valid_limit_is_accepted_including_zero(api_client, memory, limit):
    # Positive control: ge=0 rejects only NEGATIVE limits. A non-negative limit
    # — including limit=0 (a valid empty page, LIMIT 0) — must still be accepted,
    # so this fix does not change behavior for any previously-valid input.
    bank_id = f"pag-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=RequestContext())
    resp = await api_client.get(_url(bank_id, "memories/list"), params={"limit": limit, "offset": 0})
    assert resp.status_code == 200, resp.text
