"""Tests for the server-side clamp on /banks/{bank_id}/graph?limit=...

Edge count scales quadratically with node count for dense banks; the Control
Plane (Next.js) deserializes the response as a single JS string, capped at
~512 MiB by V8. The endpoint silently clamps ``limit`` at 200 to keep
responses parseable while preserving backwards-compat for callers passing
larger values.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api.http import create_app
from hindsight_api.engine.memory_engine import MemoryEngine

GRAPH_LIMIT_CAP = 200


@pytest_asyncio.fixture
async def graph_clamp_api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_graph_limit_above_cap_is_silently_clamped(
    memory: MemoryEngine, request_context, graph_clamp_api_client: httpx.AsyncClient
):
    """Requests above 200 succeed (200 OK) and the response's ``limit`` field
    reflects the clamped value, not the requested value."""
    bank_id = f"graph-clamp-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    # A single retain is enough to make the endpoint return a non-empty graph;
    # the clamp behavior is independent of bank density.
    await memory.retain_async(
        bank_id=bank_id,
        content="Alice met Bob at the conference.",
        request_context=request_context,
    )

    response = await graph_clamp_api_client.get(
        f"/v1/default/banks/{bank_id}/graph",
        params={"limit": 1000},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["limit"] == GRAPH_LIMIT_CAP, (
        f"expected clamped limit={GRAPH_LIMIT_CAP}, got {data['limit']}"
    )
    assert len(data["nodes"]) <= GRAPH_LIMIT_CAP


@pytest.mark.asyncio
async def test_graph_limit_below_cap_passes_through(
    memory: MemoryEngine, request_context, graph_clamp_api_client: httpx.AsyncClient
):
    """Requests at or below 200 are not modified."""
    bank_id = f"graph-clamp-passthrough-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    await memory.retain_async(
        bank_id=bank_id,
        content="Carol travels to Boston monthly.",
        request_context=request_context,
    )

    response = await graph_clamp_api_client.get(
        f"/v1/default/banks/{bank_id}/graph",
        params={"limit": 50},
    )
    assert response.status_code == 200, response.text
    assert response.json()["limit"] == 50
