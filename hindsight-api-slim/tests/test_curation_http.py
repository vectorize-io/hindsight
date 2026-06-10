"""HTTP integration tests for memory curation endpoints.

Exercises the FastAPI PATCH /memories/{id} route end-to-end over an ASGI
transport, covering the happy path, validation, and not-found mapping. The
deeper cascade behaviour is covered at the engine level in
test_memory_curation.py.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.api import create_app
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import embedding_processing


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _insert_fact(memory: MemoryEngine, bank_id: str, text: str) -> str:
    """Insert one world fact with a real embedding; returns its id."""
    await memory.get_bank_profile(bank_id=bank_id, request_context=RequestContext())
    emb = await embedding_processing.generate_embeddings_batch(memory.embeddings, [text])
    mem_id = uuid.uuid4()
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, event_date, created_at, updated_at, consolidated_at)
            VALUES ($1, $2, $3, 'world', $4::vector, NOW(), NOW(), NOW(), NOW())
            """,
            mem_id,
            bank_id,
            text,
            str(emb[0]),
        )
    return str(mem_id)


@pytest.mark.asyncio
async def test_patch_invalidate_and_revert_over_http(api_client, memory):
    bank_id = f"curation-http-{uuid.uuid4().hex[:8]}"
    mem_id = await _insert_fact(memory, bank_id, "srv-04 runs PostgreSQL 14.")

    # Invalidate via PATCH
    resp = await api_client.patch(
        f"/v1/default/banks/{bank_id}/memories/{mem_id}",
        json={"state": "invalidated", "reason": "decommissioned"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["state"] == "invalidated"
    assert body["invalidation_reason"] == "decommissioned"

    # GET reflects the new state
    resp = await api_client.get(f"/v1/default/banks/{bank_id}/memories/{mem_id}")
    assert resp.status_code == 200
    assert resp.json()["state"] == "invalidated"

    # Revert via PATCH
    resp = await api_client.patch(
        f"/v1/default/banks/{bank_id}/memories/{mem_id}",
        json={"state": "valid"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "valid"
    assert resp.json()["invalidation_reason"] is None

    await memory.delete_bank(bank_id, request_context=RequestContext())


@pytest.mark.asyncio
async def test_patch_not_found_returns_404(api_client, memory):
    bank_id = f"curation-http-404-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=RequestContext())
    resp = await api_client.patch(
        f"/v1/default/banks/{bank_id}/memories/{uuid.uuid4()}",
        json={"state": "invalidated"},
    )
    assert resp.status_code == 404
    await memory.delete_bank(bank_id, request_context=RequestContext())


@pytest.mark.asyncio
async def test_patch_empty_body_is_rejected(api_client, memory):
    bank_id = f"curation-http-422-{uuid.uuid4().hex[:8]}"
    mem_id = await _insert_fact(memory, bank_id, "A fact.")
    # Neither text nor state → request model validation rejects it.
    resp = await api_client.patch(
        f"/v1/default/banks/{bank_id}/memories/{mem_id}",
        json={},
    )
    assert resp.status_code == 422
    await memory.delete_bank(bank_id, request_context=RequestContext())
