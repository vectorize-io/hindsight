"""Tests for document reprocess: forces LLM re-extraction and replaces memory units."""

from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app


@pytest_asyncio.fixture
async def api_client(memory):
    """Create an async test client for the FastAPI app."""
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def bank_id():
    return f"test_reprocess_reextract_{datetime.now(timezone.utc).timestamp()}"


@pytest.mark.asyncio
async def test_reprocess_document_forces_reextraction_and_replaces_units(memory, request_context):
    """Calling reprocess_document on an existing document with unchanged content must
    force re-extraction, deleting old units and creating new units rather than
    silently classifying as crash recovery."""
    bank_id = f"test_reprocess_force_{datetime.now(timezone.utc).timestamp()}"
    document_id = "doc-force-reextract"
    content = "Alice works at Google on Android. Bob works at Apple on iOS."

    try:
        # Initial retain
        v1_units = await memory.retain_async(
            bank_id=bank_id,
            content=content,
            document_id=document_id,
            request_context=request_context,
        )
        assert len(v1_units) > 0, "Initial retain should extract facts"

        # List units before reprocess using engine API
        v1_listing = await memory.list_memory_units(
            bank_id=bank_id,
            document_id=document_id,
            request_context=request_context,
        )
        v1_unit_ids = {item["id"] for item in v1_listing["items"]}
        assert len(v1_unit_ids) > 0

        # Reprocess the document with identical content
        reprocess_result = await memory.reprocess_document(
            bank_id=bank_id,
            document_id=document_id,
            request_context=request_context,
        )
        assert reprocess_result is not None
        assert "operation_id" in reprocess_result

        # Check document state after reprocess
        doc_v2 = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc_v2 is not None
        assert doc_v2["memory_unit_count"] > 0, "Reprocessed document must have memory units"

        # List units after reprocess using engine API
        v2_listing = await memory.list_memory_units(
            bank_id=bank_id,
            document_id=document_id,
            request_context=request_context,
        )
        v2_unit_ids = {item["id"] for item in v2_listing["items"]}

        assert len(v2_unit_ids) > 0, "Should have memory units after reprocess"
        assert v2_unit_ids != v1_unit_ids, "Reprocess must replace old unit IDs with newly extracted unit IDs"
        assert not (v1_unit_ids & v2_unit_ids), "Old unit IDs should have been cascade-deleted on replace"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_http_reprocess_document_endpoint_replaces_facts(api_client, bank_id):
    """HTTP POST /v1/default/banks/{bank_id}/documents/{document_id}/reprocess replaces existing facts."""
    item = {"content": "Dave is a lead developer working on database engines.", "document_id": "doc-http-reprocess"}
    response = await api_client.post(f"/v1/default/banks/{bank_id}/memories", json={"items": [item]})
    assert response.status_code == 200

    # Get initial graph memories
    graph_res1 = await api_client.get(
        f"/v1/default/banks/{bank_id}/graph", params={"document_id": "doc-http-reprocess"}
    )
    assert graph_res1.status_code == 200
    rows_v1 = graph_res1.json()["table_rows"]
    assert len(rows_v1) > 0
    v1_ids = {r["id"] for r in rows_v1}

    # Call reprocess HTTP endpoint
    reprocess_res = await api_client.post(f"/v1/default/banks/{bank_id}/documents/doc-http-reprocess/reprocess")
    assert reprocess_res.status_code == 200
    assert reprocess_res.json()["success"] is True

    # Get new graph memories
    graph_res2 = await api_client.get(
        f"/v1/default/banks/{bank_id}/graph", params={"document_id": "doc-http-reprocess"}
    )
    assert graph_res2.status_code == 200
    rows_v2 = graph_res2.json()["table_rows"]
    assert len(rows_v2) > 0
    v2_ids = {r["id"] for r in rows_v2}

    assert v1_ids != v2_ids, "HTTP reprocess must replace old facts with newly extracted facts"
    assert not (v1_ids & v2_ids), "Old unit IDs must no longer exist"
