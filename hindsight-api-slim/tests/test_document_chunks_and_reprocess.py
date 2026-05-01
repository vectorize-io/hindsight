"""
Tests for document chunks API, reprocess, nodes_by_fact_type, and graph document/chunk filtering.
"""
from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app


# ── Fixtures ──


@pytest_asyncio.fixture
async def api_client(memory):
    """Create an async test client for the FastAPI app."""
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def bank_id():
    return f"test_doc_chunks_{datetime.now(timezone.utc).timestamp()}"


async def _retain(api_client, bank_id, document_id, content, tags=None):
    """Helper to retain a document via the HTTP API."""
    item = {"content": content, "document_id": document_id}
    if tags:
        item["tags"] = tags
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [item]},
    )
    assert response.status_code == 200
    return response.json()


# ── list_document_chunks ──


@pytest.mark.asyncio
async def test_list_document_chunks(memory, request_context):
    """list_document_chunks returns chunks ordered by chunk_index."""
    bank_id = f"test_chunks_{datetime.now(timezone.utc).timestamp()}"

    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Alice works at Google. Bob works at Meta. " * 20, "document_id": "doc1"}],
            request_context=request_context,
        )

        result = await memory.list_document_chunks(
            bank_id=bank_id,
            document_id="doc1",
            request_context=request_context,
        )

        assert result is not None
        assert result["total"] >= 1
        assert len(result["items"]) == result["total"]

        # Chunks should be ordered by chunk_index
        indices = [c["chunk_index"] for c in result["items"]]
        assert indices == sorted(indices)

        # Each chunk should have the expected fields
        for chunk in result["items"]:
            assert "chunk_id" in chunk
            assert "chunk_text" in chunk
            assert chunk["document_id"] == "doc1"
            assert chunk["bank_id"] == bank_id
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_list_document_chunks_pagination(memory, request_context):
    """list_document_chunks respects limit and offset."""
    bank_id = f"test_chunks_page_{datetime.now(timezone.utc).timestamp()}"

    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Content. " * 200, "document_id": "doc-pag"}],
            request_context=request_context,
        )

        all_chunks = await memory.list_document_chunks(
            bank_id=bank_id, document_id="doc-pag", request_context=request_context
        )
        total = all_chunks["total"]
        if total < 2:
            pytest.skip("Document produced fewer than 2 chunks, can't test pagination")

        page1 = await memory.list_document_chunks(
            bank_id=bank_id, document_id="doc-pag", limit=1, offset=0, request_context=request_context
        )
        assert len(page1["items"]) == 1
        assert page1["total"] == total

        page2 = await memory.list_document_chunks(
            bank_id=bank_id, document_id="doc-pag", limit=1, offset=1, request_context=request_context
        )
        assert len(page2["items"]) == 1
        assert page2["items"][0]["chunk_id"] != page1["items"][0]["chunk_id"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_list_document_chunks_not_found(memory, request_context):
    """list_document_chunks returns None for non-existent document."""
    bank_id = f"test_chunks_404_{datetime.now(timezone.utc).timestamp()}"

    result = await memory.list_document_chunks(
        bank_id=bank_id, document_id="nonexistent", request_context=request_context
    )
    assert result is None


# ── get_document nodes_by_fact_type ──


@pytest.mark.asyncio
async def test_get_document_nodes_by_fact_type(memory, request_context):
    """get_document returns nodes_by_fact_type with per-type counts."""
    bank_id = f"test_doc_composition_{datetime.now(timezone.utc).timestamp()}"

    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Alice works at Google on AI research.", "document_id": "doc-comp"}],
            request_context=request_context,
        )

        doc = await memory.get_document("doc-comp", bank_id, request_context=request_context)
        assert doc is not None
        assert "nodes_by_fact_type" in doc

        nbt = doc["nodes_by_fact_type"]
        assert "world" in nbt
        assert "experience" in nbt
        assert "observation" in nbt
        assert isinstance(nbt["world"], int)
        assert isinstance(nbt["experience"], int)
        assert isinstance(nbt["observation"], int)

        # memory_unit_count counts facts with document_id set (world + experience).
        # Observations have no document_id and are counted via source links, so
        # they are reported separately.
        assert nbt["world"] + nbt["experience"] == doc["memory_unit_count"]
        assert nbt["observation"] >= 0
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ── reprocess_document ──


@pytest.mark.asyncio
async def test_reprocess_document(memory, request_context):
    """reprocess_document submits an async retain operation for an existing document."""
    bank_id = f"test_reprocess_{datetime.now(timezone.utc).timestamp()}"

    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Alice works at Google.", "document_id": "doc-reprocess"}],
            request_context=request_context,
        )

        result = await memory.reprocess_document(
            bank_id=bank_id, document_id="doc-reprocess", request_context=request_context
        )

        assert result is not None
        assert "operation_id" in result
        assert "items_count" in result
        assert result["items_count"] == 1
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_reprocess_document_not_found(memory, request_context):
    """reprocess_document returns None for non-existent document."""
    result = await memory.reprocess_document(
        bank_id="nonexistent-bank", document_id="nonexistent", request_context=request_context
    )
    assert result is None


# ── Graph document_id / chunk_id filters (HTTP level) ──


@pytest.mark.asyncio
async def test_graph_document_id_filter(api_client, bank_id):
    """Graph endpoint filters by document_id."""
    await _retain(api_client, bank_id, "doc-a", "Alice works at Google on AI.")
    await _retain(api_client, bank_id, "doc-b", "Bob works at Meta on VR.")

    # Filter by doc-a
    response = await api_client.get(
        f"/v1/default/banks/{bank_id}/graph",
        params={"document_id": "doc-a"},
    )
    assert response.status_code == 200
    data = response.json()
    doc_ids = {row.get("document_id") for row in data["table_rows"]}
    assert "doc-a" in doc_ids
    assert "doc-b" not in doc_ids


@pytest.mark.asyncio
async def test_graph_chunk_id_filter(api_client, bank_id):
    """Graph endpoint filters by chunk_id."""
    await _retain(api_client, bank_id, "doc-chunk-test", "Alice works at Google. " * 20)

    # First get chunks to find a valid chunk_id
    response = await api_client.get(
        f"/v1/default/banks/{bank_id}/documents/doc-chunk-test/chunks"
    )
    assert response.status_code == 200
    chunks_data = response.json()
    if chunks_data["total"] == 0:
        pytest.skip("No chunks created")

    chunk_id = chunks_data["items"][0]["chunk_id"]

    # Filter graph by that chunk_id
    response = await api_client.get(
        f"/v1/default/banks/{bank_id}/graph",
        params={"chunk_id": chunk_id},
    )
    assert response.status_code == 200
    data = response.json()
    chunk_ids = {row.get("chunk_id") for row in data["table_rows"]}
    # All returned memories should belong to the requested chunk
    assert all(cid == chunk_id for cid in chunk_ids if cid is not None)


# ── HTTP endpoints for chunks and reprocess ──


@pytest.mark.asyncio
async def test_http_list_document_chunks(api_client, bank_id):
    """HTTP GET .../documents/{id}/chunks returns chunks."""
    await _retain(api_client, bank_id, "doc-http-chunks", "Alice works at Google on AI research. Bob works at Meta on VR systems. " * 20)

    response = await api_client.get(
        f"/v1/default/banks/{bank_id}/documents/doc-http-chunks/chunks"
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] >= 1


@pytest.mark.asyncio
async def test_http_list_document_chunks_not_found(api_client, bank_id):
    """HTTP GET .../documents/{id}/chunks returns 404 for non-existent document."""
    response = await api_client.get(
        f"/v1/default/banks/{bank_id}/documents/nonexistent/chunks"
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_http_reprocess_document(api_client, bank_id):
    """HTTP POST .../documents/{id}/reprocess returns success with operation_id."""
    await _retain(api_client, bank_id, "doc-http-reprocess", "Alice works at Google.")

    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/documents/doc-http-reprocess/reprocess"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "operation_id" in data


@pytest.mark.asyncio
async def test_http_reprocess_document_not_found(api_client, bank_id):
    """HTTP POST .../documents/{id}/reprocess returns 404 for non-existent document."""
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/documents/nonexistent/reprocess"
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_http_get_document_includes_nodes_by_fact_type(api_client, bank_id):
    """HTTP GET .../documents/{id} includes nodes_by_fact_type."""
    await _retain(api_client, bank_id, "doc-http-comp", "Alice works at Google on AI research.")

    response = await api_client.get(
        f"/v1/default/banks/{bank_id}/documents/doc-http-comp"
    )
    assert response.status_code == 200
    data = response.json()
    assert "nodes_by_fact_type" in data
    nbt = data["nodes_by_fact_type"]
    assert "world" in nbt
    assert "experience" in nbt
    assert "observation" in nbt
