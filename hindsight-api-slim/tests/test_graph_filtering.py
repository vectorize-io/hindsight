"""
Tests for server-side filtering in the graph API endpoint.

Verifies that q (text search) and tags filters work correctly
when passed as query parameters to GET /v1/default/banks/{bank_id}/graph.
"""

import uuid
from datetime import datetime

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
def test_bank_id():
    """Provide a unique bank ID for this test run."""
    return f"graph_filter_test_{datetime.now().timestamp()}"


@pytest.mark.asyncio
async def test_graph_no_filter_returns_all(api_client, test_bank_id):
    """Without filters the graph endpoint returns all memories."""
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {"content": "Alice loves hiking in the mountains.", "tags": ["user_alice"]},
                {"content": "Bob enjoys swimming at the beach.", "tags": ["user_bob"]},
            ]
        },
    )
    assert response.status_code == 200

    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/graph")
    assert response.status_code == 200
    data = response.json()
    assert "table_rows" in data
    texts = [row["text"] for row in data["table_rows"]]
    assert any("Alice" in t for t in texts)
    assert any("Bob" in t for t in texts)


@pytest.mark.asyncio
async def test_graph_q_filter_returns_matching(api_client, test_bank_id):
    """The q parameter filters memories by text content."""
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {"content": "Alice loves hiking in the mountains."},
                {"content": "Bob enjoys swimming at the beach."},
            ]
        },
    )
    assert response.status_code == 200

    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/graph", params={"q": "Alice"})
    assert response.status_code == 200
    data = response.json()
    texts = [row["text"] for row in data["table_rows"]]
    assert all("Alice" in t or "alice" in t.lower() for t in texts), f"Expected only Alice memories, got: {texts}"
    assert not any("Bob" in t for t in texts)


@pytest.mark.asyncio
async def test_graph_q_filter_case_insensitive(api_client, test_bank_id):
    """The q filter is case-insensitive."""
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {"content": "Alice loves hiking in the mountains."},
                {"content": "Bob enjoys swimming at the beach."},
            ]
        },
    )
    assert response.status_code == 200

    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/graph", params={"q": "alice"})
    assert response.status_code == 200
    data = response.json()
    texts = [row["text"] for row in data["table_rows"]]
    assert any("Alice" in t for t in texts)
    assert not any("Bob" in t for t in texts)


@pytest.mark.asyncio
async def test_graph_tags_filter_returns_matching(api_client, test_bank_id):
    """The tags parameter filters memories to only those with matching tags."""
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {"content": "Alice loves hiking.", "tags": ["user_alice"]},
                {"content": "Bob enjoys swimming.", "tags": ["user_bob"]},
            ]
        },
    )
    assert response.status_code == 200

    response = await api_client.get(
        f"/v1/default/banks/{test_bank_id}/graph",
        params={"tags": "user_alice", "tags_match": "all_strict"},
    )
    assert response.status_code == 200
    data = response.json()
    texts = [row["text"] for row in data["table_rows"]]
    assert any("Alice" in t for t in texts)
    assert not any("Bob" in t for t in texts)


@pytest.mark.asyncio
async def test_graph_q_and_tags_filter_combined(api_client, test_bank_id):
    """Combining q and tags filters applies both server-side."""
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {"content": "Alice loves hiking.", "tags": ["user_alice"]},
                {"content": "Alice also loves coding.", "tags": ["user_alice"]},
                {"content": "Bob enjoys swimming.", "tags": ["user_bob"]},
            ]
        },
    )
    assert response.status_code == 200

    response = await api_client.get(
        f"/v1/default/banks/{test_bank_id}/graph",
        params={"q": "hiking", "tags": "user_alice", "tags_match": "all_strict"},
    )
    assert response.status_code == 200
    data = response.json()
    texts = [row["text"] for row in data["table_rows"]]
    assert any("hiking" in t.lower() for t in texts)
    assert not any("coding" in t.lower() for t in texts)
    assert not any("Bob" in t for t in texts)


@pytest.mark.asyncio
async def test_graph_document_filter_includes_observations_via_source_memories(
    memory, api_client, test_bank_id, request_context
):
    """Filtering the graph by document_id surfaces observations whose source
    memories belong to that document.

    Observations are consolidated rows that have no document_id of their own,
    so a naive equality filter on memory_units.document_id excludes them and
    leaves the document detail's Observations tab perpetually empty even when
    the bank has observations consolidated from facts in that document.
    """
    bank_id = test_bank_id
    document_id = f"doc-{uuid.uuid4().hex[:8]}"

    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    fact_id = uuid.uuid4()
    observation_id = uuid.uuid4()
    other_fact_id = uuid.uuid4()
    unrelated_observation_id = uuid.uuid4()

    async with memory._pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO documents (id, bank_id, original_text, content_hash)
            VALUES ($1, $2, $3, $4)
            """,
            document_id,
            bank_id,
            "doc body",
            "hash-test",
        )

        # World fact tied to the document.
        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, document_id, history)
            VALUES ($1, $2, $3, 'world', $4, '[]'::jsonb)
            """,
            fact_id,
            bank_id,
            "fact in document",
            document_id,
        )

        # Unrelated fact NOT tied to the document.
        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, history)
            VALUES ($1, $2, $3, 'world', '[]'::jsonb)
            """,
            other_fact_id,
            bank_id,
            "unrelated fact",
        )

        # Observation consolidated from the document fact.
        await conn.execute(
            """
            INSERT INTO memory_units (
                id, bank_id, text, fact_type, source_memory_ids, history, proof_count
            )
            VALUES ($1, $2, $3, 'observation', $4::uuid[], '[]'::jsonb, 1)
            """,
            observation_id,
            bank_id,
            "observation from document",
            [fact_id],
        )

        # Observation that has no overlap with the document; should NOT match.
        await conn.execute(
            """
            INSERT INTO memory_units (
                id, bank_id, text, fact_type, source_memory_ids, history, proof_count
            )
            VALUES ($1, $2, $3, 'observation', $4::uuid[], '[]'::jsonb, 1)
            """,
            unrelated_observation_id,
            bank_id,
            "observation from elsewhere",
            [other_fact_id],
        )

    response = await api_client.get(
        f"/v1/default/banks/{bank_id}/graph",
        params={"type": "observation", "document_id": document_id},
    )
    assert response.status_code == 200
    data = response.json()
    returned_ids = {row["id"] for row in data["table_rows"]}
    assert str(observation_id) in returned_ids, (
        f"Observation linked via source memory should be returned; got {returned_ids}"
    )
    assert str(unrelated_observation_id) not in returned_ids, (
        "Unrelated observation must not leak into document-scoped results"
    )


@pytest.mark.asyncio
async def test_graph_q_filter_empty_results(api_client, test_bank_id):
    """The q filter returns empty results when no memory matches."""
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {"content": "Alice loves hiking."},
            ]
        },
    )
    assert response.status_code == 200

    response = await api_client.get(
        f"/v1/default/banks/{test_bank_id}/graph",
        params={"q": "zzznomatchzzz"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["table_rows"] == []
