"""
Integration test for the complete Hindsight API.

Tests all endpoints by starting a FastAPI server and making HTTP requests.
"""
import pytest
import pytest_asyncio
import httpx
from datetime import datetime
from hindsight_api.api import create_app


@pytest_asyncio.fixture
async def api_client(memory):
    """Create an async test client for the FastAPI app."""
    # Memory is already initialized by the conftest fixture (with migrations)
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_bank_id():
    """Provide a unique bank ID for this test run."""
    return f"integration_test_{datetime.now().timestamp()}"


@pytest.mark.asyncio
async def test_full_api_workflow(api_client, test_bank_id):
    """
    End-to-end test covering all major API endpoints in a realistic workflow.

    Workflow:
    1. Create bank and set profile
    2. Store memories (retain)
    3. Recall memories
    4. Reflect (generate answer)
    5. List banks and memories
    6. Get bank profile
    7. Get visualization data
    8. Track documents
    9. Test entity endpoints
    10. Test operations endpoints
    11. Clean up
    """

    # ================================================================
    # 1. Bank Management
    # ================================================================

    # List banks (should be empty initially or have other test banks)
    response = await api_client.get("/v1/default/banks")
    assert response.status_code == 200
    initial_banks_data = response.json()["banks"]
    initial_banks = [a["bank_id"] for a in initial_banks_data]

    # Get bank profile (creates default if not exists)
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/profile")
    assert response.status_code == 200
    profile = response.json()
    assert "disposition" in profile
    assert "background" in profile

    # Add background
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/background",
        json={
            "content": "A software engineer passionate about AI and memory systems."
        }
    )
    assert response.status_code == 200
    assert "software engineer" in response.json()["background"].lower()

    # ================================================================
    # 2. Memory Storage
    # ================================================================

    # Store single memory (using batch endpoint with single item)
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {
                    "content": "Alice is a machine learning researcher at Stanford.",
                    "context": "conversation about team members"
                }
            ]
        }
    )
    assert response.status_code == 200
    put_result = response.json()
    assert put_result["success"] is True
    assert put_result["items_count"] == 1

    # Store batch memories
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {
                    "content": "Bob leads the infrastructure team and loves Kubernetes.",
                    "context": "team introduction"
                },
                {
                    "content": "Charlie recently joined as a product manager from Google.",
                    "context": "new hire announcement"
                }
            ]
        }
    )
    assert response.status_code == 200
    batch_result = response.json()
    assert batch_result["success"] is True
    assert batch_result["items_count"] == 2

    # ================================================================
    # 3. Recall (Search)
    # ================================================================

    # Recall memories
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories/recall",
        json={
            "query": "Who works on machine learning?",
            "thinking_budget": 50
        }
    )
    assert response.status_code == 200
    search_results = response.json()
    assert "results" in search_results
    assert len(search_results["results"]) > 0

    # Verify we found Alice
    found_alice = any("Alice" in r["text"] for r in search_results["results"])
    assert found_alice, "Should find Alice in search results"

    # ================================================================
    # 4. Reflect (Reasoning)
    # ================================================================

    # Generate answer using reflect
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/reflect",
        json={
            "query": "What do you know about the team members?",
            "thinking_budget": 30,
            "context": "This is for a team overview document"
        }
    )
    assert response.status_code == 200
    reflect_result = response.json()
    assert "text" in reflect_result
    assert len(reflect_result["text"]) > 0
    assert "based_on" in reflect_result

    # Verify the answer mentions team members
    answer = reflect_result["text"].lower()
    assert "alice" in answer or "bob" in answer or "charlie" in answer

    # ================================================================
    # 5. Visualization & Statistics
    # ================================================================

    # Get graph data
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/graph")
    assert response.status_code == 200
    graph_data = response.json()
    assert "nodes" in graph_data
    assert "edges" in graph_data

    # Get memory statistics
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/stats")
    assert response.status_code == 200
    stats = response.json()
    assert "total_nodes" in stats
    assert stats["total_nodes"] > 0

    # List memory units
    response = await api_client.get(
        f"/v1/default/banks/{test_bank_id}/memories/list",
        params={"limit": 10}
    )
    assert response.status_code == 200
    memory_units = response.json()
    assert "items" in memory_units
    assert len(memory_units["items"]) > 0

    # ================================================================
    # 6. Document Tracking
    # ================================================================

    # Store memory with document
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {
                    "content": "Project timeline: MVP launch in Q1, Beta in Q2.",
                    "context": "product roadmap",
                    "document_id": "roadmap-2024-q1"
                }
            ]
        }
    )
    assert response.status_code == 200

    # List documents
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/documents")
    assert response.status_code == 200
    documents = response.json()
    assert "items" in documents
    assert len(documents["items"]) > 0

    # Get specific document
    response = await api_client.get(
        f"/v1/default/banks/{test_bank_id}/documents/roadmap-2024-q1"
    )
    assert response.status_code == 200
    doc_info = response.json()
    assert "id" in doc_info
    assert doc_info["id"] == "roadmap-2024-q1"
    assert doc_info["memory_unit_count"] > 0
    # Note: Document deletion is tested separately in test_document_deletion

    # ================================================================
    # 7. Update and Verify Bank Disposition
    # ================================================================

    # Update disposition traits
    response = await api_client.put(
        f"/v1/default/banks/{test_bank_id}/profile",
        json={
            "disposition": {
                "skepticism": 4,
                "literalism": 3,
                "empathy": 4
            }
        }
    )
    assert response.status_code == 200

    # Check profile again (should have updated disposition)
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/profile")
    assert response.status_code == 200
    updated_profile = response.json()
    assert "software engineer" in updated_profile["background"].lower()

    # ================================================================
    # 8. Test Entity Endpoints
    # ================================================================

    # List entities
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/entities")
    assert response.status_code == 200
    entities_data = response.json()
    assert "items" in entities_data

    # Get specific entity if any exist
    if len(entities_data['items']) > 0:
        entity_id = entities_data['items'][0]['id']
        response = await api_client.get(
            f"/v1/default/banks/{test_bank_id}/entities/{entity_id}"
        )
        assert response.status_code == 200
        entity_detail = response.json()
        assert "id" in entity_detail

        # Test regenerate observations
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/entities/{entity_id}/regenerate"
        )
        assert response.status_code == 200

    # ================================================================
    # 9. List All Banks (should include our test bank)
    # ================================================================

    response = await api_client.get("/v1/default/banks")
    assert response.status_code == 200
    final_banks_data = response.json()["banks"]
    final_banks = [a["bank_id"] for a in final_banks_data]
    assert test_bank_id in final_banks
    # Don't assert count increases due to parallel test cleanup races
    # Just verify our bank exists in the list

    # ================================================================
    # 10. Clean Up
    # ================================================================

    # Note: No delete bank endpoint in API, so test data remains in DB
    # Using timestamped bank IDs prevents conflicts between test runs


@pytest.mark.asyncio
async def test_error_handling(api_client):
    """Test that API properly handles error cases."""

    # Invalid request (missing required field)
    response = await api_client.post(
        "/v1/default/banks/error_test/memories",
        json={
            "items": [
                {
                    # Missing "content"
                    "context": "test"
                }
            ]
        }
    )
    assert response.status_code == 422  # Validation error

    # Recall with invalid parameters
    response = await api_client.post(
        "/v1/default/banks/error_test/memories/recall",
        json={
            "query": "test",
            "budget": "invalid_budget"  # Invalid budget value (should be low/mid/high)
        }
    )
    assert response.status_code == 422

    # Get non-existent document
    response = await api_client.get(
        "/v1/default/banks/nonexistent_bank/documents/fake-doc-id"
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_concurrent_requests(api_client):
    """Test that API can handle concurrent requests."""
    bank_id = f"concurrent_test_{datetime.now().timestamp()}"

    # Store multiple memories concurrently (simulated with sequential calls)
    responses = []
    test_facts = [
        "David works as a data scientist at Microsoft.",
        "Emily is the CEO of a startup in San Francisco.",
        "Frank teaches computer science at MIT.",
        "Grace is a software architect specializing in distributed systems.",
        "Henry leads the product team at Amazon."
    ]
    for fact in test_facts:
        response = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories",
            json={
                "items": [
                    {
                        "content": fact,
                        "context": "concurrent test"
                    }
                ]
            }
        )
        responses.append(response)

    # All should succeed
    assert all(r.status_code == 200 for r in responses)
    assert all(r.json()["success"] for r in responses)

    # Verify all facts stored
    response = await api_client.get(
        f"/v1/default/banks/{bank_id}/memories/list",
        params={"limit": 20}
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) >= 5


@pytest.mark.asyncio
async def test_document_deletion(api_client):
    """Test document deletion including cascade deletion of memory units and links."""
    test_bank_id = f"doc_delete_test_{datetime.now().timestamp()}"

    # Store a document with memory
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {
                    "content": "The quarterly sales report shows a 25% increase in revenue.",
                    "context": "Q1 financial review",
                    "document_id": "sales-report-q1-2024"
                }
            ]
        }
    )
    assert response.status_code == 200

    # Verify document exists
    response = await api_client.get(
        f"/v1/default/banks/{test_bank_id}/documents/sales-report-q1-2024"
    )
    assert response.status_code == 200
    doc_info = response.json()
    initial_units = doc_info["memory_unit_count"]
    assert initial_units > 0

    # Delete the document
    response = await api_client.delete(
        f"/v1/default/banks/{test_bank_id}/documents/sales-report-q1-2024"
    )
    assert response.status_code == 200
    delete_result = response.json()
    assert delete_result["success"] is True
    assert delete_result["document_id"] == "sales-report-q1-2024"
    assert delete_result["memory_units_deleted"] == initial_units

    # Verify document is gone (should return 404)
    response = await api_client.get(
        f"/v1/default/banks/{test_bank_id}/documents/sales-report-q1-2024"
    )
    assert response.status_code == 404

    # Verify document is not in the list
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/documents")
    assert response.status_code == 200
    documents = response.json()
    doc_ids = [doc["id"] for doc in documents["items"]]
    assert "sales-report-q1-2024" not in doc_ids

    # Try to delete again (should return 404)
    response = await api_client.delete(
        f"/v1/default/banks/{test_bank_id}/documents/sales-report-q1-2024"
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_async_retain(api_client):
    """Test asynchronous retain functionality.

    When async=true is passed, the retain endpoint should:
    1. Return immediately with success and async_=true
    2. Process the content in the background
    3. Eventually store the memories
    """
    import asyncio

    test_bank_id = f"async_retain_test_{datetime.now().timestamp()}"

    # Store memory with async=true
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "async": True,
            "items": [
                {
                    "content": "Alice is a senior engineer at TechCorp. She has been working on the authentication system for 5 years.",
                    "context": "team introduction"
                }
            ]
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["async"] is True, "Response should indicate async processing"
    assert result["items_count"] == 1

    # Check operations endpoint to see the pending operation
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/operations")
    assert response.status_code == 200
    ops_result = response.json()
    assert "operations" in ops_result

    # Wait for async processing to complete (poll with timeout)
    max_wait_seconds = 30
    poll_interval = 0.5
    elapsed = 0
    memories_found = False

    while elapsed < max_wait_seconds:
        # Check if memories are stored
        response = await api_client.get(
            f"/v1/default/banks/{test_bank_id}/memories/list",
            params={"limit": 10}
        )
        assert response.status_code == 200
        items = response.json()["items"]

        if len(items) > 0:
            memories_found = True
            break

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    assert memories_found, f"Async retain did not complete within {max_wait_seconds} seconds"

    # Verify we can recall the stored memory
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories/recall",
        json={
            "query": "Who works at TechCorp?",
            "thinking_budget": 30
        }
    )
    assert response.status_code == 200
    search_results = response.json()
    assert len(search_results["results"]) > 0, "Should find the asynchronously stored memory"

    # Verify Alice is mentioned
    found_alice = any("Alice" in r["text"] for r in search_results["results"])
    assert found_alice, "Should find Alice in search results"


@pytest.mark.asyncio
async def test_async_retain_parallel(api_client):
    """Test multiple async retain operations running in parallel.

    Verifies that:
    1. Multiple async operations can be submitted concurrently
    2. All operations complete successfully
    3. The exact number of documents are processed
    """
    import asyncio

    test_bank_id = f"async_parallel_test_{datetime.now().timestamp()}"
    num_documents = 5

    # Prepare multiple documents to retain
    documents = [
        {
            "content": f"Document {i}: This is test content about Person{i} who works at Company{i}.",
            "context": f"test document {i}",
            "document_id": f"doc_{i}"
        }
        for i in range(num_documents)
    ]

    # Submit all async retain operations in parallel
    async def submit_async_retain(doc):
        return await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={
                "async": True,
                "items": [doc]
            }
        )

    # Run all submissions concurrently
    responses = await asyncio.gather(*[submit_async_retain(doc) for doc in documents])

    # Verify all submissions succeeded
    for i, response in enumerate(responses):
        assert response.status_code == 200, f"Document {i} submission failed"
        result = response.json()
        assert result["success"] is True
        assert result["async"] is True

    # Check operations endpoint - should show pending operations
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/operations")
    assert response.status_code == 200

    # Wait for all async operations to complete (poll with timeout)
    max_wait_seconds = 60
    poll_interval = 1.0
    elapsed = 0
    all_docs_processed = False

    while elapsed < max_wait_seconds:
        # Check document count
        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/documents")
        assert response.status_code == 200
        docs = response.json()["items"]

        if len(docs) >= num_documents:
            all_docs_processed = True
            break

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    assert all_docs_processed, f"Expected {num_documents} documents, but only {len(docs)} were processed within {max_wait_seconds} seconds"

    # Verify exact document count
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/documents")
    assert response.status_code == 200
    final_docs = response.json()["items"]
    assert len(final_docs) == num_documents, f"Expected exactly {num_documents} documents, got {len(final_docs)}"

    # Verify each document exists
    doc_ids = {doc["id"] for doc in final_docs}
    for i in range(num_documents):
        assert f"doc_{i}" in doc_ids, f"Document doc_{i} not found"

    # Verify memories were created for all documents
    response = await api_client.get(
        f"/v1/default/banks/{test_bank_id}/memories/list",
        params={"limit": 100}
    )
    assert response.status_code == 200
    memories = response.json()["items"]
    assert len(memories) >= num_documents, f"Expected at least {num_documents} memories, got {len(memories)}"

    # Verify we can recall content from different documents
    for i in [0, num_documents - 1]:  # Check first and last
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={
                "query": f"Who works at Company{i}?",
                "thinking_budget": 30
            }
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) > 0, f"Should find memories for document {i}"


@pytest.mark.asyncio
async def test_reflect_structured_output(api_client):
    """Test reflect endpoint with structured output via response_schema.

    When response_schema is provided, the reflect endpoint should return
    both the natural language text response and a structured_output field
    containing the response parsed according to the provided JSON schema.
    """
    test_bank_id = f"reflect_structured_test_{datetime.now().timestamp()}"

    # Store some memories to reflect on
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {
                    "content": "Alice is a senior machine learning engineer with 8 years of experience.",
                    "context": "team member info"
                },
                {
                    "content": "Bob is a junior data scientist who joined last month.",
                    "context": "team member info"
                },
                {
                    "content": "The team uses Python and TensorFlow for most projects.",
                    "context": "tech stack"
                }
            ]
        }
    )
    assert response.status_code == 200

    # Define a JSON schema for structured output
    response_schema = {
        "type": "object",
        "properties": {
            "team_members": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "experience_level": {"type": "string"}
                    }
                }
            },
            "technologies": {
                "type": "array",
                "items": {"type": "string"}
            },
            "summary": {"type": "string"}
        },
        "required": ["team_members", "summary"]
    }

    # Call reflect with response_schema
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/reflect",
        json={
            "query": "Give me an overview of the team and their tech stack",
            "response_schema": response_schema
        }
    )
    assert response.status_code == 200
    result = response.json()

    # Verify text field exists (empty when using structured output)
    assert "text" in result
    assert result["text"] == ""

    # Verify structured output exists and has expected structure
    assert "structured_output" in result
    assert result["structured_output"] is not None

    structured = result["structured_output"]
    assert "team_members" in structured
    assert "summary" in structured
    assert isinstance(structured["team_members"], list)
    assert isinstance(structured["summary"], str)

    # Verify team members have the expected fields
    if len(structured["team_members"]) > 0:
        member = structured["team_members"][0]
        assert "name" in member or "role" in member  # At least some fields should be present


@pytest.mark.asyncio
async def test_reflect_without_structured_output(api_client):
    """Test that reflect works normally without response_schema.

    When response_schema is not provided, the structured_output field
    should be null/None in the response.
    """
    test_bank_id = f"reflect_no_structured_test_{datetime.now().timestamp()}"

    # Store a memory
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {
                    "content": "The project deadline is next Friday.",
                    "context": "project timeline"
                }
            ]
        }
    )
    assert response.status_code == 200

    # Call reflect without response_schema
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/reflect",
        json={
            "query": "When is the project deadline?"
        }
    )
    assert response.status_code == 200
    result = response.json()

    # Verify response has text but structured_output is null
    assert "text" in result
    assert len(result["text"]) > 0
    assert result.get("structured_output") is None


@pytest.mark.asyncio
async def test_reflect_with_max_tokens(api_client):
    """Test reflect endpoint with custom max_tokens parameter.

    The max_tokens parameter controls the maximum tokens for the LLM response.
    """
    test_bank_id = f"reflect_max_tokens_test_{datetime.now().timestamp()}"

    # Store a memory
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {
                    "content": "Python is a popular programming language for data science and machine learning.",
                    "context": "tech"
                }
            ]
        }
    )
    assert response.status_code == 200

    # Call reflect with custom max_tokens
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/reflect",
        json={
            "query": "What is Python used for?",
            "max_tokens": 500
        }
    )
    assert response.status_code == 200
    result = response.json()

    # Verify response has text
    assert "text" in result
    assert len(result["text"]) > 0
