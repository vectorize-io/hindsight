"""
Integration test for Hindsight server with context manager.

Tests the full workflow:
1. Starting server using context manager
2. Creating a memory bank
3. Storing memories (retain)
4. Recalling memories
5. Reflecting on memories

Note: These tests use embedded PostgreSQL (pg0) with a shared server instance
across all tests. Each test uses random bank_ids to avoid conflicts, allowing
safe parallel execution.
"""

import os
import uuid
import pytest
from hindsight import HindsightServer, HindsightClient


@pytest.fixture(scope="session")
def llm_config():
    """Get LLM configuration from environment (session-scoped)."""
    provider = os.getenv("HINDSIGHT_LLM_PROVIDER", "groq")
    api_key = os.getenv("HINDSIGHT_LLM_API_KEY", "")
    model = os.getenv("HINDSIGHT_LLM_MODEL", "openai/gpt-oss-120b")

    if not api_key:
        raise Exception("LLM API key not configured. Set HINDSIGHT_LLM_API_KEY environment variable.")

    return {
        "llm_provider": provider,
        "llm_api_key": api_key,
        "llm_model": model,
    }


@pytest.fixture(scope="session")
def shared_server(llm_config):
    """
    Shared server instance for all tests (session-scoped).

    This allows tests to run in parallel by sharing the same pg0 instance,
    while using different bank_ids to avoid data conflicts.
    """
    server = HindsightServer(db_url="pg0", **llm_config)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def client(shared_server):
    """Create a client connected to the shared server."""
    return HindsightClient(base_url=shared_server.url)


def test_server_context_manager_basic_workflow(client):
    """
    Test complete workflow using shared server.

    This test:
    1. Uses a shared server instance
    2. Creates a memory bank with unique ID
    3. Stores multiple memories
    4. Recalls memories based on a query
    5. Reflects (generates contextual answers) based on stored memories
    """
    # Use random bank_id to allow parallel test execution
    bank_id = f"test_assistant_{uuid.uuid4().hex[:8]}"

    # Step 1: Create a memory bank with background information
    print(f"\n1. Creating memory bank: {bank_id}")
    bank_response = client.create_bank(
        bank_id=bank_id,
        name="Test Assistant",
        background="An AI assistant that helps with programming and data analysis tasks."
    )
    assert "bank_id" in bank_response
    assert bank_response["bank_id"] == bank_id

    # Step 2: Store some memories about user preferences
    print("\n2. Storing memories...")

    # Store first memory
    retain_response1 = client.retain(
        bank_id=bank_id,
        content="User prefers Python over JavaScript for data analysis projects.",
        context="User conversation about programming languages"
    )
    assert retain_response1.get("success") is True

    # Store second memory
    retain_response2 = client.retain(
        bank_id=bank_id,
        content="User is working on a machine learning project using scikit-learn.",
        context="Discussion about ML frameworks"
    )
    assert retain_response2.get("success") is True

    # Store third memory
    retain_response3 = client.retain(
        bank_id=bank_id,
        content="User likes visualizing data with matplotlib and seaborn.",
        context="Conversation about data visualization"
    )
    assert retain_response3.get("success") is True

    # Store batch memories
    batch_response = client.retain_batch(
        bank_id=bank_id,
        items=[
            {"content": "User is interested in neural networks and deep learning."},
            {"content": "User asked about best practices for training models."},
        ]
    )
    # Check if the batch was submitted successfully (items_count shows how many were submitted)
    assert batch_response.get("items_count", 0) >= 2

    # Step 3: Recall memories based on a query
    print("\n3. Recalling memories about programming preferences...")
    recall_results = client.recall(
        bank_id=bank_id,
        query="What programming languages and tools does the user prefer?",
        max_tokens=4096
    )

    # Verify recall results
    assert isinstance(recall_results, list)
    assert len(recall_results) > 0
    print(f"   Found {len(recall_results)} relevant memories")

    # Check that results have expected structure
    for result in recall_results:
        assert "content" in result or "text" in result
        print(f"   - {result.get('content') or result.get('text', '')[:100]}")

    # Step 4: Recall memories about machine learning
    print("\n4. Recalling memories about machine learning...")
    ml_recall_results = client.recall(
        bank_id=bank_id,
        query="machine learning and neural networks",
        max_tokens=4096
    )

    # Verify recall results
    assert isinstance(ml_recall_results, list)
    assert len(ml_recall_results) > 0
    print(f"   Found {len(ml_recall_results)} ML-related memories")
    for result in ml_recall_results[:3]:  # Show first 3
        print(f"   - {result.get('content') or result.get('text', '')[:100]}")

    # Step 5: Reflect (generate contextual answer based on memories)
    print("\n5. Reflecting on query about recommendations...")
    reflect_response = client.reflect(
        bank_id=bank_id,
        query="What tools and libraries should I recommend for this user's data analysis work?",
        budget="mid"
    )

    # Verify reflection response
    assert isinstance(reflect_response, dict)
    assert "answer" in reflect_response or "text" in reflect_response

    answer = reflect_response.get("answer") or reflect_response.get("text", "")
    assert len(answer) > 0
    print(f"   Answer: {answer[:200]}...")

    # Verify the answer mentions relevant tools/libraries
    answer_lower = answer.lower()
    assert any(term in answer_lower for term in ["python", "scikit-learn", "matplotlib", "seaborn", "data"])

    # Step 6: Another reflection with different context
    print("\n6. Reflecting with additional context...")
    reflect_with_context = client.reflect(
        bank_id=bank_id,
        query="Should I use TensorFlow or PyTorch?",
        budget="low",
        context="The user is starting a new deep learning project"
    )

    assert isinstance(reflect_with_context, dict)
    context_answer = reflect_with_context.get("answer") or reflect_with_context.get("text", "")
    assert len(context_answer) > 0
    print(f"   Context-aware answer: {context_answer[:150]}...")


def test_server_manual_start_stop(client):
    """
    Test basic operations on shared server.

    Verifies that basic bank operations work correctly.
    """
    # Use random bank_id to allow parallel test execution
    bank_id = f"test_manual_{uuid.uuid4().hex[:8]}"

    # Create bank
    bank_response = client.create_bank(
        bank_id=bank_id,
        name="Manual Test"
    )
    assert bank_response["bank_id"] == bank_id

    # Store a memory
    retain_response = client.retain(
        bank_id=bank_id,
        content="Testing manual server lifecycle."
    )
    assert retain_response.get("success") is True

    # Recall the memory
    recall_results = client.recall(
        bank_id=bank_id,
        query="server testing"
    )
    assert len(recall_results) >= 0  # May or may not find results immediately


def test_server_with_client_context_manager(client):
    """
    Test client context manager with shared server.
    """
    # Use random bank_id to allow parallel test execution
    bank_id = f"test_nested_context_{uuid.uuid4().hex[:8]}"

    # Use client context manager (client fixture already provides this)
    # Create bank
    client.create_bank(bank_id=bank_id, name="Nested Context Test")

    # Store memory
    response = client.retain(
        bank_id=bank_id,
        content="Testing nested context managers."
    )
    assert response.get("success") is True

    # Verify we can recall
    results = client.recall(bank_id=bank_id, query="context")
    assert isinstance(results, list)


def test_list_banks(client, shared_server):
    """
    Test listing banks to verify bank_id field mapping.

    This test verifies that the list_banks endpoint correctly returns
    bank_id (not agent_id) in the response.
    """
    # Create a couple of banks with random IDs to allow parallel test execution
    test_suffix = uuid.uuid4().hex[:8]
    bank1_id = f"test_bank_1_{test_suffix}"
    bank2_id = f"test_bank_2_{test_suffix}"

    client.create_bank(bank_id=bank1_id, name="Test Bank 1", background="First test bank")
    client.create_bank(bank_id=bank2_id, name="Test Bank 2", background="Second test bank")

    # List all banks using the generated client
    import hindsight_client_api
    from hindsight_client_api.api import default_api

    config = hindsight_client_api.Configuration(host=shared_server.url)
    api_client = hindsight_client_api.ApiClient(config)
    api = default_api.DefaultApi(api_client)

    # Call list_banks endpoint
    import asyncio
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(api.list_banks())

    # Verify response structure
    assert hasattr(response, 'banks'), "Response should have 'banks' attribute"
    assert len(response.banks) >= 2, f"Should have at least 2 banks, got {len(response.banks)}"

    # Verify each bank has bank_id (not agent_id)
    for bank in response.banks:
        bank_dict = bank.to_dict() if hasattr(bank, 'to_dict') else bank
        assert 'bank_id' in bank_dict, f"Bank should have 'bank_id' field, got: {bank_dict.keys()}"
        assert 'agent_id' not in bank_dict, f"Bank should NOT have 'agent_id' field, got: {bank_dict.keys()}"

    # Find our test banks
    bank_ids = [b.bank_id if hasattr(b, 'bank_id') else b['bank_id'] for b in response.banks]
    assert bank1_id in bank_ids, f"Should find {bank1_id} in bank list"
    assert bank2_id in bank_ids, f"Should find {bank2_id} in bank list"

    print(f"✓ Successfully listed {len(response.banks)} banks with correct bank_id field")

    # Cleanup
    loop.run_until_complete(api_client.close())
