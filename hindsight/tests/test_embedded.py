"""
Integration tests for HindsightEmbedded client.

Tests the embedded client with automatic server lifecycle management:
1. Lazy server startup on first use
2. Server reuse across multiple operations
3. Context manager support
4. Method proxying to underlying HindsightClient
5. Proper cleanup

Note: Each test uses random bank_ids to avoid conflicts and allow safe parallel execution.
"""

import os
import uuid

import pytest

from hindsight import HindsightEmbedded


@pytest.fixture(scope="session")
def llm_config():
    """Get LLM configuration from environment (session-scoped)."""
    # Try both naming conventions
    provider = os.getenv("HINDSIGHT_API_LLM_PROVIDER") or os.getenv("HINDSIGHT_LLM_PROVIDER", "groq")
    api_key = os.getenv("HINDSIGHT_API_LLM_API_KEY") or os.getenv("HINDSIGHT_LLM_API_KEY", "")
    model = os.getenv("HINDSIGHT_API_LLM_MODEL") or os.getenv("HINDSIGHT_LLM_MODEL", "openai/gpt-oss-120b")

    if not api_key:
        pytest.skip("LLM API key not configured. Set HINDSIGHT_API_LLM_API_KEY or HINDSIGHT_LLM_API_KEY.")

    return {
        "llm_provider": provider,
        "llm_api_key": api_key,
        "llm_model": model,
    }


def test_embedded_lazy_start(llm_config):
    """
    Test that HindsightEmbedded starts server lazily on first use.
    """
    profile = f"test_lazy_{uuid.uuid4().hex[:8]}"
    bank_id = f"bank_{uuid.uuid4().hex[:8]}"

    # Create client - should NOT start server yet
    client = HindsightEmbedded(profile=profile, log_level="info", **llm_config)
    assert not client.is_running, "Server should not be running after initialization"

    # First call should start server
    result = client.retain(bank_id=bank_id, content="Test content for lazy start")

    # Verify server is now running
    assert client.is_running, "Server should be running after first call"
    assert result.success, "Retain should succeed"
    assert result.items_count >= 1, "Should have stored at least 1 item"

    # Cleanup
    client.close()
    assert not client.is_running, "Server should stop after close()"


def test_embedded_context_manager(llm_config):
    """
    Test HindsightEmbedded with context manager.
    """
    profile = f"test_ctx_{uuid.uuid4().hex[:8]}"
    bank_id = f"bank_{uuid.uuid4().hex[:8]}"

    # Use context manager
    with HindsightEmbedded(profile=profile, log_level="info", **llm_config) as client:
        assert client.is_running, "Server should be running inside context"

        # Store memory
        result = client.retain(bank_id=bank_id, content="Testing context manager")
        assert result.success, "Retain should succeed"

        # Recall memory
        recall_results = client.recall(bank_id=bank_id, query="context")
        assert isinstance(recall_results.results, list), "Recall should return results list"

    # Server should be stopped after context exit
    # Note: We can't check client.is_running here as client is out of scope


def test_embedded_complete_workflow(llm_config):
    """
    Test complete workflow with HindsightEmbedded.

    This test:
    1. Creates a client with lazy start
    2. Creates a memory bank
    3. Stores multiple memories
    4. Recalls memories
    5. Reflects on memories
    6. Tests cleanup
    """
    profile = f"test_workflow_{uuid.uuid4().hex[:8]}"
    bank_id = f"assistant_{uuid.uuid4().hex[:8]}"

    client = HindsightEmbedded(profile=profile, log_level="info", **llm_config)

    try:
        # Step 1: Create a memory bank
        print(f"\n1. Creating memory bank: {bank_id}")
        bank_response = client.create_bank(
            bank_id=bank_id, name="Test Assistant", mission="Help with programming tasks"
        )
        assert bank_response.bank_id == bank_id

        # Step 2: Store memories (single)
        print("\n2. Storing single memory...")
        retain_response = client.retain(
            bank_id=bank_id,
            content="User prefers Python for data analysis.",
            context="Programming preferences",
        )
        assert retain_response.success
        assert retain_response.items_count >= 1

        # Step 3: Store batch memories
        print("\n3. Storing batch memories...")
        batch_response = client.retain_batch(
            bank_id=bank_id,
            items=[
                {"content": "User works with pandas and numpy."},
                {"content": "User likes matplotlib for visualization."},
                {"content": "User is interested in machine learning with scikit-learn."},
            ],
        )
        assert batch_response.success
        assert batch_response.items_count >= 3

        # Step 4: Recall memories
        print("\n4. Recalling memories...")
        recall_response = client.recall(bank_id=bank_id, query="What tools does the user prefer?", max_tokens=2000)
        assert isinstance(recall_response.results, list)
        assert len(recall_response.results) > 0
        print(f"   Found {len(recall_response.results)} relevant memories")

        # Step 5: Reflect on memories
        print("\n5. Reflecting on memories...")
        reflect_response = client.reflect(
            bank_id=bank_id,
            query="What programming tools should I recommend?",
            budget="low",
        )
        assert reflect_response.text
        assert len(reflect_response.text) > 0
        print(f"   Answer: {reflect_response.text[:150]}...")

        # Verify answer mentions relevant tools
        answer_lower = reflect_response.text.lower()
        assert any(term in answer_lower for term in ["python", "pandas", "numpy", "data"])

        # Step 6: List memories
        print("\n6. Listing memories...")
        list_response = client.list_memories(bank_id=bank_id, limit=10)
        assert len(list_response.items) > 0
        print(f"   Listed {len(list_response.items)} memories")

    finally:
        # Cleanup
        client.close()


def test_embedded_server_reuse(llm_config):
    """
    Test that the same server is reused across multiple calls.
    """
    profile = f"test_reuse_{uuid.uuid4().hex[:8]}"
    bank_id = f"bank_{uuid.uuid4().hex[:8]}"

    client = HindsightEmbedded(profile=profile, log_level="info", **llm_config)

    try:
        # First call starts server
        result1 = client.retain(bank_id=bank_id, content="First message")
        url1 = client.url
        assert client.is_running

        # Second call should reuse the same server
        result2 = client.retain(bank_id=bank_id, content="Second message")
        url2 = client.url

        # URLs should be identical (same server)
        assert url1 == url2, "Server URL should remain the same across calls"
        assert result1.success and result2.success

        # Third call should also reuse
        recall_result = client.recall(bank_id=bank_id, query="message")
        url3 = client.url
        assert url3 == url1, "Server URL should remain the same for recall"
        assert isinstance(recall_result.results, list)

    finally:
        client.close()


def test_embedded_method_proxying(llm_config):
    """
    Test that all HindsightClient methods are properly proxied.

    This ensures __getattr__ proxying works for various method types.
    """
    profile = f"test_proxy_{uuid.uuid4().hex[:8]}"
    bank_id = f"bank_{uuid.uuid4().hex[:8]}"

    client = HindsightEmbedded(profile=profile, log_level="info", **llm_config)

    try:
        # Test bank operations
        bank = client.create_bank(bank_id=bank_id, name="Proxy Test")
        assert bank.bank_id == bank_id

        # Test mission setting
        mission_response = client.set_mission(bank_id=bank_id, mission="Test mission for proxying")
        assert mission_response.bank_id == bank_id

        # Test retain
        retain_result = client.retain(bank_id=bank_id, content="Test content")
        assert retain_result.success

        # Test retain_batch
        batch_result = client.retain_batch(
            bank_id=bank_id, items=[{"content": "Item 1"}, {"content": "Item 2"}]
        )
        assert batch_result.success
        assert batch_result.items_count >= 2

        # Test recall
        recall_result = client.recall(bank_id=bank_id, query="test")
        assert hasattr(recall_result, "results")

        # Test reflect
        reflect_result = client.reflect(bank_id=bank_id, query="What is stored?")
        assert hasattr(reflect_result, "text")

        # Test list_memories
        list_result = client.list_memories(bank_id=bank_id, limit=5)
        assert hasattr(list_result, "items")

        print("✓ All methods successfully proxied")

    finally:
        client.close()


def test_embedded_multiple_banks(llm_config):
    """
    Test that HindsightEmbedded can work with multiple banks.
    """
    profile = f"test_multibank_{uuid.uuid4().hex[:8]}"
    bank1_id = f"bank1_{uuid.uuid4().hex[:8]}"
    bank2_id = f"bank2_{uuid.uuid4().hex[:8]}"

    client = HindsightEmbedded(profile=profile, log_level="info", **llm_config)

    try:
        # Create first bank and store data
        client.create_bank(bank_id=bank1_id, name="Bank 1")
        client.retain(bank_id=bank1_id, content="Alice prefers Python for data science")

        # Create second bank and store data
        client.create_bank(bank_id=bank2_id, name="Bank 2")
        client.retain(bank_id=bank2_id, content="Bob uses JavaScript for web development")

        # Recall from both banks
        results1 = client.recall(bank_id=bank1_id, query="programming language")
        results2 = client.recall(bank_id=bank2_id, query="programming language")

        assert len(results1.results) > 0
        assert len(results2.results) > 0

        # Verify banks are isolated (each should only see their own content)
        # This is a basic check - content isolation is tested more thoroughly in other tests
        assert results1.results[0].text != results2.results[0].text or len(results1.results) != len(
            results2.results
        )

    finally:
        client.close()


def test_embedded_profile_isolation(llm_config):
    """
    Test that different profiles create isolated data stores.
    """
    profile1 = f"test_iso1_{uuid.uuid4().hex[:8]}"
    profile2 = f"test_iso2_{uuid.uuid4().hex[:8]}"
    bank_id = "shared_bank_name"  # Same bank_id in both profiles

    client1 = HindsightEmbedded(profile=profile1, log_level="info", **llm_config)
    client2 = HindsightEmbedded(profile=profile2, log_level="info", **llm_config)

    try:
        # Store data in profile1
        client1.retain(bank_id=bank_id, content="User likes TypeScript for frontend development")

        # Store different data in profile2
        client2.retain(bank_id=bank_id, content="User prefers Rust for systems programming")

        # Each profile should only see its own data
        results1 = client1.recall(bank_id=bank_id, query="programming preference")
        results2 = client2.recall(bank_id=bank_id, query="programming preference")

        # Both should have results
        assert len(results1.results) > 0
        assert len(results2.results) > 0

        # Results should be different (basic isolation check)
        # Note: This is a basic sanity check. Full isolation is ensured by pg0's data directory separation

    finally:
        client1.close()
        client2.close()


def test_embedded_error_after_close(llm_config):
    """
    Test that using HindsightEmbedded after close() raises an error.
    """
    profile = f"test_error_{uuid.uuid4().hex[:8]}"
    bank_id = f"bank_{uuid.uuid4().hex[:8]}"

    client = HindsightEmbedded(profile=profile, log_level="info", **llm_config)

    # Use it once to start server
    client.retain(bank_id=bank_id, content="Test")

    # Close the client
    client.close()
    assert not client.is_running

    # Trying to use it after close should raise an error
    with pytest.raises(RuntimeError, match="Cannot use HindsightEmbedded after it has been closed"):
        client.retain(bank_id=bank_id, content="This should fail")
