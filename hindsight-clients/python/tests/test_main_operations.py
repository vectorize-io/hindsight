"""
Tests for Hindsight Python client.

These tests require a running Hindsight API server.
"""

import os
import uuid
import pytest
from datetime import datetime
from hindsight_client import Hindsight


# Test configuration
HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")


@pytest.fixture
def client():
    """Create a Hindsight client for testing."""
    with Hindsight(base_url=HINDSIGHT_API_URL) as client:
        yield client


@pytest.fixture
def bank_id():
    """Provide a unique test bank ID for each test."""
    return f"test_bank_{uuid.uuid4().hex[:12]}"


class TestRetain:
    """Tests for storing memories."""

    def test_retain_single_memory(self, client, bank_id):
        """Test storing a single memory."""
        response = client.retain(
            bank_id=bank_id,
            content="Alice loves artificial intelligence and machine learning",
        )

        assert response is not None
        assert response.success is True

    def test_retain_memory_with_context(self, client, bank_id):
        """Test storing a memory with context and timestamp."""
        response = client.retain(
            bank_id=bank_id,
            content="Bob went hiking in the mountains",
            timestamp=datetime(2024, 1, 15, 10, 30),
            context="outdoor activities",
        )

        assert response is not None
        assert response.success is True

    def test_retain_batch_memories(self, client, bank_id):
        """Test storing multiple memories in batch."""
        items = [
            {"content": "Charlie enjoys reading science fiction books"},
            {"content": "Diana is learning to play the guitar", "context": "hobbies"},
            {
                "content": "Eve completed a marathon last month",
                "event_date": datetime(2024, 10, 15),
            },
        ]

        response = client.retain_batch(
            bank_id=bank_id,
            items=items,
        )

        assert response is not None
        assert response.success is True
        assert response.items_count == 3


class TestRecall:
    """Tests for searching memories."""

    @pytest.fixture(autouse=True)
    def setup_memories(self, client, bank_id):
        """Setup: Store some test memories before search tests."""
        client.retain_batch(
            bank_id=bank_id,
            items=[
                {"content": "Alice loves programming in Python"},
                {"content": "Bob enjoys hiking and outdoor adventures"},
                {"content": "Charlie is interested in quantum physics"},
                {"content": "Diana plays the violin beautifully"},
            ],
        )

    def test_recall_basic(self, client, bank_id):
        """Test basic memory search."""
        response = client.recall(
            bank_id=bank_id,
            query="What does Alice like?",
        )

        assert response is not None
        assert response.results is not None
        assert len(response.results) > 0

        # Check that at least one result contains relevant information
        result_texts = [r.text for r in response.results]
        assert any("Alice" in text or "Python" in text or "programming" in text for text in result_texts)

    def test_recall_with_max_tokens(self, client, bank_id):
        """Test search with token limit."""
        response = client.recall(
            bank_id=bank_id,
            query="outdoor activities",
            max_tokens=1024,
        )

        assert response is not None
        assert response.results is not None

    def test_recall_full_featured(self, client, bank_id):
        """Test recall with all features."""
        response = client.recall(
            bank_id=bank_id,
            query="What are people's hobbies?",
            types=["world"],
            max_tokens=2048,
            trace=True,
        )

        assert response is not None
        assert response.results is not None


class TestReflect:
    """Tests for thinking/reasoning operations."""

    @pytest.fixture(autouse=True)
    def setup_memories(self, client, bank_id):
        """Setup: Store some test memories and bank background."""
        client.create_bank(
            bank_id=bank_id,
            background="I am a helpful AI assistant interested in technology and science.",
        )

        client.retain_batch(
            bank_id=bank_id,
            items=[
                {"content": "The Python programming language is great for data science"},
                {"content": "Machine learning models can recognize patterns in data"},
                {"content": "Neural networks are inspired by biological neurons"},
            ],
        )

    def test_reflect_basic(self, client, bank_id):
        """Test basic reflect operation."""
        response = client.reflect(
            bank_id=bank_id,
            query="What do you think about artificial intelligence?",
        )

        assert response is not None
        assert response.text is not None
        assert len(response.text) > 0

    def test_reflect_with_context(self, client, bank_id):
        """Test reflect with additional context."""
        response = client.reflect(
            bank_id=bank_id,
            query="Should I learn Python?",
            context="I'm interested in starting a career in data science",
        )

        assert response is not None
        assert response.text is not None
        assert len(response.text) > 0


class TestListMemories:
    """Tests for listing memories."""

    @pytest.fixture(autouse=True)
    def setup_memories(self, client, bank_id):
        """Setup: Store some test memories synchronously."""
        client.retain_batch(
            bank_id=bank_id,
            items=[
                {"content": f"Alice likes topic number {i}"} for i in range(5)
            ],
            retain_async=False,  # Wait for fact extraction to complete
        )

    def test_list_all_memories(self, client, bank_id):
        """Test listing all memories."""
        response = client.list_memories(bank_id=bank_id)

        assert response is not None
        assert response.items is not None
        assert response.total is not None
        assert len(response.items) > 0

    def test_list_with_pagination(self, client, bank_id):
        """Test listing with pagination."""
        response = client.list_memories(
            bank_id=bank_id,
            limit=2,
            offset=0,
        )

        assert response is not None
        assert response.items is not None
        assert len(response.items) <= 2


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_complete_workflow(self, client):
        """Test a complete workflow: create bank, store, search, reflect."""
        workflow_bank_id = "workflow_test_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Create bank
        client.create_bank(
            bank_id=workflow_bank_id,
            background="I am a software engineer who loves Python programming.",
        )

        # 2. Store memories
        store_response = client.retain_batch(
            bank_id=workflow_bank_id,
            items=[
                {"content": "I completed a project using FastAPI"},
                {"content": "I learned about async programming in Python"},
                {"content": "I enjoy working on open source projects"},
            ],
        )
        assert store_response.success is True

        # 3. Search for relevant memories
        search_results = client.recall(
            bank_id=workflow_bank_id,
            query="What programming technologies do I use?",
        )
        assert len(search_results.results) > 0

        # 4. Generate contextual answer
        reflect_response = client.reflect(
            bank_id=workflow_bank_id,
            query="What are my professional interests?",
        )
        assert reflect_response.text is not None
        assert len(reflect_response.text) > 0
