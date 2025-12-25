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

    def test_reflect_with_max_tokens(self, client, bank_id):
        """Test reflect with custom max_tokens parameter."""
        response = client.reflect(
            bank_id=bank_id,
            query="What do you think about Python?",
            max_tokens=500,
        )

        assert response is not None
        assert response.text is not None
        assert len(response.text) > 0

    def test_reflect_with_structured_output(self, client, bank_id):
        """Test reflect with structured output via response_schema.

        When response_schema is provided, the response returns structured_output
        field parsed according to the provided JSON schema. The text field is empty
        since only a single LLM call is made for structured output.
        """
        from typing import Optional
        from pydantic import BaseModel

        # Define schema using Pydantic model
        class RecommendationResponse(BaseModel):
            recommendation: str
            reasons: list[str]
            confidence: Optional[str] = None  # Optional for LLM flexibility

        response = client.reflect(
            bank_id=bank_id,
            query="What programming language should I learn for data science?",
            response_schema=RecommendationResponse.model_json_schema(),
            max_tokens=10000,
        )

        assert response is not None
        # Text is empty when using structured output (single LLM call)
        assert response.text == ""

        # Verify structured output is present and can be parsed into model
        assert response.structured_output is not None
        result = RecommendationResponse.model_validate(response.structured_output)
        assert result.recommendation
        assert isinstance(result.reasons, list)


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


class TestBankStats:
    """Tests for bank statistics endpoint."""

    @pytest.fixture(autouse=True)
    def setup_memories(self, client, bank_id):
        """Setup: Store some test memories."""
        client.retain_batch(
            bank_id=bank_id,
            items=[
                {"content": "Alice likes Python programming"},
                {"content": "Bob enjoys hiking in the mountains"},
            ],
            retain_async=False,
        )

    @pytest.mark.asyncio
    async def test_get_bank_stats(self, client, bank_id):
        """Test getting bank statistics."""
        from hindsight_client_api import ApiClient, Configuration
        from hindsight_client_api.api import BanksApi

        config = Configuration(host=HINDSIGHT_API_URL)
        api_client = ApiClient(config)
        api = BanksApi(api_client)
        stats = await api.get_agent_stats(bank_id=bank_id)

        assert stats is not None
        assert stats.bank_id == bank_id
        assert stats.total_nodes >= 0
        assert stats.total_links >= 0
        assert stats.total_documents >= 0
        assert isinstance(stats.nodes_by_fact_type, dict)
        assert isinstance(stats.links_by_link_type, dict)


class TestOperations:
    """Tests for operations endpoints."""

    @pytest.mark.asyncio
    async def test_list_operations(self, client, bank_id):
        """Test listing operations."""
        from hindsight_client_api import ApiClient, Configuration
        from hindsight_client_api.api import OperationsApi

        # First create an async operation using aretain_batch
        await client.aretain_batch(
            bank_id=bank_id,
            items=[{"content": "Test content for async operation"}],
            retain_async=True,
        )

        config = Configuration(host=HINDSIGHT_API_URL)
        api_client = ApiClient(config)
        api = OperationsApi(api_client)
        response = await api.list_operations(bank_id=bank_id)

        assert response is not None
        assert response.bank_id == bank_id
        assert isinstance(response.operations, list)


class TestDocuments:
    """Tests for document endpoints."""

    def test_delete_document(self, client, bank_id):
        """Test deleting a document."""
        import asyncio
        from hindsight_client_api import ApiClient, Configuration
        from hindsight_client_api.api import DocumentsApi

        # First create a document using sync retain
        doc_id = f"test-doc-{uuid.uuid4().hex[:8]}"
        client.retain(
            bank_id=bank_id,
            content="Test document content for deletion",
            document_id=doc_id,
        )

        # Use async API for delete (run in event loop)
        async def do_delete():
            config = Configuration(host=HINDSIGHT_API_URL)
            api_client = ApiClient(config)
            api = DocumentsApi(api_client)

            # Try to delete
            response = await api.delete_document(bank_id=bank_id, document_id=doc_id)
            return response

        response = asyncio.get_event_loop().run_until_complete(do_delete())

        assert response is not None
        assert response.success is True
        assert response.document_id == doc_id
        assert response.memory_units_deleted >= 0

    def test_get_document(self, client, bank_id):
        """Test getting a document."""
        import asyncio
        from hindsight_client_api import ApiClient, Configuration
        from hindsight_client_api.api import DocumentsApi

        # First create a document
        doc_id = f"test-doc-{uuid.uuid4().hex[:8]}"
        client.retain(
            bank_id=bank_id,
            content="Test document content for retrieval",
            document_id=doc_id,
        )

        async def do_get():
            config = Configuration(host=HINDSIGHT_API_URL)
            api_client = ApiClient(config)
            api = DocumentsApi(api_client)
            return await api.get_document(bank_id=bank_id, document_id=doc_id)

        document = asyncio.get_event_loop().run_until_complete(do_get())

        assert document is not None
        assert document.id == doc_id
        assert "Test document content" in document.original_text


class TestEntities:
    """Tests for entity endpoints."""

    @pytest.fixture(autouse=True)
    def setup_memories(self, client, bank_id):
        """Setup: Store memories that will generate entities."""
        client.retain_batch(
            bank_id=bank_id,
            items=[
                {"content": "Alice works at Google as a software engineer"},
                {"content": "Bob is friends with Alice and works at Microsoft"},
            ],
            retain_async=False,
        )

    def test_list_entities(self, client, bank_id):
        """Test listing entities."""
        import asyncio
        from hindsight_client_api import ApiClient, Configuration
        from hindsight_client_api.api import EntitiesApi

        async def do_list():
            config = Configuration(host=HINDSIGHT_API_URL)
            api_client = ApiClient(config)
            api = EntitiesApi(api_client)
            return await api.list_entities(bank_id=bank_id)

        response = asyncio.get_event_loop().run_until_complete(do_list())

        assert response is not None
        assert response.items is not None
        assert isinstance(response.items, list)

    def test_get_entity(self, client, bank_id):
        """Test getting a specific entity."""
        import asyncio
        from hindsight_client_api import ApiClient, Configuration
        from hindsight_client_api.api import EntitiesApi

        async def do_test():
            config = Configuration(host=HINDSIGHT_API_URL)
            api_client = ApiClient(config)
            api = EntitiesApi(api_client)

            # First list entities to get an ID
            list_response = await api.list_entities(bank_id=bank_id)

            if list_response.items and len(list_response.items) > 0:
                entity_id = list_response.items[0].id

                # Get the entity
                entity = await api.get_entity(bank_id=bank_id, entity_id=entity_id)
                return entity_id, entity
            return None, None

        entity_id, entity = asyncio.get_event_loop().run_until_complete(do_test())

        if entity_id:
            assert entity is not None
            assert entity.id == entity_id

    def test_regenerate_entity_observations(self, client, bank_id):
        """Test regenerating observations for an entity."""
        import asyncio
        from hindsight_client_api import ApiClient, Configuration
        from hindsight_client_api.api import EntitiesApi

        async def do_test():
            config = Configuration(host=HINDSIGHT_API_URL)
            api_client = ApiClient(config)
            api = EntitiesApi(api_client)

            # First list entities to get an ID
            list_response = await api.list_entities(bank_id=bank_id)

            if list_response.items and len(list_response.items) > 0:
                entity_id = list_response.items[0].id

                # Regenerate observations
                result = await api.regenerate_entity_observations(
                    bank_id=bank_id,
                    entity_id=entity_id,
                )
                return entity_id, result
            return None, None

        entity_id, result = asyncio.get_event_loop().run_until_complete(do_test())

        if entity_id:
            assert result is not None
            assert result.id == entity_id


class TestDeleteBank:
    """Tests for bank deletion."""

    def test_delete_bank(self, client):
        """Test deleting a bank."""
        import asyncio
        from hindsight_client_api import ApiClient, Configuration
        from hindsight_client_api.api import BanksApi

        # Create a unique bank for this test
        bank_id = f"test_bank_delete_{uuid.uuid4().hex[:12]}"

        # Create bank with some data
        client.create_bank(
            bank_id=bank_id,
            background="This bank will be deleted",
        )
        client.retain(
            bank_id=bank_id,
            content="Some memory to store",
        )

        async def do_delete():
            config = Configuration(host=HINDSIGHT_API_URL)
            api_client = ApiClient(config)
            api = BanksApi(api_client)
            return await api.delete_bank(bank_id=bank_id)

        response = asyncio.get_event_loop().run_until_complete(do_delete())

        assert response is not None
        assert response.success is True

        # Verify bank data is deleted - memories should be gone
        memories = client.list_memories(bank_id=bank_id)
        assert memories.total == 0
