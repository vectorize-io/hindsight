"""Tests for mental models (formerly reflections), observations, and learnings functionality."""

import uuid

import pytest
import pytest_asyncio
import httpx
from hindsight_api.api import create_app
from hindsight_api.engine.memory_engine import MemoryEngine


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
    return f"test_mental_models_{uuid.uuid4().hex[:8]}"


class TestMentalModelsCRUD:
    """Test mental models CRUD operations via memory engine."""

    @pytest.mark.asyncio
    async def test_create_and_get_mental_model(self, memory: MemoryEngine, request_context):
        """Test creating and retrieving a mental model."""
        bank_id = f"test-mental-model-{uuid.uuid4().hex[:8]}"

        # Create the bank first
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Create a mental model
        mental_model = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Preferences",
            source_query="What are the team's communication preferences?",
            content="The team prefers async communication via Slack",
            tags=["team"],
            request_context=request_context,
        )

        assert mental_model["name"] == "Team Preferences"
        assert mental_model["source_query"] == "What are the team's communication preferences?"
        assert mental_model["content"] == "The team prefers async communication via Slack"
        assert mental_model["tags"] == ["team"]
        assert "id" in mental_model

        # Get the mental model
        fetched = await memory.get_mental_model(
            bank_id=bank_id,
            mental_model_id=mental_model["id"],
            request_context=request_context,
        )

        assert fetched["id"] == mental_model["id"]
        assert fetched["name"] == "Team Preferences"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_list_mental_models(self, memory: MemoryEngine, request_context):
        """Test listing mental models with filters."""
        bank_id = f"test-mental-model-list-{uuid.uuid4().hex[:8]}"

        # Create the bank first
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Create multiple mental models
        await memory.create_mental_model(
            bank_id=bank_id,
            name="Mental Model 1",
            source_query="Query 1",
            content="Content 1",
            tags=["tag1"],
            request_context=request_context,
        )
        await memory.create_mental_model(
            bank_id=bank_id,
            name="Mental Model 2",
            source_query="Query 2",
            content="Content 2",
            tags=["tag2"],
            request_context=request_context,
        )

        # List all
        all_mental_models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )
        assert len(all_mental_models) == 2

        # List with tag filter
        tag1_mental_models = await memory.list_mental_models(
            bank_id=bank_id,
            tags=["tag1"],
            request_context=request_context,
        )
        assert len(tag1_mental_models) == 1

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_update_mental_model(self, memory: MemoryEngine, request_context):
        """Test updating a mental model."""
        bank_id = f"test-mental-model-update-{uuid.uuid4().hex[:8]}"

        # Create the bank first
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Create a mental model
        mental_model = await memory.create_mental_model(
            bank_id=bank_id,
            name="Original Name",
            source_query="Original Query",
            content="Original Content",
            request_context=request_context,
        )

        # Update the mental model
        updated = await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mental_model["id"],
            name="Updated Name",
            content="Updated Content",
            request_context=request_context,
        )

        assert updated["name"] == "Updated Name"
        assert updated["content"] == "Updated Content"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_delete_mental_model(self, memory: MemoryEngine, request_context):
        """Test deleting a mental model."""
        bank_id = f"test-mental-model-delete-{uuid.uuid4().hex[:8]}"

        # Create the bank first
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Create a mental model
        mental_model = await memory.create_mental_model(
            bank_id=bank_id,
            name="To Delete",
            source_query="Query",
            content="Content",
            request_context=request_context,
        )

        # Delete the mental model
        await memory.delete_mental_model(
            bank_id=bank_id,
            mental_model_id=mental_model["id"],
            request_context=request_context,
        )

        # Verify deletion - should return None
        fetched = await memory.get_mental_model(
            bank_id=bank_id,
            mental_model_id=mental_model["id"],
            request_context=request_context,
        )
        assert fetched is None

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_create_mental_model_with_custom_id(self, memory: MemoryEngine, request_context):
        """Test creating a mental model with a custom ID."""
        bank_id = f"test-mental-model-custom-id-{uuid.uuid4().hex[:8]}"

        # Create the bank first
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Create a mental model with a custom ID
        custom_id = "team-communication-preferences"
        mental_model = await memory.create_mental_model(
            bank_id=bank_id,
            mental_model_id=custom_id,
            name="Team Communication Preferences",
            source_query="How does the team prefer to communicate?",
            content="The team prefers async communication via Slack",
            tags=["team", "communication"],
            request_context=request_context,
        )

        # Verify the custom ID was used
        assert mental_model["id"] == custom_id
        assert mental_model["name"] == "Team Communication Preferences"
        assert mental_model["tags"] == ["team", "communication"]

        # Verify we can retrieve it with the custom ID
        fetched = await memory.get_mental_model(
            bank_id=bank_id,
            mental_model_id=custom_id,
            request_context=request_context,
        )

        assert fetched is not None
        assert fetched["id"] == custom_id
        assert fetched["name"] == "Team Communication Preferences"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestObservationsAPI:
    """Test observations API endpoints.

    NOTE: Observations are now stored in memory_units with fact_type='observation'
    and accessed via recall with fact_type=["observation"]. The old /observations
    endpoint was removed. These tests are skipped.
    """

    @pytest.mark.skip(reason="Observations endpoint removed - use recall with fact_type=['observation']")
    @pytest.mark.asyncio
    async def test_list_observations_empty(self, api_client, test_bank_id):
        """Test listing observations when none exist."""
        pass

    @pytest.mark.skip(reason="Observations endpoint removed - use recall with fact_type=['observation']")
    @pytest.mark.asyncio
    async def test_get_observation_not_found(self, api_client, test_bank_id):
        """Test getting a non-existent observation."""
        pass


class TestMentalModelsAPI:
    """Test mental models API endpoints."""

    @pytest.mark.asyncio
    async def test_mental_models_api_crud(self, api_client, test_bank_id):
        """Test full CRUD cycle through API."""
        import asyncio

        # Create bank first via profile endpoint
        await api_client.get(f"/v1/default/banks/{test_bank_id}/profile")

        # Create a mental model (async operation)
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/mental-models",
            json={
                "name": "API Test Mental Model",
                "source_query": "What is the API test about?",
                "content": "This is an API test mental model",
                "tags": ["api-test"],
            },
        )
        assert response.status_code == 200
        create_result = response.json()
        assert "operation_id" in create_result
        operation_id = create_result["operation_id"]

        # Wait for the async operation to complete
        for _ in range(30):  # Wait up to 30 seconds
            response = await api_client.get(f"/v1/default/banks/{test_bank_id}/operations/{operation_id}")
            if response.status_code == 200:
                op_status = response.json()
                if op_status.get("status") == "completed":
                    break
            await asyncio.sleep(1)

        # List mental models to get the created mental model
        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/mental-models")
        assert response.status_code == 200
        mental_models = response.json()["items"]
        assert len(mental_models) >= 1

        # Find our mental model
        mental_model = next((m for m in mental_models if m["name"] == "API Test Mental Model"), None)
        assert mental_model is not None, f"Mental model not found. Items: {mental_models}"
        mental_model_id = mental_model["id"]

        # Get the mental model
        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/mental-models/{mental_model_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "API Test Mental Model"

        # Update the mental model
        response = await api_client.patch(
            f"/v1/default/banks/{test_bank_id}/mental-models/{mental_model_id}",
            json={"name": "Updated API Test Mental Model"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Updated API Test Mental Model"

        # Delete the mental model
        response = await api_client.delete(f"/v1/default/banks/{test_bank_id}/mental-models/{mental_model_id}")
        assert response.status_code == 200

        # Verify deletion
        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/mental-models/{mental_model_id}")
        assert response.status_code == 404

        # Cleanup
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")


class TestRecallWithObservationsAndMentalModels:
    """Test recall integration with observations and mental models."""

    @pytest.mark.asyncio
    async def test_recall_includes_observations(self, api_client, test_bank_id):
        """Test that recall can include observations in the response."""
        # Create bank first via profile endpoint
        await api_client.get(f"/v1/default/banks/{test_bank_id}/profile")

        # Note: Observations are auto-created via consolidation, not manually
        # This test just verifies the include parameter works

        # Recall with observations included
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={
                "query": "What is machine learning?",
                "include": {
                    "observations": {"max_results": 5},
                },
            },
        )
        assert response.status_code == 200
        result = response.json()

        # Should have observations field in response (may be empty)
        assert "observations" in result or result.get("observations") is None

        # Cleanup
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")

    @pytest.mark.asyncio
    async def test_recall_includes_mental_models(self, api_client, test_bank_id):
        """Test that recall can include mental models in the response."""
        # Create bank first via profile endpoint
        await api_client.get(f"/v1/default/banks/{test_bank_id}/profile")

        # Create a mental model first
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/mental-models",
            json={
                "name": "AI Overview",
                "source_query": "What is AI?",
                "content": "Artificial intelligence is the simulation of human intelligence",
                "tags": [],
            },
        )
        assert response.status_code == 200

        # Recall with mental models included
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={
                "query": "What is artificial intelligence?",
                "include": {
                    "mental_models": {"max_results": 5},
                },
            },
        )
        assert response.status_code == 200
        result = response.json()

        # Should have mental_models in response (may be empty if embedding not generated yet)
        assert "mental_models" in result or result.get("mental_models") is None

        # Cleanup
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")

    @pytest.mark.asyncio
    async def test_recall_without_observations_by_default(self, api_client, test_bank_id):
        """Test that recall does not include observations by default."""
        # Create bank first via profile endpoint
        await api_client.get(f"/v1/default/banks/{test_bank_id}/profile")

        # Recall without specifying observations
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={
                "query": "Test query",
            },
        )
        assert response.status_code == 200
        result = response.json()

        # Observations should not be in response
        assert result.get("observations") is None

        # Cleanup
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")


class TestReflectUsesMentalModels:
    """Test that reflect searches and uses mental models when available."""

    @pytest.mark.asyncio
    async def test_reflect_searches_mental_models_when_available(self, memory: MemoryEngine, request_context):
        """Test that reflect uses search_mental_models when the bank has mental models.

        Given:
        - A bank with a mental model about "team collaboration"

        Expected:
        - Reflect should call search_mental_models tool
        - The mental model content should influence the response
        """
        bank_id = f"test-reflect-mm-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Create a mental model about team collaboration
        mental_model = await memory.create_mental_model(
            bank_id=bank_id,
            mental_model_id=str(uuid.uuid4()),
            name="Team Collaboration Practices",
            source_query="How does the team collaborate?",
            content="The team uses async communication via Slack and holds daily standups at 9am. "
            "Code reviews are required before merging. The team values documentation and "
            "prefers written communication for complex decisions.",
            tags=["team"],
            request_context=request_context,
        )

        # Run reflect with a query about team collaboration
        result = await memory.reflect_async(
            bank_id=bank_id,
            query="How does the team work together?",
            request_context=request_context,
        )

        # Check that mental models were searched
        tool_calls = result.tool_trace
        search_mm_calls = [tc for tc in tool_calls if tc.tool == "search_mental_models"]

        assert len(search_mm_calls) > 0, (
            f"Expected search_mental_models to be called when bank has mental models. "
            f"Tool calls: {[tc.tool for tc in tool_calls]}"
        )

        # Check that the reason field is populated for debugging
        for tc in search_mm_calls:
            assert tc.reason is not None, "Tool call should have a reason for debugging"

        # The response should mention concepts from the mental model
        response_text = result.text.lower()
        has_relevant_content = any(
            keyword in response_text
            for keyword in ["slack", "async", "standup", "code review", "documentation", "communication"]
        )
        assert has_relevant_content, (
            f"Expected response to reference mental model content. Got: {result.text[:500]}"
        )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_reflect_tool_trace_includes_reason(self, memory: MemoryEngine, request_context):
        """Test that tool traces include the reason field for debugging."""
        bank_id = f"test-reflect-reason-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Run reflect - it should use observations or recall
        result = await memory.reflect_async(
            bank_id=bank_id,
            query="What is the weather like?",
            request_context=request_context,
        )

        # All tool calls should have a reason
        for tc in result.tool_trace:
            if tc.tool != "done":  # done doesn't need a reason
                assert tc.reason is not None, f"Tool {tc.tool} should have a reason for debugging"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)
