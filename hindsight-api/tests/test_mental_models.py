"""Tests for mental model functionality (v4 system)."""

import uuid

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine


@pytest.fixture
async def memory_with_goal(memory: MemoryEngine, request_context):
    """Memory engine with a bank that has a goal set.

    Uses a unique bank_id to avoid conflicts between parallel tests.
    """
    # Use unique bank_id to avoid conflicts between parallel tests
    bank_id = f"test-mental-models-{uuid.uuid4().hex[:8]}"

    # Set up the bank with a goal
    await memory.set_bank_goal(
        bank_id=bank_id,
        goal="Be a PM for the engineering team",
        request_context=request_context,
    )

    # Add some test data
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[
            {"content": "The team has daily standups at 9am where everyone shares their progress."},
            {"content": "Alice is the frontend engineer and specializes in React."},
            {"content": "Bob is the backend engineer and owns the API services."},
            {"content": "Sprint retrospectives happen every two weeks to discuss improvements."},
            {"content": "John is the tech lead and makes final decisions on architecture."},
        ],
        request_context=request_context,
    )

    # Wait for any background tasks from retain to complete
    await memory.wait_for_background_tasks()

    yield memory, bank_id

    # Cleanup
    await memory.delete_bank(bank_id, request_context=request_context)


class TestBankGoal:
    """Test bank goal operations."""

    async def test_set_and_get_goal(self, memory: MemoryEngine, request_context):
        """Test setting and getting a bank's goal."""
        bank_id = f"test-goal-{uuid.uuid4().hex[:8]}"

        # Set goal
        result = await memory.set_bank_goal(
            bank_id=bank_id,
            goal="Track customer feedback",
            request_context=request_context,
        )

        assert result["bank_id"] == bank_id
        assert result["goal"] == "Track customer feedback"

        # Get goal
        goal = await memory.get_bank_goal(bank_id=bank_id, request_context=request_context)
        assert goal == "Track customer feedback"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestRefreshMentalModels:
    """Test the main refresh_mental_models flow."""

    async def test_refresh_creates_structural_models(self, memory_with_goal, request_context):
        """Test that refresh creates structural models from the goal."""
        memory, bank_id = memory_with_goal

        # Refresh mental models (async - returns operation_id)
        result = await memory.refresh_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )

        # Check that we got an operation ID back
        assert "operation_id" in result
        assert result["status"] == "queued"

        # Wait for background task to complete
        await memory.wait_for_background_tasks()

        # Get the created models
        models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )

        assert len(models) > 0

        # Check that structural models were created
        structural_models = [m for m in models if m["subtype"] == "structural"]
        assert len(structural_models) > 0

        # Check that models have the expected structure
        for model in models:
            assert "id" in model
            assert "name" in model
            assert "description" in model
            assert model["subtype"] in ["structural", "emergent"]
            assert model["type"] in ["entity", "concept", "event"]

    async def test_refresh_without_goal_fails(self, memory: MemoryEngine, request_context):
        """Test that refresh fails when no goal is set."""
        bank_id = f"test-no-goal-refresh-{uuid.uuid4().hex[:8]}"

        # Add some data but don't set a goal
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice is the frontend engineer."},
                {"content": "Bob is the backend engineer."},
            ],
            request_context=request_context,
        )

        # Wait for any background tasks from retain to complete
        await memory.wait_for_background_tasks()

        # Refresh mental models should fail without a goal
        with pytest.raises(ValueError) as exc_info:
            await memory.refresh_mental_models(
                bank_id=bank_id,
                request_context=request_context,
            )

        assert "no goal is set" in str(exc_info.value).lower()

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestMentalModelCRUD:
    """Test basic CRUD operations for mental models."""

    async def test_list_mental_models(self, memory_with_goal, request_context):
        """Test listing mental models."""
        memory, bank_id = memory_with_goal

        # Refresh to create models (async)
        await memory.refresh_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # List all models
        models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )

        assert len(models) > 0

        # Test filtering by subtype
        structural_models = await memory.list_mental_models(
            bank_id=bank_id,
            subtype="structural",
            request_context=request_context,
        )

        assert all(m["subtype"] == "structural" for m in structural_models)

    async def test_get_mental_model(self, memory_with_goal, request_context):
        """Test getting a mental model by ID."""
        memory, bank_id = memory_with_goal

        # Refresh to create models (async)
        await memory.refresh_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Get the created models
        models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )

        # Get one by ID
        model_id = models[0]["id"]
        model = await memory.get_mental_model(
            bank_id=bank_id,
            model_id=model_id,
            request_context=request_context,
        )

        assert model is not None
        assert model["id"] == model_id

        # Test non-existent
        not_found = await memory.get_mental_model(
            bank_id=bank_id,
            model_id="non-existent",
            request_context=request_context,
        )
        assert not_found is None

    async def test_delete_mental_model(self, memory_with_goal, request_context):
        """Test deleting a mental model."""
        memory, bank_id = memory_with_goal

        # Refresh to create models (async)
        await memory.refresh_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Get the created models
        models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )

        # Delete one
        model_id = models[0]["id"]
        deleted = await memory.delete_mental_model(
            bank_id=bank_id,
            model_id=model_id,
            request_context=request_context,
        )
        assert deleted is True

        # Verify it's gone
        model = await memory.get_mental_model(
            bank_id=bank_id,
            model_id=model_id,
            request_context=request_context,
        )
        assert model is None

        # Delete non-existent returns False
        deleted_again = await memory.delete_mental_model(
            bank_id=bank_id,
            model_id=model_id,
            request_context=request_context,
        )
        assert deleted_again is False


class TestMentalModelRefresh:
    """Test mental model summary refresh functionality."""

    async def test_refresh_creates_models_with_summaries(self, memory_with_goal, request_context):
        """Test that refresh_mental_models creates models and generates summaries."""
        memory, bank_id = memory_with_goal

        # Refresh mental models (async - creates models and generates summaries)
        result = await memory.refresh_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )

        assert "operation_id" in result
        assert result["status"] == "queued"

        # Wait for background task to complete (includes summary generation)
        await memory.wait_for_background_tasks()

        # Get the created models
        models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )

        assert len(models) > 0

        # After async refresh completes, models should have summaries generated
        for model in models:
            assert "id" in model
            assert "name" in model
            # Summaries should be generated now (unless no relevant facts found)
            # We don't strictly assert on summary presence since it depends on data

    async def test_refresh_nonexistent_mental_model(self, memory: MemoryEngine, request_context):
        """Test refreshing a non-existent mental model returns None."""
        bank_id = f"test-refresh-noexist-{uuid.uuid4().hex[:8]}"

        result = await memory.refresh_mental_model(
            bank_id=bank_id,
            model_id="does-not-exist",
            request_context=request_context,
        )

        assert result is None


class TestResearch:
    """Test research endpoint."""

    async def test_research_basic(self, memory_with_goal, request_context):
        """Test basic research query - research works even without mental models."""
        memory, bank_id = memory_with_goal

        # Run a research query directly (without full refresh to keep test fast)
        # Research should work even without mental models, using supplementary facts
        result = await memory.research(
            bank_id=bank_id,
            query="Who are the team members?",
            request_context=request_context,
        )

        assert "answer" in result
        assert "mental_models_used" in result
        assert "question_type" in result
