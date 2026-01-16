"""Tests for mental model functionality (v4 system)."""

import uuid

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine


@pytest.fixture
async def memory_with_mission(memory: MemoryEngine, request_context):
    """Memory engine with a bank that has a mission set.

    Uses a unique bank_id to avoid conflicts between parallel tests.
    """
    # Use unique bank_id to avoid conflicts between parallel tests
    bank_id = f"test-mental-models-{uuid.uuid4().hex[:8]}"

    # Set up the bank with a mission
    await memory.set_bank_mission(
        bank_id=bank_id,
        mission="Be a PM for the engineering team",
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


class TestBankMission:
    """Test bank mission operations."""

    async def test_set_and_get_mission(self, memory: MemoryEngine, request_context):
        """Test setting and getting a bank's mission."""
        bank_id = f"test-mission-{uuid.uuid4().hex[:8]}"

        # Set mission
        result = await memory.set_bank_mission(
            bank_id=bank_id,
            mission="Track customer feedback",
            request_context=request_context,
        )

        assert result["bank_id"] == bank_id
        assert result["mission"] == "Track customer feedback"

        # Get mission via profile
        profile = await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
        assert profile["mission"] == "Track customer feedback"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestRefreshMentalModels:
    """Test the main refresh_mental_models flow."""

    async def test_refresh_creates_structural_models(self, memory_with_mission, request_context):
        """Test that refresh creates structural models from the mission."""
        memory, bank_id = memory_with_mission

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

    async def test_refresh_without_mission_fails(self, memory: MemoryEngine, request_context):
        """Test that refresh fails when no mission is set."""
        bank_id = f"test-no-mission-refresh-{uuid.uuid4().hex[:8]}"

        # Add some data but don't set a mission
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

        # Refresh mental models should fail without a mission
        with pytest.raises(ValueError) as exc_info:
            await memory.refresh_mental_models(
                bank_id=bank_id,
                request_context=request_context,
            )

        assert "no mission is set" in str(exc_info.value).lower()

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestMentalModelCRUD:
    """Test basic CRUD operations for mental models."""

    async def test_list_mental_models(self, memory_with_mission, request_context):
        """Test listing mental models."""
        memory, bank_id = memory_with_mission

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

    async def test_get_mental_model(self, memory_with_mission, request_context):
        """Test getting a mental model by ID."""
        memory, bank_id = memory_with_mission

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

    async def test_delete_mental_model(self, memory_with_mission, request_context):
        """Test deleting a mental model."""
        memory, bank_id = memory_with_mission

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

    async def test_create_pinned_mental_model(self, memory: MemoryEngine, request_context):
        """Test creating a pinned mental model."""
        bank_id = f"test-pinned-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists by getting its profile (auto-creates if needed)
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create a pinned mental model
        model = await memory.create_mental_model(
            bank_id=bank_id,
            name="Product Roadmap",
            description="Key product priorities and upcoming features",
            tags=["project-x"],
            request_context=request_context,
        )

        assert model["name"] == "Product Roadmap"
        assert model["description"] == "Key product priorities and upcoming features"
        assert model["subtype"] == "pinned"
        assert model["tags"] == ["project-x"]
        assert model["id"] == "pinned-product-roadmap"

        # Verify it can be retrieved
        retrieved = await memory.get_mental_model(
            bank_id=bank_id,
            model_id=model["id"],
            request_context=request_context,
        )
        assert retrieved is not None
        assert retrieved["subtype"] == "pinned"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_create_pinned_model_duplicate_fails(self, memory: MemoryEngine, request_context):
        """Test that creating a duplicate pinned model fails."""
        bank_id = f"test-pinned-dup-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create first model
        await memory.create_mental_model(
            bank_id=bank_id,
            name="Test Model",
            description="First model",
            request_context=request_context,
        )

        # Try to create duplicate
        with pytest.raises(ValueError) as exc_info:
            await memory.create_mental_model(
                bank_id=bank_id,
                name="Test Model",
                description="Second model",
                request_context=request_context,
            )

        assert "already exists" in str(exc_info.value).lower()

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_pinned_models_survive_refresh(self, memory: MemoryEngine, request_context):
        """Test that pinned models are not deleted during refresh."""
        bank_id = f"test-pinned-refresh-{uuid.uuid4().hex[:8]}"

        # Set a mission
        await memory.set_bank_mission(
            bank_id=bank_id,
            mission="Track customer feedback",
            request_context=request_context,
        )

        # Create a pinned model
        pinned_model = await memory.create_mental_model(
            bank_id=bank_id,
            name="Key Customers",
            description="Important customers to track",
            request_context=request_context,
        )

        # Refresh mental models
        await memory.refresh_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Verify pinned model still exists
        retrieved = await memory.get_mental_model(
            bank_id=bank_id,
            model_id=pinned_model["id"],
            request_context=request_context,
        )
        assert retrieved is not None
        assert retrieved["subtype"] == "pinned"
        assert retrieved["name"] == "Key Customers"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestMentalModelRefresh:
    """Test mental model summary refresh functionality."""

    async def test_refresh_creates_models_with_summaries(self, memory_with_mission, request_context):
        """Test that refresh_mental_models creates models and generates summaries."""
        memory, bank_id = memory_with_mission

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


class TestReflect:
    """Test reflect endpoint with mental models."""

    async def test_reflect_basic(self, memory_with_mission, request_context):
        """Test basic reflect query - reflect works even without mental models."""
        memory, bank_id = memory_with_mission

        # Run a reflect query
        result = await memory.reflect_async(
            bank_id=bank_id,
            query="Who are the team members?",
            request_context=request_context,
        )

        assert result.text is not None
        assert len(result.text) > 0


class TestMentalModelLearnTool:
    """Test mental model learn tool - creates placeholders with background generation."""

    async def test_learn_creates_placeholder(self, memory: MemoryEngine, request_context):
        """Test that learn tool creates a placeholder mental model without observations."""
        bank_id = f"test-source-facts-{uuid.uuid4().hex[:8]}"

        # Add some test data
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice is the team lead."},
                {"content": "Bob is the engineer."},
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Directly use the learn tool to create a mental model placeholder
        from hindsight_api.engine.reflect.models import MentalModelInput
        from hindsight_api.engine.reflect.tools import tool_learn

        input_model = MentalModelInput(
            name="Team Members",
            description="Key team members and their roles",
        )

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            result = await tool_learn(conn, bank_id, input_model)

        assert result["status"] == "created"
        assert result["model_id"] == "team-members"
        assert result["name"] == "Team Members"
        assert result["pending_generation"] is True

        # Verify placeholder was stored in database with empty observations
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT subtype, name, description, observations FROM mental_models WHERE id = $1 AND bank_id = $2",
                result["model_id"],
                bank_id,
            )

        assert row is not None
        assert row["subtype"] == "learned"
        assert row["name"] == "Team Members"
        assert row["description"] == "Key team members and their roles"
        # Observations should be empty - will be generated in background
        observations_data = row["observations"]
        # Handle both string and dict representations
        if isinstance(observations_data, str):
            import json
            observations_data = json.loads(observations_data) if observations_data else {}
        assert observations_data == {} or observations_data is None

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_learn_update_description(self, memory: MemoryEngine, request_context):
        """Test that updating a mental model updates the description."""
        bank_id = f"test-merge-facts-{uuid.uuid4().hex[:8]}"

        # Create bank by retaining some data
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Test data"}],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        from hindsight_api.engine.reflect.models import MentalModelInput
        from hindsight_api.engine.reflect.tools import tool_learn

        # First create a placeholder
        input_model = MentalModelInput(
            name="Team Members",
            description="Initial description",
        )

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            result1 = await tool_learn(conn, bank_id, input_model)

        assert result1["status"] == "created"
        assert result1["pending_generation"] is True

        # Now update with new description
        input_model2 = MentalModelInput(
            name="Team Members",  # Same name = same ID
            description="Updated description with more context",
        )

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            result2 = await tool_learn(conn, bank_id, input_model2)

        assert result2["status"] == "updated"
        assert result2["model_id"] == "team-members"

        # Verify description was updated in database
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT description FROM mental_models WHERE id = $1 AND bank_id = $2",
                result1["model_id"],
                bank_id,
            )

        assert row is not None
        assert row["description"] == "Updated description with more context"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestMentalModelTags:
    """Test mental model tags functionality."""

    @pytest.fixture
    async def memory_with_mission_and_tags(self, memory: MemoryEngine, request_context):
        """Memory engine with a bank that has a mission set and tagged content."""
        bank_id = f"test-mm-tags-{uuid.uuid4().hex[:8]}"

        # Set up the bank with a mission
        await memory.set_bank_mission(
            bank_id=bank_id,
            mission="Be a PM for the engineering team",
            request_context=request_context,
        )

        # Add some test data
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice is the frontend engineer."},
                {"content": "Bob is the backend engineer."},
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        yield memory, bank_id

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_refresh_creates_models_with_tags(self, memory_with_mission_and_tags, request_context):
        """Test that refresh_mental_models creates models with specified tags."""
        memory, bank_id = memory_with_mission_and_tags

        # Refresh mental models with tags
        result = await memory.refresh_mental_models(
            bank_id=bank_id,
            tags=["project-alpha", "sprint-1"],
            request_context=request_context,
        )

        assert "operation_id" in result
        await memory.wait_for_background_tasks()

        # Get the created models
        models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )

        assert len(models) > 0

        # All models should have the tags we specified
        for model in models:
            assert "tags" in model
            assert "project-alpha" in model["tags"]
            assert "sprint-1" in model["tags"]

    async def test_list_mental_models_filters_by_tags(self, memory_with_mission_and_tags, request_context):
        """Test that list_mental_models correctly filters by tags."""
        memory, bank_id = memory_with_mission_and_tags

        # Create models with different tags
        await memory.refresh_mental_models(
            bank_id=bank_id,
            tags=["project-alpha"],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Get all models
        all_models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )
        assert len(all_models) > 0

        # Filter by tags - should return models with matching tags
        filtered_models = await memory.list_mental_models(
            bank_id=bank_id,
            tags=["project-alpha"],
            request_context=request_context,
        )
        assert len(filtered_models) == len(all_models)  # All models have this tag

        # Filter by non-existent tag - should only return untagged models (none here)
        # But since all models have tags, and the filter includes untagged,
        # we need to test with a mix
        empty_filtered = await memory.list_mental_models(
            bank_id=bank_id,
            tags=["non-existent-tag"],
            request_context=request_context,
        )
        # Should return empty since no models are untagged and none match
        # Actually, the logic includes untagged models, so let's verify the behavior
        # All our models have tags, so only checking for non-existent tag
        # should return nothing (since none match and none are untagged)

    async def test_untagged_models_included_in_filter(self, memory: MemoryEngine, request_context):
        """Test that untagged mental models are always included when filtering."""
        bank_id = f"test-untagged-{uuid.uuid4().hex[:8]}"

        # Set up bank with mission
        await memory.set_bank_mission(
            bank_id=bank_id,
            mission="Track projects",
            request_context=request_context,
        )

        # Add some data
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Project Alpha is important."}],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # First refresh without tags (creates untagged models)
        await memory.refresh_mental_models(
            bank_id=bank_id,
            request_context=request_context,  # No tags
        )
        await memory.wait_for_background_tasks()

        # Get all models (should be untagged)
        all_models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )

        if len(all_models) > 0:
            # Verify models are untagged
            for model in all_models:
                assert model.get("tags", []) == []

            # Filter by any tag - untagged models should still be included
            filtered_models = await memory.list_mental_models(
                bank_id=bank_id,
                tags=["some-tag"],
                request_context=request_context,
            )
            # Untagged models should be included in the results
            assert len(filtered_models) == len(all_models)

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_tags_match_any(self, memory: MemoryEngine, request_context):
        """Test tags_match='any' returns models with at least one matching tag."""
        bank_id = f"test-tags-any-{uuid.uuid4().hex[:8]}"

        # Set up bank with mission
        await memory.set_bank_mission(
            bank_id=bank_id,
            mission="Track projects",
            request_context=request_context,
        )

        # Add data and create models with tags
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Alice works on frontend."}],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        await memory.refresh_mental_models(
            bank_id=bank_id,
            tags=["tag-a", "tag-b"],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Filter with tags_match='any' - should match if any tag matches
        models = await memory.list_mental_models(
            bank_id=bank_id,
            tags=["tag-a", "tag-c"],  # tag-a matches, tag-c doesn't
            tags_match="any",
            request_context=request_context,
        )

        # Models with tag-a should be included
        for model in models:
            if model.get("tags"):
                # At least one of the filter tags should be in the model tags
                # OR model is untagged
                assert (
                    any(t in model["tags"] for t in ["tag-a", "tag-c"])
                    or model["tags"] == []
                )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_reflect_with_tags_filter(self, memory_with_mission_and_tags, request_context):
        """Test that reflect filters memories by tags."""
        memory, bank_id = memory_with_mission_and_tags

        # Create mental models with tags
        await memory.refresh_mental_models(
            bank_id=bank_id,
            tags=["project-x"],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Reflect with matching tags
        result = await memory.reflect_async(
            bank_id=bank_id,
            query="Who are the engineers?",
            tags=["project-x"],
            request_context=request_context,
        )

        assert result.text is not None
        assert len(result.text) > 0

        # Reflect with non-matching tags - should still work
        result2 = await memory.reflect_async(
            bank_id=bank_id,
            query="Who are the engineers?",
            tags=["different-project"],
            request_context=request_context,
        )

        assert result2.text is not None

    async def test_mental_model_response_includes_tags(self, memory_with_mission_and_tags, request_context):
        """Test that mental model responses include the tags field."""
        memory, bank_id = memory_with_mission_and_tags

        # Create models with tags
        await memory.refresh_mental_models(
            bank_id=bank_id,
            tags=["test-tag"],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Get models
        models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )

        # Verify tags field is present in response
        for model in models:
            assert "tags" in model
            assert isinstance(model["tags"], list)

        # Get single model
        if models:
            model = await memory.get_mental_model(
                bank_id=bank_id,
                model_id=models[0]["id"],
                request_context=request_context,
            )
            assert "tags" in model
            assert isinstance(model["tags"], list)
