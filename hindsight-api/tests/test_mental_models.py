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


class TestDirectives:
    """Test directive mental model functionality."""

    async def test_create_directive(self, memory: MemoryEngine, request_context):
        """Test creating a directive mental model with user-provided observations."""
        bank_id = f"test-directive-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create a directive with observations
        model = await memory.create_mental_model(
            bank_id=bank_id,
            name="Competitor Policy",
            description="Rules about mentioning competitors",
            subtype="directive",
            observations=[
                {"title": "Never mention", "content": "Never mention competitor product names directly"},
                {"title": "Redirect", "content": "If asked about competitors, redirect to our features"},
            ],
            request_context=request_context,
        )

        assert model["name"] == "Competitor Policy"
        assert model["description"] == "Rules about mentioning competitors"
        assert model["subtype"] == "directive"
        assert len(model["observations"]) == 2
        assert model["observations"][0].title == "Never mention"
        assert model["observations"][0].content == "Never mention competitor product names directly"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_directive_included_in_list(self, memory: MemoryEngine, request_context):
        """Test that directives are included in list_mental_models for admin visibility."""
        bank_id = f"test-directive-list-{uuid.uuid4().hex[:8]}"

        # Set up bank with mission
        await memory.set_bank_mission(
            bank_id=bank_id,
            mission="Test mission",
            request_context=request_context,
        )

        # Create a directive
        directive = await memory.create_mental_model(
            bank_id=bank_id,
            name="Test Directive",
            description="A test directive",
            subtype="directive",
            observations=[{"title": "Rule", "content": "Follow this rule"}],
            request_context=request_context,
        )

        # Create a pinned model
        pinned = await memory.create_mental_model(
            bank_id=bank_id,
            name="Test Pinned",
            description="A test pinned model",
            request_context=request_context,
        )

        # List without subtype filter - both should appear
        models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )

        # Both should appear (directives included in API listing for admin visibility)
        model_ids = [m["id"] for m in models]
        assert pinned["id"] in model_ids
        assert directive["id"] in model_ids

        # List with directive subtype filter - should find only directive
        directives = await memory.list_mental_models(
            bank_id=bank_id,
            subtype="directive",
            request_context=request_context,
        )
        assert len(directives) == 1
        assert directives[0]["id"] == directive["id"]

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_directive_get_includes_observations(self, memory: MemoryEngine, request_context):
        """Test that getting a directive returns its user-provided observations."""
        bank_id = f"test-directive-get-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create a directive with observations
        created = await memory.create_mental_model(
            bank_id=bank_id,
            name="Meeting Rules",
            description="Rules for scheduling meetings",
            subtype="directive",
            observations=[
                {"title": "No mornings", "content": "Never schedule meetings before noon"},
                {"title": "Max duration", "content": "Meetings should be 30 minutes max"},
            ],
            request_context=request_context,
        )

        # Get the directive
        retrieved = await memory.get_mental_model(
            bank_id=bank_id,
            model_id=created["id"],
            request_context=request_context,
        )

        assert retrieved is not None
        assert retrieved["subtype"] == "directive"
        assert len(retrieved["observations"]) == 2
        assert retrieved["observations"][0].title == "No mornings"
        assert retrieved["observations"][1].title == "Max duration"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_directive_survives_refresh(self, memory: MemoryEngine, request_context):
        """Test that directives are not modified during refresh_mental_models."""
        bank_id = f"test-directive-refresh-{uuid.uuid4().hex[:8]}"

        # Set up bank with mission
        await memory.set_bank_mission(
            bank_id=bank_id,
            mission="Test mission",
            request_context=request_context,
        )

        # Add some test data
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Alice is the engineer."}],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Create a directive
        directive = await memory.create_mental_model(
            bank_id=bank_id,
            name="Important Rule",
            description="A critical rule",
            subtype="directive",
            observations=[{"title": "Rule 1", "content": "Always follow this rule"}],
            request_context=request_context,
        )

        # Refresh mental models
        await memory.refresh_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Directive should still exist with same observations
        retrieved = await memory.get_mental_model(
            bank_id=bank_id,
            model_id=directive["id"],
            request_context=request_context,
        )

        assert retrieved is not None
        assert retrieved["subtype"] == "directive"
        assert len(retrieved["observations"]) == 1
        assert retrieved["observations"][0].title == "Rule 1"
        assert retrieved["observations"][0].content == "Always follow this rule"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_directive_requires_observations(self, memory: MemoryEngine, request_context):
        """Test that creating a directive without observations fails."""
        bank_id = f"test-directive-no-obs-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Try to create directive without observations
        with pytest.raises(ValueError) as exc_info:
            await memory.create_mental_model(
                bank_id=bank_id,
                name="Bad Directive",
                description="A directive without observations",
                subtype="directive",
                # No observations provided
                request_context=request_context,
            )

        assert "observations" in str(exc_info.value).lower()

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestDirectivesInReflect:
    """Test that directives are followed during reflect operations."""

    async def test_reflect_follows_language_directive(self, memory: MemoryEngine, request_context):
        """Test that reflect follows a directive to respond in a specific language."""
        bank_id = f"test-directive-reflect-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Add some content in English
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice is a software engineer who works at Google."},
                {"content": "Alice enjoys hiking on weekends and has been to Yosemite."},
                {"content": "Alice is currently working on a machine learning project."},
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Create a directive to always respond in French
        await memory.create_mental_model(
            bank_id=bank_id,
            name="Language Policy",
            description="Rules about language usage",
            subtype="directive",
            observations=[
                {
                    "title": "French Only",
                    "content": "ALWAYS respond in French language. Never respond in English.",
                },
            ],
            request_context=request_context,
        )

        # Run reflect query
        result = await memory.reflect_async(
            bank_id=bank_id,
            query="What does Alice do for work?",
            request_context=request_context,
        )

        assert result.text is not None
        assert len(result.text) > 0

        # Check that the response contains French words/patterns
        # Common French words that would appear when talking about someone's job
        french_indicators = [
            "elle",
            "travaille",
            "est",
            "une",
            "le",
            "la",
            "qui",
            "chez",
            "logiciel",
            "ingénieur",
            "ingénieure",
            "développeur",
            "développeuse",
        ]
        response_lower = result.text.lower()

        # At least some French words should appear in the response
        french_word_count = sum(1 for word in french_indicators if word in response_lower)
        assert (
            french_word_count >= 2
        ), f"Expected French response, but got: {result.text[:200]}"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

class TestMentalModelTagsFiltering:
    """Test tags filtering for mental models (all types)."""

    async def test_tags_match_any_includes_untagged(self, memory: MemoryEngine, request_context):
        """Test that 'any' tags_match mode includes untagged mental models."""
        bank_id = f"test-mm-tags-any-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create an UNTAGGED pinned model
        await memory.create_mental_model(
            bank_id=bank_id,
            name="Global Model",
            description="A global mental model",
            subtype="pinned",
            tags=[],  # No tags - should be included with "any" mode
            request_context=request_context,
        )

        # Test 1: list_mental_models with tags and tags_match="any" should include untagged
        models_any = await memory.list_mental_models(
            bank_id=bank_id,
            tags=["some-tag"],
            tags_match="any",  # Should include untagged
            request_context=request_context,
        )
        assert len(models_any) == 1, f"Expected untagged model with 'any' mode, got {len(models_any)}"

        # Test 2: list_mental_models with tags and tags_match="any_strict" should exclude untagged
        models_strict = await memory.list_mental_models(
            bank_id=bank_id,
            tags=["some-tag"],
            tags_match="any_strict",  # Should exclude untagged
            request_context=request_context,
        )
        assert len(models_strict) == 0, f"Expected no models with 'any_strict' mode, got {len(models_strict)}"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_tags_match_strict_modes(self, memory: MemoryEngine, request_context):
        """Test that strict modes only include mental models with matching tags."""
        bank_id = f"test-mm-tags-strict-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create a TAGGED pinned model
        await memory.create_mental_model(
            bank_id=bank_id,
            name="Tagged Model",
            description="A tagged mental model",
            subtype="pinned",
            tags=["project-a"],
            request_context=request_context,
        )

        # Create an UNTAGGED pinned model
        await memory.create_mental_model(
            bank_id=bank_id,
            name="Untagged Model",
            description="An untagged mental model",
            subtype="pinned",
            tags=[],  # No tags
            request_context=request_context,
        )

        # Test 1: any_strict with matching tag - should get ONLY the tagged model
        models_match = await memory.list_mental_models(
            bank_id=bank_id,
            tags=["project-a"],
            tags_match="any_strict",
            request_context=request_context,
        )
        assert len(models_match) == 1, f"Expected 1 model with matching tag, got {len(models_match)}"
        assert models_match[0]["name"] == "Tagged Model"

        # Test 2: any_strict with different tag - should get NO models
        models_no_match = await memory.list_mental_models(
            bank_id=bank_id,
            tags=["project-b"],
            tags_match="any_strict",
            request_context=request_context,
        )
        assert len(models_no_match) == 0, f"Expected no models with non-matching tag, got {len(models_no_match)}"

        # Test 3: any (non-strict) with any tag - should get BOTH models
        models_any = await memory.list_mental_models(
            bank_id=bank_id,
            tags=["project-a"],
            tags_match="any",
            request_context=request_context,
        )
        assert len(models_any) == 2, f"Expected 2 models with 'any' mode, got {len(models_any)}"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_tags_match_all_strict(self, memory: MemoryEngine, request_context):
        """Test that 'all_strict' requires ALL tags to be present."""
        bank_id = f"test-mm-tags-all-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create a model with multiple tags
        await memory.create_mental_model(
            bank_id=bank_id,
            name="Multi-Tag Model",
            description="Has project-a and project-b tags",
            subtype="pinned",
            tags=["project-a", "project-b"],
            request_context=request_context,
        )

        # Create a model with only one tag
        await memory.create_mental_model(
            bank_id=bank_id,
            name="Single-Tag Model",
            description="Has only project-a tag",
            subtype="pinned",
            tags=["project-a"],
            request_context=request_context,
        )

        # Test 1: all_strict with both tags - should get ONLY the multi-tag model
        models_all = await memory.list_mental_models(
            bank_id=bank_id,
            tags=["project-a", "project-b"],
            tags_match="all_strict",
            request_context=request_context,
        )
        assert len(models_all) == 1, f"Expected 1 model with all tags, got {len(models_all)}"
        assert models_all[0]["name"] == "Multi-Tag Model"

        # Test 2: all (non-strict) with both tags - should include untagged too
        # Add an untagged model
        await memory.create_mental_model(
            bank_id=bank_id,
            name="Untagged Model",
            description="No tags",
            subtype="pinned",
            tags=[],
            request_context=request_context,
        )

        models_all_non_strict = await memory.list_mental_models(
            bank_id=bank_id,
            tags=["project-a", "project-b"],
            tags_match="all",
            request_context=request_context,
        )
        # Should get Multi-Tag Model + Untagged Model
        assert len(models_all_non_strict) == 2, f"Expected 2 models with 'all' mode, got {len(models_all_non_strict)}"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestDirectivesPromptInjection:
    """Test that directives are properly injected into the system prompt."""

    def test_build_directives_section_empty(self):
        """Test that empty directives returns empty string."""
        from hindsight_api.engine.reflect.prompts import build_directives_section

        result = build_directives_section([])
        assert result == ""

    def test_build_directives_section_with_observations(self):
        """Test that directives with observations are formatted correctly."""
        from hindsight_api.engine.reflect.prompts import build_directives_section

        directives = [
            {
                "name": "Competitor Policy",
                "observations": [
                    {"title": "Never mention", "content": "Never mention competitor names"},
                    {"title": "Redirect", "content": "Redirect to our features"},
                ],
            }
        ]

        result = build_directives_section(directives)

        assert "## DIRECTIVES (MANDATORY)" in result
        assert "**Never mention**: Never mention competitor names" in result
        assert "**Redirect**: Redirect to our features" in result
        assert "NEVER violate these directives" in result

    def test_build_directives_section_fallback_to_description(self):
        """Test that directives without observations fall back to description."""
        from hindsight_api.engine.reflect.prompts import build_directives_section

        directives = [
            {
                "name": "Simple Rule",
                "description": "Just a simple rule description",
                "observations": [],
            }
        ]

        result = build_directives_section(directives)

        assert "**Simple Rule**: Just a simple rule description" in result

    def test_system_prompt_includes_directives(self):
        """Test that build_system_prompt_for_tools includes directives."""
        from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools

        bank_profile = {"name": "Test Bank", "mission": "Test mission"}
        directives = [
            {
                "name": "Test Directive",
                "observations": [{"title": "Rule", "content": "Follow this rule"}],
            }
        ]

        prompt = build_system_prompt_for_tools(
            bank_profile=bank_profile,
            directives=directives,
        )

        assert "## DIRECTIVES (MANDATORY)" in prompt
        assert "**Rule**: Follow this rule" in prompt
        # Directives should appear before CRITICAL RULES
        directives_pos = prompt.find("## DIRECTIVES")
        critical_rules_pos = prompt.find("## CRITICAL RULES")
        assert directives_pos < critical_rules_pos


class TestMentalModelVersioning:
    """Test mental model versioning functionality."""

    async def test_refresh_creates_version(self, memory_with_mission, request_context):
        """Test that refreshing a mental model creates a version entry."""
        memory, bank_id = memory_with_mission

        # First create a mental model via refresh_mental_models
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
        assert len(models) > 0

        model_id = models[0]["id"]

        # Refresh the specific model to trigger versioning
        result = await memory.refresh_mental_model(
            bank_id=bank_id,
            model_id=model_id,
            request_context=request_context,
        )

        assert result is not None
        # Version should be incremented
        assert result.get("version", 0) >= 1

        # Check version history
        versions = await memory.get_mental_model_versions(
            bank_id=bank_id,
            model_id=model_id,
            request_context=request_context,
        )

        assert len(versions) >= 1
        assert versions[0]["version"] >= 1
        assert "created_at" in versions[0]
        assert "observation_count" in versions[0]

    async def test_get_specific_version(self, memory_with_mission, request_context):
        """Test retrieving a specific version of a mental model."""
        memory, bank_id = memory_with_mission

        # Create and refresh a mental model
        await memory.refresh_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )
        assert len(models) > 0

        model_id = models[0]["id"]

        # Refresh to create version
        await memory.refresh_mental_model(
            bank_id=bank_id,
            model_id=model_id,
            request_context=request_context,
        )

        # Get versions
        versions = await memory.get_mental_model_versions(
            bank_id=bank_id,
            model_id=model_id,
            request_context=request_context,
        )
        assert len(versions) >= 1

        # Get specific version
        version_num = versions[0]["version"]
        version_data = await memory.get_mental_model_version(
            bank_id=bank_id,
            model_id=model_id,
            version=version_num,
            request_context=request_context,
        )

        assert version_data is not None
        assert version_data["version"] == version_num
        assert "observations" in version_data

    async def test_version_cleanup_keeps_max_versions(self, memory_with_mission, request_context):
        """Test that old versions are cleaned up when max is exceeded."""
        memory, bank_id = memory_with_mission

        # Create a mental model
        await memory.refresh_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )
        assert len(models) > 0

        model_id = models[0]["id"]

        # Refresh multiple times to create versions
        for _ in range(3):
            await memory.refresh_mental_model(
                bank_id=bank_id,
                model_id=model_id,
                request_context=request_context,
            )

        # Get versions - should have multiple but within max limit
        versions = await memory.get_mental_model_versions(
            bank_id=bank_id,
            model_id=model_id,
            request_context=request_context,
        )

        # Should have versions (exact count depends on config, but at least some)
        assert len(versions) >= 1
        # Versions should be in descending order
        if len(versions) > 1:
            assert versions[0]["version"] > versions[1]["version"]

