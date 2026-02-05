"""Tests for directive functionality.

Directives are hard rules injected into prompts.
They are stored in the 'directives' table.
"""

import uuid

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine


@pytest.fixture
async def memory_with_bank(memory: MemoryEngine, request_context):
    """Memory engine with a bank that has some data.

    Uses a unique bank_id to avoid conflicts between parallel tests.
    """
    # Use unique bank_id to avoid conflicts between parallel tests
    bank_id = f"test-directives-{uuid.uuid4().hex[:8]}"

    # Ensure bank exists
    await memory.get_bank_profile(bank_id, request_context=request_context)

    # Add some test data
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[
            {"content": "The team has daily standups at 9am where everyone shares their progress."},
            {"content": "Alice is the frontend engineer and specializes in React."},
            {"content": "Bob is the backend engineer and owns the API services."},
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


class TestDirectives:
    """Test directive functionality."""

    async def test_create_directive(self, memory: MemoryEngine, request_context):
        """Test creating a directive."""
        bank_id = f"test-directive-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create a directive
        directive = await memory.create_directive(
            bank_id=bank_id,
            name="Competitor Policy",
            content="Never mention competitor product names directly. If asked about competitors, redirect to our features.",
            request_context=request_context,
        )

        assert directive["name"] == "Competitor Policy"
        assert "Never mention competitor" in directive["content"]
        assert directive["is_active"] is True
        assert directive["priority"] == 0

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_directive_crud(self, memory: MemoryEngine, request_context):
        """Test basic CRUD operations for directives."""
        bank_id = f"test-directive-crud-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create
        directive = await memory.create_directive(
            bank_id=bank_id,
            name="Test Directive",
            content="Follow this rule",
            request_context=request_context,
        )
        directive_id = directive["id"]

        # Read
        retrieved = await memory.get_directive(
            bank_id=bank_id,
            directive_id=directive_id,
            request_context=request_context,
        )
        assert retrieved is not None
        assert retrieved["name"] == "Test Directive"
        assert retrieved["content"] == "Follow this rule"

        # List
        directives = await memory.list_directives(
            bank_id=bank_id,
            request_context=request_context,
        )
        assert len(directives) == 1
        assert directives[0]["id"] == directive_id

        # Update
        updated = await memory.update_directive(
            bank_id=bank_id,
            directive_id=directive_id,
            content="Updated rule content",
            request_context=request_context,
        )
        assert updated["content"] == "Updated rule content"

        # Delete
        deleted = await memory.delete_directive(
            bank_id=bank_id,
            directive_id=directive_id,
            request_context=request_context,
        )
        assert deleted is True

        # Verify deletion
        retrieved_after = await memory.get_directive(
            bank_id=bank_id,
            directive_id=directive_id,
            request_context=request_context,
        )
        assert retrieved_after is None

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_directive_priority(self, memory: MemoryEngine, request_context):
        """Test that directive priority works correctly."""
        bank_id = f"test-directive-priority-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create directives with different priorities
        await memory.create_directive(
            bank_id=bank_id,
            name="Low Priority",
            content="Low priority rule",
            priority=1,
            request_context=request_context,
        )

        await memory.create_directive(
            bank_id=bank_id,
            name="High Priority",
            content="High priority rule",
            priority=10,
            request_context=request_context,
        )

        # List should order by priority (desc)
        directives = await memory.list_directives(
            bank_id=bank_id,
            request_context=request_context,
        )
        assert len(directives) == 2
        assert directives[0]["name"] == "High Priority"
        assert directives[1]["name"] == "Low Priority"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_directive_is_active(self, memory: MemoryEngine, request_context):
        """Test that inactive directives are filtered by default."""
        bank_id = f"test-directive-active-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create active and inactive directives
        await memory.create_directive(
            bank_id=bank_id,
            name="Active Rule",
            content="This is active",
            is_active=True,
            request_context=request_context,
        )

        await memory.create_directive(
            bank_id=bank_id,
            name="Inactive Rule",
            content="This is inactive",
            is_active=False,
            request_context=request_context,
        )

        # List active only (default)
        active_directives = await memory.list_directives(
            bank_id=bank_id,
            active_only=True,
            request_context=request_context,
        )
        assert len(active_directives) == 1
        assert active_directives[0]["name"] == "Active Rule"

        # List all
        all_directives = await memory.list_directives(
            bank_id=bank_id,
            active_only=False,
            request_context=request_context,
        )
        assert len(all_directives) == 2

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestDirectiveTags:
    """Test tags functionality for directives."""

    async def test_directive_with_tags(self, memory: MemoryEngine, request_context):
        """Test creating a directive with tags."""
        bank_id = f"test-directive-tags-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create a directive with tags
        directive = await memory.create_directive(
            bank_id=bank_id,
            name="Tagged Rule",
            content="Follow this rule",
            tags=["project-a", "team-x"],
            request_context=request_context,
        )

        assert directive["tags"] == ["project-a", "team-x"]

        # Retrieve and verify tags
        retrieved = await memory.get_directive(
            bank_id=bank_id,
            directive_id=directive["id"],
            request_context=request_context,
        )
        assert retrieved["tags"] == ["project-a", "team-x"]

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_list_directives_by_tags(self, memory: MemoryEngine, request_context):
        """Test listing directives filtered by tags."""
        bank_id = f"test-directive-tags-list-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create directives with different tags
        await memory.create_directive(
            bank_id=bank_id,
            name="Rule A",
            content="Rule for project A",
            tags=["project-a"],
            request_context=request_context,
        )

        await memory.create_directive(
            bank_id=bank_id,
            name="Rule B",
            content="Rule for project B",
            tags=["project-b"],
            request_context=request_context,
        )

        # List all
        all_directives = await memory.list_directives(
            bank_id=bank_id,
            request_context=request_context,
        )
        assert len(all_directives) == 2

        # Filter by project-a tag
        filtered = await memory.list_directives(
            bank_id=bank_id,
            tags=["project-a"],
            request_context=request_context,
        )
        assert len(filtered) == 1
        assert filtered[0]["name"] == "Rule A"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_list_all_directives_without_filter(self, memory: MemoryEngine, request_context):
        """Test that listing directives without tags returns ALL directives (both tagged and untagged)."""
        bank_id = f"test-directive-list-all-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create untagged directive
        await memory.create_directive(
            bank_id=bank_id,
            name="Untagged Directive",
            content="This has no tags",
            request_context=request_context,
        )

        # Create tagged directive
        await memory.create_directive(
            bank_id=bank_id,
            name="Tagged Directive",
            content="This has tags",
            tags=["project-x"],
            request_context=request_context,
        )

        # List ALL directives (no tag filter, isolation_mode defaults to False)
        all_directives = await memory.list_directives(
            bank_id=bank_id,
            request_context=request_context,
        )

        # Should return BOTH tagged and untagged directives
        assert len(all_directives) == 2
        directive_names = {d["name"] for d in all_directives}
        assert "Untagged Directive" in directive_names
        assert "Tagged Directive" in directive_names

        # Verify the tagged directive has its tags
        tagged = next(d for d in all_directives if d["name"] == "Tagged Directive")
        assert tagged["tags"] == ["project-x"]

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestReflect:
    """Test reflect endpoint."""

    async def test_reflect_basic(self, memory_with_bank, request_context):
        """Test basic reflect query works."""
        memory, bank_id = memory_with_bank

        # Run a reflect query
        result = await memory.reflect_async(
            bank_id=bank_id,
            query="Who are the team members?",
            request_context=request_context,
        )

        assert result.text is not None
        assert len(result.text) > 0


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
        await memory.create_directive(
            bank_id=bank_id,
            name="Language Policy",
            content="ALWAYS respond in French language. Never respond in English.",
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

    async def test_tagged_directive_not_applied_without_tags(self, memory: MemoryEngine, request_context):
        """Test that directives with tags are NOT applied to untagged reflect operations."""
        bank_id = f"test-directive-isolation-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Add some untagged content
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "The sky is blue."},
                {"content": "Water is wet."},
            ],
            request_context=request_context,
        )

        # Add some tagged content for the project-x context
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "The sky is blue according to project X standards.", "tags": ["project-x"]},
                {"content": "Project X color guidelines specify sky is blue.", "tags": ["project-x"]},
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Create an untagged directive (should be applied)
        await memory.create_directive(
            bank_id=bank_id,
            name="General Policy",
            content="Always be polite and start responses with 'Hello!'",
            request_context=request_context,
        )

        # Create a tagged directive (should NOT be applied to untagged reflect)
        await memory.create_directive(
            bank_id=bank_id,
            name="Tagged Policy",
            content="ALWAYS respond in ALL CAPS and end with 'PROJECT-X ONLY'",
            tags=["project-x"],
            request_context=request_context,
        )

        # Run reflect without tags - should only apply the untagged directive
        result = await memory.reflect_async(
            bank_id=bank_id,
            query="What color is the sky?",
            request_context=request_context,
        )

        response_lower = result.text.lower()

        # Should follow the untagged directive (polite greeting)
        assert "hello" in response_lower, f"Expected 'Hello' from untagged directive, but got: {result.text}"

        # Should NOT follow the tagged directive (all caps and PROJECT-X)
        # If it did follow, the entire response would be in caps
        all_caps = result.text.replace(" ", "").replace("!", "").replace(".", "").isupper()
        assert not all_caps, f"Tagged directive was incorrectly applied to untagged operation: {result.text}"
        assert "project-x only" not in response_lower, f"Tagged directive was incorrectly applied: {result.text}"

        # Now run reflect WITH the tag - should apply BOTH directives
        result_tagged = await memory.reflect_async(
            bank_id=bank_id,
            query="What color is the sky?",
            tags=["project-x"],
            tags_match="all_strict",
            request_context=request_context,
        )

        response_tagged_lower = result_tagged.text.lower()

        # With strict matching and tags, should apply the tagged directive
        assert "project-x only" in response_tagged_lower, f"Tagged directive should be applied with tags: {result_tagged.text}"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_reflect_based_on_structure(self, memory: MemoryEngine, request_context):
        """Test that reflect returns correct based_on structure with directives and memories separated."""
        bank_id = f"test-reflect-based-on-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Add some memories
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice works at Google as a software engineer."},
                {"content": "Bob is a product manager at Microsoft."},
                {"content": "The team meets every Monday at 9am."},
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Create a directive
        directive = await memory.create_directive(
            bank_id=bank_id,
            name="Professional Tone",
            content="Always maintain a professional and formal tone in responses.",
            request_context=request_context,
        )
        directive_id = directive["id"]

        # Run reflect which returns the core result
        result = await memory.reflect_async(
            bank_id=bank_id,
            query="Who works at Google?",
            request_context=request_context,
        )

        # Verify based_on structure exists
        assert result.based_on is not None

        # Verify directives key exists and contains our directive
        assert "directives" in result.based_on
        directives_list = result.based_on.get("directives", [])

        # Verify directives are dicts with id, name, content (not MemoryFact objects)
        assert len(directives_list) > 0, "Should have at least one directive"
        directive_found = False
        for d in directives_list:
            assert isinstance(d, dict), f"Directive should be dict, got {type(d)}"
            assert "id" in d, "Directive dict should have 'id'"
            assert "name" in d, "Directive dict should have 'name'"
            assert "content" in d, "Directive dict should have 'content'"
            # Check if this is our directive
            if d["id"] == directive_id:
                directive_found = True
                assert d["name"] == "Professional Tone"
                assert "professional" in d["content"].lower()

        assert directive_found, f"Our directive {directive_id} should be in based_on.directives"

        # Verify memories (world/experience) are separate from directives
        has_memories = "world" in result.based_on or "experience" in result.based_on
        assert has_memories, "Should have world or experience memories"

        # Verify that if mental-models key exists, it's separate from directives
        if "mental-models" in result.based_on:
            mental_models = result.based_on.get("mental-models", [])
            # Verify mental models are MemoryFact objects, not dicts like directives
            for mm in mental_models:
                assert hasattr(mm, "fact_type"), "Mental model should be MemoryFact with fact_type"
                assert mm.fact_type == "mental-models"
                assert hasattr(mm, "context")
                assert "mental model" in mm.context.lower()

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestDirectivesPromptInjection:
    """Test that directives are properly injected into the system prompt."""

    def test_build_directives_section_empty(self):
        """Test that empty directives returns empty string."""
        from hindsight_api.engine.reflect.prompts import build_directives_section

        result = build_directives_section([])
        assert result == ""

    def test_build_directives_section_with_content(self):
        """Test that directives with content are formatted correctly."""
        from hindsight_api.engine.reflect.prompts import build_directives_section

        directives = [
            {
                "name": "Competitor Policy",
                "content": "Never mention competitor names. Redirect to our features.",
            }
        ]

        result = build_directives_section(directives)

        assert "## DIRECTIVES (MANDATORY)" in result
        assert "Competitor Policy" in result
        assert "Never mention competitor names" in result
        assert "NEVER violate these directives" in result

    def test_system_prompt_includes_directives(self):
        """Test that build_system_prompt_for_tools includes directives."""
        from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools

        bank_profile = {"name": "Test Bank", "mission": "Test mission"}
        directives = [
            {
                "name": "Test Directive",
                "content": "Follow this rule",
            }
        ]

        prompt = build_system_prompt_for_tools(
            bank_profile=bank_profile,
            directives=directives,
        )

        assert "## DIRECTIVES (MANDATORY)" in prompt
        assert "Follow this rule" in prompt
        # Directives should appear before CRITICAL RULES
        directives_pos = prompt.find("## DIRECTIVES")
        critical_rules_pos = prompt.find("## CRITICAL RULES")
        assert directives_pos < critical_rules_pos


class TestMentalModelRefreshTagSecurity:
    """Test that mental model refresh respects tag-based security boundaries."""

    async def test_refresh_with_tags_only_accesses_same_tagged_models(
        self, memory: MemoryEngine, request_context
    ):
        """Test that refreshing a mental model with tags can only access other models with the same tags.

        This is a security test to ensure that mental models with tags (e.g., user:alice)
        cannot access mental models from other scopes (e.g., user:bob or no tags) during refresh.
        """
        bank_id = f"test-refresh-tags-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Add some facts with different tags
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice works on the frontend React project. Alice's favorite color is blue.", "tags": ["user:alice"]},
                {"content": "Alice prefers working in the morning. Alice drinks coffee every day.", "tags": ["user:alice"]},
                {"content": "Bob works on the backend API services. Bob's favorite language is Python.", "tags": ["user:bob"]},
                {"content": "Bob prefers working at night. Bob drinks tea every day.", "tags": ["user:bob"]},
                {"content": "The company has 100 employees and is growing fast.", "tags": []},  # No tags
            ],
            request_context=request_context,
        )

        # Wait for background processing
        await memory.wait_for_background_tasks()

        # Create mental model for user:alice with sensitive data
        mm_alice = await memory.create_mental_model(
            bank_id=bank_id,
            name="Alice's Work Profile",
            source_query="What does Alice work on?",
            content="Alice is a frontend engineer specializing in React",
            tags=["user:alice"],
            request_context=request_context,
        )

        # Create mental model for user:bob with sensitive data
        mm_bob = await memory.create_mental_model(
            bank_id=bank_id,
            name="Bob's Work Profile",
            source_query="What does Bob work on?",
            content="Bob is a backend engineer specializing in Python",
            tags=["user:bob"],
            request_context=request_context,
        )

        # Create mental model with no tags (should not be accessible from tagged models)
        mm_untagged = await memory.create_mental_model(
            bank_id=bank_id,
            name="Company Info",
            source_query="What is the company info?",
            content="The company has 100 employees",
            request_context=request_context,
        )

        # Create a mental model for user:alice that will be refreshed
        mm_alice_refresh = await memory.create_mental_model(
            bank_id=bank_id,
            name="Alice's Summary",
            source_query="What are all the facts about work and preferences?",  # Broad query that should match all facts
            content="Initial content",
            tags=["user:alice"],
            request_context=request_context,
        )

        # Refresh Alice's mental model
        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=mm_alice_refresh["id"],
            request_context=request_context,
        )

        # SECURITY CHECK: The refreshed content should ONLY include information from
        # memories/models tagged with user:alice, NOT from user:bob or untagged
        refreshed_content = refreshed["content"].lower()

        # Should include Alice's content (either from facts or mental models)
        assert "alice" in refreshed_content, \
            "Refreshed model should access memories/models with matching tags (user:alice)"

        # MUST NOT include Bob's content (security violation)
        assert "bob" not in refreshed_content and "python" not in refreshed_content and "tea" not in refreshed_content, \
            f"SECURITY VIOLATION: Refreshed model accessed memories/models with different tags (user:bob). Content: {refreshed_content}"

        # MUST NOT include untagged content (security violation)
        assert "100 employees" not in refreshed_content and "growing fast" not in refreshed_content, \
            f"SECURITY VIOLATION: Refreshed model accessed untagged memories/models. Content: {refreshed_content}"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_consolidation_only_refreshes_matching_tagged_models(
        self, memory: MemoryEngine, request_context
    ):
        """Test that consolidation only triggers refresh for mental models with matching tags.

        This is a security test to ensure that when tagged memories are consolidated,
        only mental models with overlapping tags get refreshed, not all mental models.
        """
        bank_id = f"test-consolidation-refresh-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create mental models with different tags, all with refresh_after_consolidation=true
        mm_alice = await memory.create_mental_model(
            bank_id=bank_id,
            name="Alice's Model",
            source_query="What about Alice?",
            content="Initial Alice content",
            tags=["user:alice"],
            trigger={"refresh_after_consolidation": True},
            request_context=request_context,
        )

        mm_bob = await memory.create_mental_model(
            bank_id=bank_id,
            name="Bob's Model",
            source_query="What about Bob?",
            content="Initial Bob content",
            tags=["user:bob"],
            trigger={"refresh_after_consolidation": True},
            request_context=request_context,
        )

        mm_untagged = await memory.create_mental_model(
            bank_id=bank_id,
            name="Untagged Model",
            source_query="What about general stuff?",
            content="Initial untagged content",
            trigger={"refresh_after_consolidation": True},
            request_context=request_context,
        )

        # Record initial last_refreshed_at timestamps
        alice_initial = mm_alice["last_refreshed_at"]
        bob_initial = mm_bob["last_refreshed_at"]
        untagged_initial = mm_untagged["last_refreshed_at"]

        # Add memories with user:alice tags
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice likes React", "tags": ["user:alice"]},
                {"content": "Alice drinks coffee", "tags": ["user:alice"]},
            ],
            request_context=request_context,
        )

        # Trigger consolidation manually (this should only refresh Alice's mental model)
        from hindsight_api.engine.consolidation.consolidator import run_consolidation_job

        result = await run_consolidation_job(
            memory_engine=memory,
            bank_id=bank_id,
            request_context=request_context,
        )

        # Wait for background refresh tasks to complete
        await memory.wait_for_background_tasks()

        # Check that mental models were refreshed appropriately
        mm_alice_after = await memory.get_mental_model(
            bank_id, mm_alice["id"], request_context=request_context
        )
        mm_bob_after = await memory.get_mental_model(
            bank_id, mm_bob["id"], request_context=request_context
        )
        mm_untagged_after = await memory.get_mental_model(
            bank_id, mm_untagged["id"], request_context=request_context
        )

        # SECURITY CHECK: Only Alice's mental model and untagged model should be refreshed
        # Alice's model should be refreshed (tags match)
        assert mm_alice_after["last_refreshed_at"] != alice_initial or mm_alice_after["content"] != mm_alice["content"], \
            "Alice's mental model should be refreshed when user:alice memories are consolidated"

        # Bob's model should NOT be refreshed (tags don't match)
        assert mm_bob_after["last_refreshed_at"] == bob_initial, \
            "SECURITY VIOLATION: Bob's mental model was refreshed even though user:bob memories were not consolidated"

        # Untagged model should be refreshed (untagged models are always refreshed)
        assert mm_untagged_after["last_refreshed_at"] != untagged_initial or mm_untagged_after["content"] != mm_untagged["content"], \
            "Untagged mental model should be refreshed after any consolidation"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)
