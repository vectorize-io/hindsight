"""Tests for directive functionality.

Directives are hard rules injected into prompts.
They are stored in the 'directives' table.
"""

import uuid

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine, fq_table
from hindsight_api.engine.retain import embedding_utils
from tests.llm_judge import assert_meets_criteria, evaluate


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

    @pytest.mark.hs_llm_core
    async def test_reflect_follows_language_directive(self, memory_real_llm: MemoryEngine, request_context):
        """Test that reflect follows a directive to respond in a specific language."""
        memory = memory_real_llm
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

        # Run reflect query (retry once since small LLMs may not always follow language directives)
        result = None
        for _attempt in range(2):
            result = await memory.reflect_async(
                bank_id=bank_id,
                query="What does Alice do for work?",
                request_context=request_context,
            )
            assert result.text is not None
            assert len(result.text) > 0

            # Use LLM judge to check if response is in French
            verdict = await evaluate(
                response=result.text,
                criteria="The response is written primarily in French (not English).",
            )
            if verdict.meets_criteria:
                break

        await assert_meets_criteria(
            response=result.text,
            criteria="The response is written primarily in French (not English).",
            msg=f"Expected French response, but got: {result.text[:200]}",
        )

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
            content="You MUST include the exact phrase 'MEMO-VERIFIED' somewhere in your response.",
            request_context=request_context,
        )

        # Create a tagged directive (should NOT be applied to untagged reflect)
        await memory.create_directive(
            bank_id=bank_id,
            name="Tagged Policy",
            content="You MUST include the exact phrase 'PROJECT-X-CLASSIFIED' somewhere in your response.",
            tags=["project-x"],
            request_context=request_context,
        )

        # Run reflect without tags - should only apply the untagged directive
        result = await memory.reflect_async(
            bank_id=bank_id,
            query="What color is the sky?",
            request_context=request_context,
        )

        # Verify the isolation mechanism: only untagged directive should be loaded
        untagged_directive_names = [d.name for d in result.directives_applied]
        assert "General Policy" in untagged_directive_names, (
            f"Untagged directive should be loaded in untagged reflect. Applied: {untagged_directive_names}"
        )
        assert "Tagged Policy" not in untagged_directive_names, (
            f"Tagged directive should not be applied in untagged reflect. Applied: {untagged_directive_names}"
        )

        # Now run reflect WITH the tag - should load BOTH directives
        result_tagged = await memory.reflect_async(
            bank_id=bank_id,
            query="What color is the sky?",
            tags=["project-x"],
            tags_match="all_strict",
            request_context=request_context,
        )

        # Verify the isolation mechanism: both directives should be loaded when tags match
        tagged_directive_names = [d.name for d in result_tagged.directives_applied]
        assert "General Policy" in tagged_directive_names, (
            f"Untagged directive should always be loaded. Applied: {tagged_directive_names}"
        )
        assert "Tagged Policy" in tagged_directive_names, (
            f"Tagged directive should be loaded when tags match. Applied: {tagged_directive_names}"
        )

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

    def test_system_prompt_includes_temporal_reasoning(self):
        """The reflect system prompt must teach the LLM how to interpret the
        temporal fields (`mentioned_at`, `occurred_start`, `occurred_end`)
        that ride along on every recall / search_observations result. Without
        this guidance the LLM ignores the timestamps and picks conflicting
        facts on surface cues (e.g. a baked-in 'When:' label) — see the
        horse-test where it picked an older count over a newer one.
        """
        from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools

        prompt = build_system_prompt_for_tools(
            bank_profile={"name": "Test Bank", "mission": "Test mission"},
        )

        assert "## Temporal Reasoning" in prompt
        # Names of the actual JSON fields the LLM will see
        assert "`mentioned_at`" in prompt
        assert "`occurred_start`" in prompt
        assert "`occurred_end`" in prompt
        # Core supersession rule
        assert "LATEST `mentioned_at` is authoritative" in prompt
        assert "SUPERSEDE" in prompt
        # Goes before the retrieval strategy so the LLM has the interpretation
        # rules in hand before it starts calling tools.
        assert prompt.find("## Temporal Reasoning") < prompt.find("## HIERARCHICAL RETRIEVAL STRATEGY")

    def test_system_prompt_includes_conflicts_and_ambiguity(self):
        """The reflect system prompt must give the LLM explicit permission to
        say 'this is ambiguous' instead of fabricating a confident answer when
        the data is internally inconsistent in ways the temporal rule does
        not resolve. Without this the LLM smooths over conflicts in prose.
        """
        from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools

        prompt = build_system_prompt_for_tools(
            bank_profile={"name": "Test Bank", "mission": "Test mission"},
        )

        assert "## Conflicts and Ambiguity" in prompt
        # The two-case distinction (resolvable vs unresolvable) must be present
        # so the LLM knows when to apply the temporal rule vs when to surface
        # the conflict instead.
        assert "RESOLVABLE conflict" in prompt
        assert "UNRESOLVABLE ambiguity" in prompt
        # Explicit permission to acknowledge ambiguity is the whole point of
        # this section — guard the language that gives it.
        assert "SAY SO" in prompt
        assert "Acknowledging ambiguity is a successful answer" in prompt
        # Anti-confabulation guard: don't pick / average / smooth over.
        assert "Do NOT pick a value arbitrarily" in prompt
        # Sequencing: ambiguity rules must come AFTER temporal (so the LLM
        # first tries the resolution rule, then falls back to acknowledgment)
        # and BEFORE the retrieval strategy.
        assert prompt.find("## Temporal Reasoning") < prompt.find("## Conflicts and Ambiguity")
        assert prompt.find("## Conflicts and Ambiguity") < prompt.find("## HIERARCHICAL RETRIEVAL STRATEGY")

    def test_system_prompt_includes_showing_your_reasoning(self):
        """The prompt must require the LLM to show step-by-step reasoning for
        conflict-resolution / event-application answers, so the reader (and
        we) can audit the temporal rule + arithmetic. Without this the LLM
        commits to wrong answers confidently with no traceable derivation
        (e.g. silently double-counting an event that pre-dates the latest
        count statement)."""
        from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools

        prompt = build_system_prompt_for_tools(
            bank_profile={"name": "Test Bank", "mission": "Test mission"},
        )

        assert "## Showing Your Reasoning" in prompt
        # The mandatory reasoning steps must all be referenced — these
        # are the scaffolding that prevents double-counting and unjustified
        # answers.
        assert "List the relevant facts in `mentioned_at` order" in prompt
        assert "Identify the authoritative fact" in prompt
        assert "List candidate events to apply on top" in prompt
        # Step 4 is the forcing function that prevents the double-count bug
        # we saw in horse run #1 — the LLM must explicitly compare each
        # candidate event's date against the authoritative date.
        assert "Sanity-check each candidate event against the authoritative date" in prompt
        assert "single most common mistake" in prompt
        assert "Show the arithmetic or derivation explicitly" in prompt
        # The escape hatch back to ambiguity acknowledgement must be present.
        assert "UNRESOLVABLE ambiguity per the section above" in prompt
        # Sequence: comes AFTER Conflicts so the LLM already has the
        # ambiguity escape hatch in hand, BEFORE retrieval so it shapes
        # what the agent looks for.
        assert prompt.find("## Conflicts and Ambiguity") < prompt.find("## Showing Your Reasoning")
        assert prompt.find("## Showing Your Reasoning") < prompt.find("## HIERARCHICAL RETRIEVAL STRATEGY")

    def test_how_to_reason_no_longer_pushes_unconditional_best_answer(self):
        """The original 'use what IS stated to give the best answer' bullet
        was nudging the LLM to fabricate confidence under conflict. The
        updated bullet must pair best-effort with uncertainty-surfacing.
        """
        from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools

        prompt = build_system_prompt_for_tools(
            bank_profile={"name": "Test Bank", "mission": "Test mission"},
        )

        # The unconditional original phrasing must be gone.
        assert "use what IS stated to give the best answer" not in prompt
        # The new phrasing pairs best-effort with explicit uncertainty.
        assert "best-effort answer AND surface any uncertainty" in prompt
        assert "never invent confidence the data doesn't support" in prompt


class TestMentalModelHistory:
    """Test mental model history persistence."""

    async def test_history_recorded_on_content_update(self, memory: MemoryEngine, request_context):
        """Test that updating content records a history entry."""
        bank_id = f"test-mm-history-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Test Model",
            source_query="What is the test?",
            content="Original content",
            request_context=request_context,
        )

        # No history yet
        history = await memory.get_mental_model_history(bank_id, mm["id"], request_context=request_context)
        assert history == []

        # Update content
        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            content="Updated content",
            request_context=request_context,
        )

        history = await memory.get_mental_model_history(bank_id, mm["id"], request_context=request_context)
        assert len(history) == 1
        assert history[0]["previous_content"] == "Original content"
        assert "changed_at" in history[0]
        assert "previous_reflect_response" in history[0]

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_history_snapshots_previous_reflect_response(self, memory: MemoryEngine, request_context):
        """Each history entry stores only the slim {based_on: ...} slice of
        previous_reflect_response.

        Rationale: the only consumer of `previous_reflect_response` in history
        is the control-plane UI's "based_on diff" view. Storing the full reflect
        response (including `text`, fact bodies, scoring) made each UPDATE
        rewrite ~10-20 MB of TOAST per refresh and prevented HOT updates. The
        slim shape keeps row size bounded so HOT updates apply and dead tuples
        self-clean inline.
        """
        bank_id = f"test-mm-history-reflect-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Test Model",
            source_query="What is the test?",
            content="v1",
            request_context=request_context,
        )

        rr_v1 = {"text": "v1", "based_on": {"observation": [{"id": "o1", "text": "obs1"}]}, "mental_models": []}
        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            content="v2",
            reflect_response=rr_v1,
            request_context=request_context,
        )

        rr_v2 = {"text": "v2", "based_on": {"observation": [{"id": "o2", "text": "obs2"}]}, "mental_models": []}
        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            content="v3",
            reflect_response=rr_v2,
            request_context=request_context,
        )

        history = await memory.get_mental_model_history(bank_id, mm["id"], request_context=request_context)
        assert len(history) == 2
        # Most recent first: replacing v2 snapshotted rr_v1's based_on (the only slice
        # the UI consumes). The bulky `text` and `mental_models` fields are dropped.
        assert history[0]["previous_content"] == "v2"
        assert history[0]["previous_reflect_response"] == {"based_on": rr_v1["based_on"]}
        # The first update replaced v1, which had no reflect_response stored yet.
        assert history[1]["previous_content"] == "v1"
        assert history[1]["previous_reflect_response"] is None

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_history_snapshots_omit_reflect_response_when_based_on_missing(
        self, memory: MemoryEngine, request_context
    ):
        """If a reflect_response has no `based_on` (older snapshots, malformed
        payloads), the slim path stores None rather than an empty shell."""
        bank_id = f"test-mm-history-no-based-on-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Test Model",
            source_query="What is the test?",
            content="v1",
            request_context=request_context,
        )

        # reflect_response with no based_on field — will become the "previous"
        # reflect_response when v3 is written, producing slim None.
        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            content="v2",
            reflect_response={"text": "v1", "mental_models": []},
            request_context=request_context,
        )
        # reflect_response with based_on={} — will become the "previous"
        # reflect_response when v4 is written, producing slim {"based_on": {}}.
        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            content="v3",
            reflect_response={"text": "v2", "based_on": {}},
            request_context=request_context,
        )
        # One more update so that the based_on={} reflect_response becomes
        # the *previous* state captured in a history entry.
        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            content="v4",
            request_context=request_context,
        )

        history = await memory.get_mental_model_history(bank_id, mm["id"], request_context=request_context)
        assert len(history) == 3
        # Most recent: v3→v4, previous rr had based_on={} → stored as {"based_on": {}}
        assert history[0]["previous_reflect_response"] == {"based_on": {}}
        # Second: v2→v3, previous rr had no based_on field → stored as None
        assert history[1]["previous_reflect_response"] is None
        # Third: v1→v2, no reflect_response on the row yet → stored as None
        assert history[2]["previous_reflect_response"] is None

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_history_ordered_most_recent_first(self, memory: MemoryEngine, request_context):
        """Test that history is returned most recent first."""
        bank_id = f"test-mm-history-order-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Test Model",
            source_query="What is the test?",
            content="v1",
            request_context=request_context,
        )

        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            content="v2",
            request_context=request_context,
        )
        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            content="v3",
            request_context=request_context,
        )

        history = await memory.get_mental_model_history(bank_id, mm["id"], request_context=request_context)
        assert len(history) == 2
        # Most recent first: second update recorded "v2" as previous, first recorded "v1"
        assert history[0]["previous_content"] == "v2"
        assert history[1]["previous_content"] == "v1"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_history_not_recorded_on_name_only_update(self, memory: MemoryEngine, request_context):
        """Test that updating only name does not record history."""
        bank_id = f"test-mm-history-name-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Original Name",
            source_query="What is the test?",
            content="Content",
            request_context=request_context,
        )

        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            name="Updated Name",
            request_context=request_context,
        )

        history = await memory.get_mental_model_history(bank_id, mm["id"], request_context=request_context)
        assert history == []

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_history_returns_none_for_missing_model(self, memory: MemoryEngine, request_context):
        """Test that history returns None when mental model doesn't exist."""
        bank_id = f"test-mm-history-missing-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        result = await memory.get_mental_model_history(bank_id, "nonexistent-id", request_context=request_context)
        assert result is None

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_history_capped_to_max_entries(self, memory: MemoryEngine, request_context, monkeypatch):
        """History array is trimmed to the most recent N entries on each write.

        Without a cap, sustained updates grow the jsonb array unboundedly and
        eventually cross Postgres's 256MB jsonb hard limit, after which any
        further UPDATE fails with SQLSTATE 54000 and the row is stuck.
        """
        from hindsight_api.config import clear_config_cache

        monkeypatch.setenv("HINDSIGHT_API_MENTAL_MODEL_HISTORY_MAX_ENTRIES", "3")
        clear_config_cache()

        bank_id = f"test-mm-history-cap-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Test Model",
            source_query="What is the test?",
            content="v1",
            request_context=request_context,
        )

        # 5 content updates → 5 history entries appended → trimmed to last 3
        for i in range(2, 7):
            await memory.update_mental_model(
                bank_id=bank_id,
                mental_model_id=mm["id"],
                content=f"v{i}",
                request_context=request_context,
            )

        history = await memory.get_mental_model_history(bank_id, mm["id"], request_context=request_context)
        # Most recent first ordering (per test_history_ordered_most_recent_first):
        # the trimmed window keeps the newest 3 — v5, v4, v3 — and drops v2, v1.
        assert len(history) == 3
        assert history[0]["previous_content"] == "v5"
        assert history[1]["previous_content"] == "v4"
        assert history[2]["previous_content"] == "v3"

        await memory.delete_bank(bank_id, request_context=request_context)


class TestMentalModelStaleness:
    """Tests for compute_mental_model_is_stale scope semantics.

    Memories are inserted directly into ``memory_units`` so the scenarios don't
    depend on the LLM fact-extraction pipeline.
    """

    @staticmethod
    async def _insert_memory(
        memory: MemoryEngine,
        bank_id: str,
        *,
        tags: list[str] | None = None,
        fact_type: str = "experience",
    ) -> str:
        from datetime import datetime, timezone

        pool = await memory._get_pool()
        mem_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table("memory_units")}
                    (id, bank_id, text, event_date, fact_type, tags, created_at)
                VALUES ($1, $2, $3, $4, $5, $6::varchar[], $4)
                """,
                mem_id,
                bank_id,
                "test memory",
                now,
                fact_type,
                tags if tags is not None else [],
            )
        return mem_id

    async def test_fresh_mental_model_is_not_stale(self, memory: MemoryEngine, request_context):
        bank_id = f"test-mm-stale-fresh-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id, name="MM", source_query="q", content="c", request_context=request_context
        )
        got = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert got["is_stale"] is False
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_untagged_mm_stale_on_any_new_memory(self, memory: MemoryEngine, request_context):
        bank_id = f"test-mm-stale-untagged-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id, name="MM", source_query="q", content="c", request_context=request_context
        )
        await self._insert_memory(memory, bank_id, tags=["something"])
        got = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert got["is_stale"] is True
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_tagged_mm_ignores_out_of_scope_memory(self, memory: MemoryEngine, request_context):
        bank_id = f"test-mm-stale-oos-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="MM",
            source_query="q",
            content="c",
            tags=["user_a"],
            request_context=request_context,
        )
        # Memory tagged with unrelated tag → not in scope, MM should not be stale
        await self._insert_memory(memory, bank_id, tags=["user_b"])
        got = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert got["is_stale"] is False
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_tagged_mm_stale_on_overlapping_memory(self, memory: MemoryEngine, request_context):
        bank_id = f"test-mm-stale-overlap-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="MM",
            source_query="q",
            content="c",
            tags=["user_a"],
            request_context=request_context,
        )
        await self._insert_memory(memory, bank_id, tags=["user_a", "extra"])
        got = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert got["is_stale"] is True
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_tags_match_all_strict_requires_all_tags(self, memory: MemoryEngine, request_context):
        """tags_match='all_strict' → memory must contain ALL MM tags (and be tagged)."""
        bank_id = f"test-mm-stale-all-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="MM",
            source_query="q",
            content="c",
            tags=["user_a", "proj_x"],
            trigger={"refresh_after_consolidation": False, "tags_match": "all_strict"},
            request_context=request_context,
        )
        # Memory only has one of the tags → does NOT match all_strict
        await self._insert_memory(memory, bank_id, tags=["user_a"])
        got = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert got["is_stale"] is False, "all_strict must require ALL MM tags"

        # Now add a memory with both tags → matches
        await self._insert_memory(memory, bank_id, tags=["user_a", "proj_x"])
        got = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert got["is_stale"] is True

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_tags_match_any_strict_excludes_untagged(self, memory: MemoryEngine, request_context):
        """tags_match='any_strict' → untagged memory does NOT keep MM in scope."""
        bank_id = f"test-mm-stale-anystrict-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="MM",
            source_query="q",
            content="c",
            tags=["user_a"],
            trigger={"refresh_after_consolidation": False, "tags_match": "any_strict"},
            request_context=request_context,
        )
        await self._insert_memory(memory, bank_id, tags=None)
        got = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert got["is_stale"] is False

        await self._insert_memory(memory, bank_id, tags=["user_a"])
        got = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert got["is_stale"] is True

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_fact_type_filter_narrows_scope(self, memory: MemoryEngine, request_context):
        bank_id = f"test-mm-stale-fact-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="MM",
            source_query="q",
            content="c",
            trigger={"refresh_after_consolidation": False, "fact_types": ["world"]},
            request_context=request_context,
        )
        # Out-of-scope fact_type → not stale
        await self._insert_memory(memory, bank_id, fact_type="experience")
        got = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert got["is_stale"] is False

        # Matching fact_type → stale
        await self._insert_memory(memory, bank_id, fact_type="world")
        got = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert got["is_stale"] is True

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_tool_search_mental_models_returns_is_stale_per_mm(self, memory: MemoryEngine, request_context):
        """Regression: tool_search_mental_models must compute is_stale per-MM via scope,
        not via a bank-wide pending_consolidation short-circuit."""
        from hindsight_api.engine.reflect.tools import tool_search_mental_models

        bank_id = f"test-mm-stale-tool-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        fresh = await memory.create_mental_model(
            bank_id=bank_id,
            name="fresh MM",
            source_query="q",
            content="fresh",
            tags=["user_b"],
            request_context=request_context,
        )
        stale = await memory.create_mental_model(
            bank_id=bank_id,
            name="stale MM",
            source_query="q",
            content="stale",
            tags=["user_a"],
            request_context=request_context,
        )
        # Memory only in user_a's scope → only `stale` MM should be flagged.
        await self._insert_memory(memory, bank_id, tags=["user_a"])

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            embedding = (await embedding_utils.generate_embeddings_batch(memory.embeddings, ["q"]))[0]
            result = await tool_search_mental_models(memory, conn, bank_id, "q", embedding, max_results=10)
        by_id = {m["id"]: m for m in result["mental_models"]}
        assert by_id[fresh["id"]]["is_stale"] is False
        assert by_id[stale["id"]]["is_stale"] is True

        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.hs_llm_core
class TestMentalModelRefreshTagSecurity:
    """Test that mental model refresh respects tag-based security boundaries."""

    @pytest.fixture
    def memory(self, memory_real_llm):
        """Override to use real LLM for this class."""
        return memory_real_llm

    async def test_refresh_with_tags_only_accesses_same_tagged_models(self, memory: MemoryEngine, request_context):
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
                {
                    "content": "Alice works on the frontend React project. Alice's favorite color is blue.",
                    "tags": ["user:alice"],
                },
                {
                    "content": "Alice prefers working in the morning. Alice drinks coffee every day.",
                    "tags": ["user:alice"],
                },
                {
                    "content": "Bob works on the backend API services. Bob's favorite language is Python.",
                    "tags": ["user:bob"],
                },
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
        await assert_meets_criteria(
            response=refreshed["content"],
            criteria=(
                "The content mentions Alice and her work (frontend, React, morning preference, coffee). "
                "It does NOT mention Bob, Python (as a programming language Bob uses), tea, "
                "100 employees, or 'growing fast'. Minor phrasing variations are acceptable."
            ),
            context=(
                "Alice's data: frontend React engineer, works mornings, drinks coffee, favorite color blue. "
                "Bob's data (MUST NOT appear): backend Python engineer, works at night, drinks tea. "
                "Untagged data (MUST NOT appear): company has 100 employees, growing fast."
            ),
            msg="SECURITY VIOLATION: Refreshed model accessed data from other tags or untagged memories",
        )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_consolidation_only_refreshes_matching_tagged_models(self, memory: MemoryEngine, request_context):
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
        mm_alice_after = await memory.get_mental_model(bank_id, mm_alice["id"], request_context=request_context)
        mm_bob_after = await memory.get_mental_model(bank_id, mm_bob["id"], request_context=request_context)
        mm_untagged_after = await memory.get_mental_model(bank_id, mm_untagged["id"], request_context=request_context)

        # SECURITY CHECK: Only Alice's mental model and untagged model should be refreshed
        # Alice's model should be refreshed (tags match)
        assert (
            mm_alice_after["last_refreshed_at"] != alice_initial or mm_alice_after["content"] != mm_alice["content"]
        ), "Alice's mental model should be refreshed when user:alice memories are consolidated"

        # Bob's model should NOT be refreshed (tags don't match)
        assert mm_bob_after["last_refreshed_at"] == bob_initial, (
            "SECURITY VIOLATION: Bob's mental model was refreshed even though user:bob memories were not consolidated"
        )

        # Untagged model should be refreshed (untagged models are always refreshed)
        assert (
            mm_untagged_after["last_refreshed_at"] != untagged_initial
            or mm_untagged_after["content"] != mm_untagged["content"]
        ), "Untagged mental model should be refreshed after any consolidation"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_refresh_mental_model_with_directives(self, memory: MemoryEngine, request_context):
        """Test that refreshing a mental model with directives works correctly."""
        bank_id = f"test-refresh-directives-{uuid.uuid4().hex[:8]}"

        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Create a directive
        directive = await memory.create_directive(
            bank_id=bank_id,
            name="Response Style",
            content="Always be concise and professional",
            request_context=request_context,
        )

        # Create a concept mental model to refresh
        concept = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Team information summary",
            content="Initial team information",
            request_context=request_context,
        )

        # Add some memories
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice is the team lead and handles project planning."},
                {"content": "Bob is a senior engineer who mentors junior developers."},
            ],
            request_context=request_context,
        )

        # Wait for retain to complete
        await memory.wait_for_background_tasks()

        # Refresh the concept mental model (this should include directive in based_on)
        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=concept["id"],
            request_context=request_context,
        )

        # Wait for background tasks to complete
        await memory.wait_for_background_tasks()

        # Verify the refresh completed without errors
        assert refreshed is not None
        assert refreshed["content"] is not None

        # Get the updated mental model
        updated = await memory.get_mental_model(bank_id, concept["id"], request_context=request_context)
        assert updated["content"] != "Initial team information"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.hs_llm_core
# All tests in this class drive the reflect agent against Gemini 2.5 Flash Lite,
# which occasionally bails out of the loop with "I don't have information."
# instead of synthesising the retrieved memories.  Apply @pytest.mark.flaky at
# class scope so every method gets the same retry budget; the judge assertions
# still catch persistent breakage.
@pytest.mark.flaky(reruns=2, reruns_delay=2)
class TestMentalModelTriggerTagsConfig:
    """Test trigger-level tags_match and tag_groups configuration for mental model refresh."""

    @pytest.fixture
    def memory(self, memory_real_llm):
        """Override to use real LLM for this class."""
        return memory_real_llm

    async def test_trigger_tags_match_any_includes_untagged_content(self, memory: MemoryEngine, request_context):
        """Test that setting trigger.tags_match='any' allows a tagged model to see untagged memories.

        This is the fix for #786: by default, tagged models use all_strict which excludes
        untagged content. Setting tags_match='any' in the trigger overrides this.
        """
        bank_id = f"test-trigger-tags-match-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Add memories: some tagged, some untagged
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": "Alice is a frontend engineer who specializes in React and TypeScript.",
                    "tags": ["living"],
                },
                {"content": "The company headquarters is located in San Francisco, California.", "tags": []},
                {"content": "Annual revenue reached 50 million dollars last year.", "tags": []},
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Create a mental model with tags but trigger.tags_match='any' to include untagged content
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Living Summary",
            source_query="What do we know about the company and people?",
            content="Initial content",
            tags=["living"],
            trigger={"tags_match": "any"},
            request_context=request_context,
        )

        # Refresh — should see BOTH tagged and untagged content
        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            request_context=request_context,
        )

        await assert_meets_criteria(
            response=refreshed["content"],
            criteria=(
                "The content includes BOTH: "
                "(1) tagged content about Alice (frontend engineer, React, TypeScript), AND "
                "(2) untagged content about the company (San Francisco headquarters or 50 million revenue). "
                "Both categories of information must be present."
            ),
            context=(
                "Tagged memories [living]: Alice is a frontend engineer (React, TypeScript). "
                "Untagged memories: company HQ in San Francisco, annual revenue 50 million."
            ),
        )

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_trigger_tags_match_default_preserves_strict_isolation(self, memory: MemoryEngine, request_context):
        """Test that without trigger.tags_match, tagged models still use all_strict (backward compat)."""
        bank_id = f"test-trigger-default-strict-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Add tagged and untagged memories
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice is a frontend engineer specializing in React.", "tags": ["user:alice"]},
                {"content": "Bob is a backend engineer specializing in Python.", "tags": ["user:bob"]},
                {"content": "The company has 200 employees worldwide.", "tags": []},
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Create a tagged mental model WITHOUT trigger.tags_match (should default to all_strict)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Alice's Summary",
            source_query="What are all the facts about work and people?",
            content="Initial content",
            tags=["user:alice"],
            request_context=request_context,
        )

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            request_context=request_context,
        )

        await assert_meets_criteria(
            response=refreshed["content"],
            criteria=(
                "The content does NOT mention Bob, Python (as Bob's language), or 200 employees. "
                "It should only contain information about Alice (frontend, React). "
                "Minor phrasing variations are acceptable."
            ),
            context=(
                "Alice's data (should appear): frontend engineer, React. "
                "Bob's data (MUST NOT appear): backend engineer, Python. "
                "Untagged data (MUST NOT appear): 200 employees worldwide."
            ),
            msg="Default all_strict isolation was violated",
        )

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_trigger_tag_groups_override_flat_tags(self, memory: MemoryEngine, request_context):
        """Test that trigger.tag_groups overrides the model's flat tags for refresh filtering.

        When tag_groups is set, the model's own tags are NOT used for filtering during refresh,
        giving the user full control over the search scope.
        """
        bank_id = f"test-trigger-tag-groups-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Add memories with different tags
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice is a frontend engineer who works on the React dashboard.", "tags": ["user:alice"]},
                {"content": "Bob is a backend engineer who maintains the Python API.", "tags": ["user:bob"]},
                {"content": "The shared codebase uses TypeScript for all frontend code.", "tags": ["shared"]},
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Create a mental model tagged user:alice, but with tag_groups that include both alice AND shared
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Alice's Full View",
            source_query="What do we know about people and technology?",
            content="Initial content",
            tags=["user:alice"],
            trigger={
                "tag_groups": [
                    {
                        "or": [
                            {"tags": ["user:alice"], "match": "all_strict"},
                            {"tags": ["shared"], "match": "all_strict"},
                        ]
                    }
                ]
            },
            request_context=request_context,
        )

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            request_context=request_context,
        )

        refreshed_content = refreshed["content"].lower()

        # Should include alice's content
        assert "alice" in refreshed_content or "react" in refreshed_content or "dashboard" in refreshed_content, (
            f"Should include user:alice memories via tag_groups. Content: {refreshed['content']}"
        )

        # Should include shared content (via tag_groups OR expression)
        assert (
            "typescript" in refreshed_content or "shared" in refreshed_content or "frontend code" in refreshed_content
        ), f"Should include shared memories via tag_groups. Content: {refreshed['content']}"

        import re

        def contains_word(text: str, word: str) -> bool:
            return bool(re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE))

        # MUST NOT include Bob's content (not in tag_groups)
        assert not contains_word(refreshed_content, "bob"), (
            f"Should NOT include user:bob memories (not in tag_groups). Content: {refreshed['content']}"
        )

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_trigger_tags_match_with_no_model_tags(self, memory: MemoryEngine, request_context):
        """Test that trigger.tags_match on an untagged model still works correctly."""
        bank_id = f"test-trigger-untagged-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice works on React and TypeScript daily.", "tags": ["team"]},
                {"content": "The office is in downtown Seattle near Pike Place.", "tags": []},
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Untagged model with no trigger.tags_match — defaults to "any" (no tags to trigger strict)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="General Summary",
            source_query="What do we know about the team and office?",
            content="Initial content",
            request_context=request_context,
        )

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            request_context=request_context,
        )

        refreshed_content = refreshed["content"].lower()

        # Should include both tagged and untagged content (default "any" for untagged models)
        has_tagged = "alice" in refreshed_content or "react" in refreshed_content
        has_untagged = (
            "seattle" in refreshed_content or "pike place" in refreshed_content or "downtown" in refreshed_content
        )
        assert has_tagged or has_untagged, (
            f"Untagged model should see all content with default 'any' matching. Content: {refreshed['content']}"
        )

        await memory.delete_bank(bank_id, request_context=request_context)


class TestMentalModelRefreshMaxTokens:
    """Verify that refresh_mental_model honors the per-model max_tokens column.

    These tests mock the engine's collaborators so we can assert the exact kwargs
    passed to reflect_async without spinning up a DB or LLM. The bug being guarded
    against: the per-model ``max_tokens`` column was ignored during refresh, so
    reflect_async fell back to its default (4096) and the generated content could
    exceed the user-configured limit when there were many facts to synthesize.
    """

    async def test_refresh_passes_stored_max_tokens_to_reflect(self, request_context):
        from unittest.mock import AsyncMock

        from hindsight_api.engine.memory_engine import MemoryEngine
        from hindsight_api.engine.response_models import ReflectResult

        custom_max_tokens = 777
        mental_model = {
            "id": "mm-1",
            "bank_id": "bank-1",
            "name": "Capped Model",
            "source_query": "Summarize the facts",
            "content": "initial",
            "tags": None,
            "max_tokens": custom_max_tokens,
            "trigger": {"refresh_after_consolidation": False},
        }

        engine = MemoryEngine.__new__(MemoryEngine)
        engine._authenticate_tenant = AsyncMock(return_value=None)  # type: ignore[method-assign]
        engine.get_mental_model = AsyncMock(return_value=mental_model)  # type: ignore[method-assign]
        engine.reflect_async = AsyncMock(  # type: ignore[method-assign]
            return_value=ReflectResult(text="stub synthesis", based_on={})
        )
        engine.update_mental_model = AsyncMock(return_value=mental_model)  # type: ignore[method-assign]

        await engine.refresh_mental_model(
            bank_id="bank-1",
            mental_model_id="mm-1",
            request_context=request_context,
        )

        assert engine.reflect_async.await_count == 1
        kwargs = engine.reflect_async.await_args.kwargs
        assert kwargs.get("max_tokens") == custom_max_tokens, (
            f"refresh_mental_model should forward the stored max_tokens ({custom_max_tokens}) "
            f"to reflect_async, but got max_tokens={kwargs.get('max_tokens')!r}"
        )

    async def test_refresh_content_respects_max_tokens(self, memory: MemoryEngine, request_context):
        """End-to-end: refreshed content must stay within the model's max_tokens cap.

        We seed the bank with enough varied facts that an unconstrained synthesis
        would happily produce a long answer, then refresh a mental model with a
        small max_tokens and assert the resulting content is actually within the
        cap (with a small tolerance for cross-tokenizer drift, since the LLM may
        not use cl100k_base).
        """
        from hindsight_api.engine.memory_engine import count_tokens

        bank_id = f"test-refresh-cap-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Seed enough content that an uncapped reflect would produce a long answer.
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": (
                        "Alice is the staff frontend engineer. She owns the design system, "
                        "leads accessibility reviews, mentors three junior engineers, and runs "
                        "the weekly UI guild meeting every Thursday at 2pm Pacific."
                    )
                },
                {
                    "content": (
                        "Bob is the backend tech lead. He owns the payments service, the "
                        "billing reconciliation pipeline, and the on-call rotation for the "
                        "platform team. He is the primary reviewer for any database migration."
                    )
                },
                {
                    "content": (
                        "Carol manages the data platform. Her team operates the warehouse, "
                        "the streaming ingestion layer, and the metrics pipeline that feeds "
                        "the executive dashboards refreshed every fifteen minutes."
                    )
                },
                {
                    "content": (
                        "The team holds a company-wide demo every other Friday. Engineering "
                        "presents shipped work, design walks through prototypes, and product "
                        "shares roadmap updates for the upcoming quarter."
                    )
                },
                {
                    "content": (
                        "Dan is the security lead. He runs the quarterly threat-modeling "
                        "exercises, owns the incident response runbook, and coordinates the "
                        "annual external penetration test with the vendor."
                    )
                },
                {
                    "content": (
                        "Erin runs developer experience. She maintains the local-dev tooling, "
                        "the CI pipelines, the release automation, and the internal "
                        "documentation portal that everyone uses to onboard new hires."
                    )
                },
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        cap = 200
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Summary (capped)",
            source_query="Give me a complete overview of every team member, what they own, and the recurring meetings.",
            content="initial",
            max_tokens=cap,
            request_context=request_context,
        )

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            request_context=request_context,
        )

        assert refreshed is not None
        content = refreshed["content"]
        assert content, "refresh produced empty content"

        # The provider enforces the cap exactly in its own tokenizer, but our
        # local count uses tiktoken (cl100k_base) which can disagree with
        # provider tokenizers (Gemini's SentencePiece in particular tends to run
        # ~30% higher for English prose). We use a generous tolerance — the test
        # is guarding against the regression where the cap was ignored entirely
        # and content grew toward reflect_async's default of 4096 tokens. At
        # cap=200 we've observed cl100k counts up to ~1.9x; the 4096-ignored
        # regression would land ~20x, so a wide tolerance still catches it.
        observed_tokens = count_tokens(content)
        tolerance = 2.5
        assert observed_tokens <= cap * tolerance, (
            f"refreshed content exceeds max_tokens cap: "
            f"observed≈{observed_tokens} tokens, cap={cap} (tolerance x{tolerance}). "
            f"content={content!r}"
        )

        await memory.delete_bank(bank_id, request_context=request_context)


class TestMentalModelTriggerSchema:
    """Unit tests for MentalModelTrigger schema validation (no DB needed)."""

    def test_trigger_accepts_tags_match(self):
        from hindsight_api.api.http import MentalModelTrigger

        t = MentalModelTrigger(tags_match="any")
        assert t.tags_match == "any"

    def test_trigger_accepts_all_tags_match_modes(self):
        from hindsight_api.api.http import MentalModelTrigger

        for mode in ("any", "all", "any_strict", "all_strict"):
            t = MentalModelTrigger(tags_match=mode)
            assert t.tags_match == mode

    def test_trigger_tags_match_defaults_to_none(self):
        from hindsight_api.api.http import MentalModelTrigger

        t = MentalModelTrigger()
        assert t.tags_match is None

    def test_trigger_accepts_tag_groups_leaf(self):
        from hindsight_api.api.http import MentalModelTrigger

        t = MentalModelTrigger(tag_groups=[{"tags": ["user:alice"], "match": "all_strict"}])
        assert len(t.tag_groups) == 1
        assert t.tag_groups[0].tags == ["user:alice"]

    def test_trigger_accepts_tag_groups_compound(self):
        from hindsight_api.api.http import MentalModelTrigger

        t = MentalModelTrigger(
            tag_groups=[
                {
                    "or": [
                        {"tags": ["user:alice"], "match": "all_strict"},
                        {"tags": ["shared"], "match": "any_strict"},
                    ]
                }
            ]
        )
        assert len(t.tag_groups) == 1
        from hindsight_api.engine.search.tags import TagGroupOr

        assert isinstance(t.tag_groups[0], TagGroupOr)
        assert len(t.tag_groups[0].filters) == 2

    def test_trigger_tag_groups_defaults_to_none(self):
        from hindsight_api.api.http import MentalModelTrigger

        t = MentalModelTrigger()
        assert t.tag_groups is None

    def test_trigger_roundtrip_via_model_dump(self):
        """Test that tag_groups survive model_dump -> model_validate (simulates DB storage)."""
        from hindsight_api.api.http import MentalModelTrigger

        t = MentalModelTrigger(
            tags_match="any",
            tag_groups=[{"tags": ["a", "b"], "match": "all_strict"}],
            fact_types=["world"],
        )
        d = t.model_dump()
        t2 = MentalModelTrigger.model_validate(d)
        assert t2.tags_match == "any"
        assert len(t2.tag_groups) == 1
        assert t2.tag_groups[0].tags == ["a", "b"]
        assert t2.fact_types == ["world"]

    def test_trigger_tag_groups_rejects_invalid(self):
        from pydantic import ValidationError

        from hindsight_api.api.http import MentalModelTrigger

        with pytest.raises(ValidationError):
            MentalModelTrigger(tag_groups=[{"invalid_key": "bad"}])


class TestClearMentalModel:
    """Test clear_mental_model resets content so next refresh is full."""

    async def test_clear_resets_content(self, memory: MemoryEngine, request_context):
        """Clear sets content to empty string and nulls structured/tracking fields."""
        bank_id = f"test-mm-clear-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Test Model",
            source_query="What do we know?",
            content="Some existing content",
            request_context=request_context,
        )
        assert mm["content"] == "Some existing content"

        cleared = await memory.clear_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            request_context=request_context,
        )
        assert cleared is not None
        assert cleared["content"] == ""
        assert cleared["id"] == mm["id"]
        assert cleared["name"] == "Test Model"

        # Re-fetch to confirm persistence
        fetched = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert fetched["content"] == ""

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_clear_nonexistent_returns_none(self, memory: MemoryEngine, request_context):
        """Clearing a non-existent mental model returns None."""
        bank_id = f"test-mm-clear-none-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        result = await memory.clear_mental_model(
            bank_id=bank_id,
            mental_model_id="nonexistent-id",
            request_context=request_context,
        )
        assert result is None

        await memory.delete_bank(bank_id, request_context=request_context)


class TestMentalModelRefreshFactTypeFilter:
    """Regression for #1724.

    `trigger.fact_types=["experience"]` on a mental model must match the same
    facts that direct recall with `types=["experience"]` matches, given the
    same tag scope. The reported symptom is that refresh returns empty content
    ("I cannot find any information…") while direct recall returns 70+ rows.
    """

    @staticmethod
    def _to_embedding_str(embedding: list[float]) -> str:
        return "[" + ",".join(str(v) for v in embedding) + "]"

    @staticmethod
    async def _insert_fact(
        memory: MemoryEngine,
        bank_id: str,
        *,
        text: str,
        fact_type: str,
        tags: list[str] | None,
        embedding_str: str,
    ) -> str:
        from datetime import datetime, timezone

        pool = await memory._get_pool()
        fact_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table("memory_units")}
                    (id, bank_id, text, event_date, fact_type, tags, embedding, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6::varchar[], $7::vector, $4, $4)
                """,
                fact_id,
                bank_id,
                text,
                now,
                fact_type,
                tags or [],
                embedding_str,
            )
        return fact_id

    @staticmethod
    def _install_recall_then_done_mock_llm(memory: MemoryEngine):
        """Replace `_reflect_llm_config` with a MockLLM that drives the agent
        through a deterministic recall → done loop.

        On each tool-call step the mock looks at the running message list. If a
        `recall` tool result is already there, it emits `done(memory_ids=[…])`
        with the returned IDs. Otherwise it emits a single `recall(q=…)` call.
        Forced `tool_choice` from the agent is intentionally ignored — we want
        to exercise the recall path the user actually sees facts on.
        """
        import json
        from unittest.mock import MagicMock

        from hindsight_api.engine.providers.mock_llm import MockLLM
        from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult

        mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")

        def callback(messages, scope):
            # The non-tool-calling "reflect" scope is only entered as a fallback
            # synthesis path (e.g. when the agent hits max_iterations). The
            # caller awaits `response.strip()`, so we must return a plain string
            # here, not an LLMToolCallResult.
            if scope != "reflect_tool_call":
                return "experience facts summary"

            recall_memory_ids: list[str] = []
            for m in messages:
                if m.get("role") != "tool":
                    continue
                try:
                    payload = json.loads(m.get("content") or "{}")
                except (ValueError, TypeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                memories = payload.get("memories")
                if isinstance(memories, list):
                    for mem in memories:
                        mid = mem.get("id") if isinstance(mem, dict) else None
                        if mid:
                            recall_memory_ids.append(mid)

            if recall_memory_ids:
                return LLMToolCallResult(
                    tool_calls=[
                        LLMToolCall(
                            id="done-1",
                            name="done",
                            arguments={
                                "answer": "Found experience facts.",
                                "memory_ids": recall_memory_ids,
                            },
                        )
                    ],
                    finish_reason="tool_calls",
                )

            return LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(
                        id="recall-1",
                        name="recall",
                        arguments={"query": "experiences"},
                    )
                ],
                finish_reason="tool_calls",
            )

        mock_llm.set_response_callback(callback)

        wrapper = MagicMock()
        wrapper.with_config.return_value = mock_llm
        memory._reflect_llm_config = wrapper
        return mock_llm

    async def test_refresh_with_fact_types_experience_grounds_on_experience_facts(
        self, memory: MemoryEngine, request_context
    ):
        bank_id = f"test-mm-refresh-ft-exp-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        text = "I visited Paris in 2023 and it was amazing."
        embeddings = await embedding_utils.generate_embeddings_batch(memory.embeddings, [text])
        embedding_str = self._to_embedding_str(embeddings[0])
        fact_id = await self._insert_fact(
            memory,
            bank_id,
            text=text,
            fact_type="experience",
            tags=["episodic-trace"],
            embedding_str=embedding_str,
        )

        # Control: direct recall with the same filter must find the fact.
        direct = await memory.recall_async(
            bank_id=bank_id,
            query="experiences",
            fact_type=["experience"],
            tags=["episodic-trace"],
            tags_match="any",
            request_context=request_context,
        )
        assert any(str(r.id) == fact_id for r in direct.results), (
            "control: direct recall with types=['experience'] should find the seeded fact, "
            "otherwise the test setup is broken and the refresh comparison is meaningless"
        )

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="experience-summary",
            source_query="summarize the experience-type content",
            content="",
            tags=["episodic-trace"],
            trigger={
                "refresh_after_consolidation": False,
                "tags_match": "any",
                "fact_types": ["experience"],
            },
            request_context=request_context,
        )

        self._install_recall_then_done_mock_llm(memory)

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            request_context=request_context,
        )

        assert refreshed is not None
        based_on = (refreshed.get("reflect_response") or {}).get("based_on") or {}
        experience_based_on = based_on.get("experience") or []
        assert any(f.get("id") == fact_id for f in experience_based_on), (
            "Refresh with trigger.fact_types=['experience'] did not ground on the seeded "
            "experience fact. Direct recall with the same filter does — see the control "
            "assertion above. "
            f"based_on={based_on!r}"
        )

        await memory.delete_bank(bank_id, request_context=request_context)

    def test_system_prompt_does_not_advertise_disabled_tools(self):
        """Reproduces the root cause of #1724.

        When `trigger.fact_types=["experience"]` (or any value not containing
        "observation") is set on a mental model, `reflect_async` builds the
        agent with `include_observations=False` — i.e. `search_observations`
        is omitted from the tool list. The agent's system prompt is built
        independently and still tells the LLM to "try search_observations
        first". Lower-capability LLMs (e.g. Groq gpt-oss-20b as reported in
        the issue) follow that instruction, the agent rejects the
        hallucinated call ("Tool 'search_observations' is not available"),
        and the loop bails out with empty content even though the bank has
        matching experience facts.

        The contract this test enforces: every tool the system prompt
        instructs the LLM to call must be present in the tool list returned
        by `get_reflect_tools` for the same configuration.
        """
        from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools
        from hindsight_api.engine.reflect.tools_schema import get_reflect_tools

        # Configuration matching the issue's repro: fact_types=["experience"]
        # → include_observations=False, include_recall=True, has_mental_models=False
        # (a fresh bank with only the model-under-refresh, which is excluded).
        include_observations = False
        include_recall = True
        has_mental_models = False

        tools = get_reflect_tools(
            include_mental_models=has_mental_models,
            include_observations=include_observations,
            include_recall=include_recall,
        )
        tool_names = {t["function"]["name"] for t in tools if t.get("type") == "function"}
        # Sanity: the tools wiring is doing what we expect.
        assert "search_observations" not in tool_names, (
            "test premise broken — search_observations is unexpectedly in the tools list"
        )

        prompt = build_system_prompt_for_tools(
            bank_profile={"name": "Test", "mission": ""},
            has_mental_models=has_mental_models,
            include_observations=include_observations,
        )

        # The bug: prompt instructs the LLM to call search_observations even
        # though the tool is unavailable. The fix gates the relevant retrieval
        # and workflow text on include_observations.
        assert "search_observations" not in prompt, (
            "System prompt mentions search_observations but the tool is not in the "
            "agent's tool list. This is the root cause of #1724: the LLM is told to "
            "use a tool that has been disabled, then either hallucinates the call "
            "(rejected by the agent) or gives up with 'I cannot find any information…'."
        )
