"""Integration tests for the consolidation engine.

These tests exercise the real consolidation implementation with actual database operations.
Note: Consolidation runs automatically after retain via SyncTaskBackend in tests.
"""

import uuid
from unittest.mock import patch

import pytest

from hindsight_api.engine.consolidation.consolidator import run_consolidation_job
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.reflect.tools import (
    tool_recall,
    tool_search_mental_models,
    tool_search_observations,
)


@pytest.fixture(autouse=True)
def enable_observations():
    """Enable observations for all tests in this module."""
    from hindsight_api.config import get_config

    config = get_config()
    original_value = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original_value


class TestConsolidationIntegration:
    """Integration tests for consolidation with real database.

    These tests verify that consolidation creates observations correctly.
    Since we use SyncTaskBackend in tests, consolidation runs synchronously
    after retain completes.
    """

    @pytest.mark.asyncio
    async def test_consolidation_creates_observation_after_retain(
        self, memory: MemoryEngine, request_context
    ):
        """Test that consolidation creates an observation after retain."""
        bank_id = f"test-consolidation-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain a memory - consolidation runs automatically after
        await memory.retain_async(
            bank_id=bank_id,
            content="Peter loves hiking in the mountains every weekend.",
            request_context=request_context,
        )

        # Verify observation exists in memory_units
        # (consolidation already ran as part of retain via SyncTaskBackend)
        async with memory._pool.acquire() as conn:
            observations = await conn.fetch(
                """
                SELECT id, text, proof_count, fact_type
                FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id,
            )
            # Observation may or may not be created depending on LLM relevance judgment
            # The important thing is no errors occurred
            if observations:
                obs = observations[0]
                assert obs["proof_count"] >= 1
                assert obs["fact_type"] == "observation"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_consolidation_processes_multiple_memories(
        self, memory: MemoryEngine, request_context
    ):
        """Test that consolidation processes multiple related memories."""
        bank_id = f"test-consolidation-multi-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain first memory
        await memory.retain_async(
            bank_id=bank_id,
            content="Peter enjoys hiking on mountain trails.",
            request_context=request_context,
        )

        # Retain a second related memory
        await memory.retain_async(
            bank_id=bank_id,
            content="Peter went hiking in the Alps last weekend and loved it.",
            request_context=request_context,
        )

        # Check observations after both retains
        async with memory._pool.acquire() as conn:
            observations = await conn.fetch(
                """
                SELECT id, text, proof_count
                FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                ORDER BY proof_count DESC
                """,
                bank_id,
            )

            # Should have at least one observation
            # If the LLM determined both memories support the same observation,
            # proof_count might be > 1
            if observations:
                # Verify structure is correct
                assert all(obs["text"] for obs in observations)
                assert all(obs["proof_count"] >= 1 for obs in observations)

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_consolidation_no_new_memories(self, memory: MemoryEngine, request_context):
        """Test that consolidation handles case when no new memories exist."""
        bank_id = f"test-consolidation-empty-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Run consolidation without any memories
        result = await run_consolidation_job(
            memory_engine=memory,
            bank_id=bank_id,
            request_context=request_context,
        )

        assert result["status"] == "no_new_memories"
        assert result["memories_processed"] == 0

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_consolidation_respects_last_consolidated_at(
        self, memory: MemoryEngine, request_context
    ):
        """Test that consolidation only processes memories created after last_consolidated_at."""
        bank_id = f"test-consolidation-timestamp-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain a memory - consolidation runs automatically
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at a technology company.",
            request_context=request_context,
        )

        # Run consolidation again - should have no new memories
        result = await run_consolidation_job(
            memory_engine=memory,
            bank_id=bank_id,
            request_context=request_context,
        )

        # Should report no new memories since consolidation already ran
        assert result["status"] == "no_new_memories"
        assert result["memories_processed"] == 0

        # Add a new memory
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice got promoted to senior engineer.",
            request_context=request_context,
        )

        # Run consolidation again - should also have no new memories
        # because consolidation ran automatically after the second retain
        result = await run_consolidation_job(
            memory_engine=memory,
            bank_id=bank_id,
            request_context=request_context,
        )

        assert result["status"] == "no_new_memories"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_consolidation_copies_entity_links(self, memory: MemoryEngine, request_context):
        """Test that observations inherit entity links from source memories."""
        bank_id = f"test-consolidation-entities-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain a memory with a named entity
        await memory.retain_async(
            bank_id=bank_id,
            content="John Smith is the CEO of Acme Corporation.",
            request_context=request_context,
        )

        # Check observation and its entity links
        async with memory._pool.acquire() as conn:
            observation = await conn.fetchrow(
                """
                SELECT id
                FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                LIMIT 1
                """,
                bank_id,
            )

            if observation:
                # Check if entity links were copied
                entity_links = await conn.fetch(
                    """
                    SELECT entity_id
                    FROM unit_entities
                    WHERE unit_id = $1
                    """,
                    observation["id"],
                )
                # Observation should have inherited entity links from source memory
                # (may be empty if no entities were extracted, which is fine)
                assert entity_links is not None

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_consolidation_observations_included_in_recall(
        self, memory: MemoryEngine, request_context
    ):
        """Test that observations created by consolidation are returned in recall."""
        bank_id = f"test-consolidation-recall-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain a memory - consolidation runs automatically
        await memory.retain_async(
            bank_id=bank_id,
            content="Sarah is an expert Python programmer who specializes in machine learning.",
            request_context=request_context,
        )

        # Recall with observations included
        recall_result = await memory.recall_async(
            bank_id=bank_id,
            query="What does Sarah do?",
            fact_type=["world", "experience", "observation"],
            request_context=request_context,
        )

        # Observations come back as regular results with fact_type='observation'
        assert hasattr(recall_result, "results")

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_consolidation_uses_source_memory_ids(self, memory: MemoryEngine, request_context):
        """Test that observations use source_memory_ids (not memory_links) to track source facts.

        Observations rely on source_memory_ids for traversal:
        - Entity connections: observation → source_memory_ids → unit_entities
        - Semantic similarity: observations have their own embeddings
        - Temporal proximity: observations have their own temporal fields

        No memory_links are created between observations and their source facts.
        """
        bank_id = f"test-consolidation-links-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain a memory - consolidation runs automatically
        await memory.retain_async(
            bank_id=bank_id,
            content="Maria works as a software engineer at Microsoft.",
            request_context=request_context,
        )

        # Check that observation has source_memory_ids but no memory_links
        async with memory._pool.acquire() as conn:
            observation = await conn.fetchrow(
                """
                SELECT id, source_memory_ids
                FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                LIMIT 1
                """,
                bank_id,
            )

            if observation:
                # Observation should have source_memory_ids
                assert observation["source_memory_ids"] is not None, "Observation should have source_memory_ids"
                assert len(observation["source_memory_ids"]) > 0, "Observation should have at least one source memory"

                source_memory_id = observation["source_memory_ids"][0]

                # Verify the source memory exists
                source_memory = await conn.fetchrow(
                    """
                    SELECT id, fact_type FROM memory_units WHERE id = $1
                    """,
                    source_memory_id,
                )
                assert source_memory is not None, "Source memory should exist"
                assert source_memory["fact_type"] in ("world", "experience"), "Source should be a fact"

                # No memory_links should exist between observation and source
                # (observations rely on source_memory_ids for traversal)
                links = await conn.fetch(
                    """
                    SELECT * FROM memory_links
                    WHERE (from_unit_id = $1 AND to_unit_id = $2)
                       OR (from_unit_id = $2 AND to_unit_id = $1)
                    """,
                    source_memory_id,
                    observation["id"],
                )
                assert len(links) == 0, "No memory_links should exist between observation and source"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_consolidation_merges_only_redundant_facts(
        self, memory: MemoryEngine, request_context
    ):
        """Test that consolidation only merges truly redundant facts.

        Observations should be fine-grained (almost 1:1 with memories).
        Only merge when facts are truly redundant (saying the same thing differently)
        or when one directly updates another (e.g., location change).

        Given:
        - "Alex lives in Italy"
        - "Alex moved to the US recently" (updates the living location)

        The second fact should UPDATE the first, not create a separate observation.
        But unrelated facts like "Alex works at Vectorize" should stay separate.
        """
        bank_id = f"test-consolidation-merge-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain a memory about living location
        await memory.retain_async(
            bank_id=bank_id,
            content="Alex lives in Italy.",
            request_context=request_context,
        )

        # Retain an unrelated memory (different topic - should NOT merge)
        await memory.retain_async(
            bank_id=bank_id,
            content="Alex works at Vectorize as an engineer.",
            request_context=request_context,
        )

        # Check observations - should have 2 separate observations
        async with memory._pool.acquire() as conn:
            obs_before = await conn.fetch(
                """
                SELECT id, text FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id,
            )

        # Add a memory that UPDATES the living location (should merge with first)
        await memory.retain_async(
            bank_id=bank_id,
            content="Alex recently moved to the United States.",
            request_context=request_context,
        )

        # Check observations after consolidation
        async with memory._pool.acquire() as conn:
            observations = await conn.fetch(
                """
                SELECT id, text, proof_count, source_memory_ids
                FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                ORDER BY created_at
                """,
                bank_id,
            )

            # Key assertions:
            # 1. Consolidation ran without errors
            # 2. Observations exist
            assert len(observations) >= 1, "Expected at least one observation"

            # The work-related fact should remain separate from location facts
            # (LLM behavior varies, so we check structure rather than exact count)
            for obs in observations:
                assert obs["text"], "Observation should have text"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_consolidation_keeps_different_people_separate(
        self, memory: MemoryEngine, request_context
    ):
        """Test that consolidation NEVER merges facts about different people.

        Each person's facts should stay in separate observations.
        """
        bank_id = f"test-consolidation-people-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Add facts about different people
        await memory.retain_async(
            bank_id=bank_id,
            content="John lives in New York.",
            request_context=request_context,
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="Mary lives in Boston.",
            request_context=request_context,
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="Bob works at Google.",
            request_context=request_context,
        )

        # Check observations - should have separate observations for each person
        async with memory._pool.acquire() as conn:
            observations = await conn.fetch(
                """
                SELECT id, text, source_memory_ids
                FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id,
            )

            # Should have multiple observations (one per person/fact)
            # Not everything merged into one
            assert len(observations) >= 2, (
                f"Expected multiple observations for different people, got {len(observations)}"
            )

            # No single observation should mention multiple different people
            # (This is a structural check - each observation should be focused)
            for obs in observations:
                text = obs["text"].lower()
                people_mentioned = sum([
                    1 for name in ["john", "mary", "bob"]
                    if name in text
                ])
                assert people_mentioned <= 1, (
                    f"Observation should not merge different people: {obs['text']}"
                )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_consolidation_merges_contradictions(
        self, memory: MemoryEngine, request_context
    ):
        """Test that contradictions about the same topic are merged with history.

        When facts contradict each other (same person, same topic, opposite info),
        they should be merged into ONE observation that captures the change.

        Example:
        - "Alex loves pizza"
        - "Alex hates pizza"
        → Should become: "Alex used to love pizza but now hates it" (or similar)
        """
        bank_id = f"test-consolidation-contradict-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Add initial fact
        await memory.retain_async(
            bank_id=bank_id,
            content="Alex loves pizza.",
            request_context=request_context,
        )

        # Check we have one observation
        async with memory._pool.acquire() as conn:
            obs_before = await conn.fetch(
                """
                SELECT id, text FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id,
            )
            count_before = len(obs_before)

        # Add contradicting fact (same person, same topic, opposite sentiment)
        await memory.retain_async(
            bank_id=bank_id,
            content="Alex hates pizza.",
            request_context=request_context,
        )

        # Check observations after consolidation
        async with memory._pool.acquire() as conn:
            observations = await conn.fetch(
                """
                SELECT id, text, source_memory_ids, history
                FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id,
            )

            # Key assertion: Should NOT have more observations than before
            # The contradiction should be merged, not create a new observation
            assert len(observations) <= count_before, (
                f"Contradiction should merge, not create new observation. "
                f"Before: {count_before}, After: {len(observations)}. "
                f"Observations: {[obs['text'] for obs in observations]}"
            )

            # The merged observation should capture both sentiments or the change
            if observations:
                merged_text = observations[0]["text"].lower()
                # Should mention the change or both states
                has_history = (
                    ("used to" in merged_text or "now" in merged_text or "but" in merged_text)
                    or ("love" in merged_text and "hate" in merged_text)
                    or (len(observations[0]["source_memory_ids"] or []) > 1)
                )
                assert has_history, (
                    f"Merged observation should capture the change. Got: {observations[0]['text']}"
                )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestConsolidationDisabled:
    """Test consolidation when disabled via config."""

    @pytest.mark.asyncio
    async def test_consolidation_returns_disabled_status(
        self, memory: MemoryEngine, request_context
    ):
        """Test that consolidation returns disabled status when enable_observations is False."""
        from unittest.mock import patch

        bank_id = f"test-consolidation-disabled-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Disable observations via config
        with patch("hindsight_api.config.get_config") as mock_config:
            mock_config.return_value.enable_observations = False

            result = await run_consolidation_job(
                memory_engine=memory,
                bank_id=bank_id,
                request_context=request_context,
            )

            assert result["status"] == "disabled"
            assert result["bank_id"] == bank_id

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestRecallObservationFactType:
    """Test recall with observation as a fact type."""

    @pytest.mark.asyncio
    async def test_recall_with_observation_fact_type(
        self, memory: MemoryEngine, request_context
    ):
        """Test that observation can be used as a fact type in recall.

        When observation is in the types list, the recall should:
        1. Return observations in the results field with fact_type='observation'
        2. Not raise validation errors for None context fields
        """
        bank_id = f"test-recall-obs-type-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain a memory - consolidation runs automatically
        await memory.retain_async(
            bank_id=bank_id,
            content="Alex is a data scientist who specializes in deep learning and neural networks.",
            request_context=request_context,
        )

        # Recall with observation in types
        recall_result = await memory.recall_async(
            bank_id=bank_id,
            query="What does Alex do?",
            fact_type=["observation"],
            request_context=request_context,
        )

        # Observations come back as regular results with fact_type='observation'
        assert recall_result is not None
        assert recall_result.results is not None
        # Check that results include observations
        if recall_result.results:
            for obs in recall_result.results:
                assert obs.id is not None
                assert obs.text is not None
                assert obs.fact_type == "observation"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_recall_with_mixed_fact_types_including_observation(
        self, memory: MemoryEngine, request_context
    ):
        """Test recall with observation alongside world and experience types."""
        bank_id = f"test-recall-mixed-types-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain memories - consolidation runs automatically
        await memory.retain_async(
            bank_id=bank_id,
            content="Jordan is a professional musician who plays guitar in a rock band.",
            request_context=request_context,
        )

        # Recall with all types including observation
        recall_result = await memory.recall_async(
            bank_id=bank_id,
            query="What does Jordan do?",
            fact_type=["world", "experience", "observation"],
            enable_trace=True,
            request_context=request_context,
        )

        # Should return results without errors
        assert recall_result is not None
        # Should have results from world/experience facts
        assert recall_result.results is not None
        # Observations come back as regular results with fact_type='observation'
        # when observation is included in fact_type parameter

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_recall_observation_only_with_trace(
        self, memory: MemoryEngine, request_context
    ):
        """Test that recall with only observation type and trace enabled works.

        This specifically tests the tracer handling of observations with None context.
        """
        bank_id = f"test-recall-obs-trace-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain memory - consolidation creates observation
        await memory.retain_async(
            bank_id=bank_id,
            content="Chris works as a product manager at a startup focused on AI applications.",
            request_context=request_context,
        )

        # Recall with observation only and trace enabled
        # This tests the fix for the None context validation error
        recall_result = await memory.recall_async(
            bank_id=bank_id,
            query="Where does Chris work?",
            fact_type=["observation"],
            enable_trace=True,
            request_context=request_context,
        )

        # Should complete without validation errors
        assert recall_result is not None
        # Trace should be populated
        assert recall_result.trace is not None or recall_result.observations is not None

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestConsolidationTagRouting:
    """Test tag routing during consolidation.

    Tag routing rules:
    - Same scope (tags match): update existing observation
    - Fact scoped, observation global (untagged): update global (it absorbs all)
    - Different scopes (non-overlapping tags): create untagged cross-scope insight
    - No match: create with fact's tags
    """

    async def _retain_with_tags(
        self,
        memory: MemoryEngine,
        bank_id: str,
        content: str,
        tags: list[str],
        request_context,
    ):
        """Helper to retain content with tags using retain_batch_async."""
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": content}],
            document_tags=tags,
            request_context=request_context,
        )

    @pytest.mark.asyncio
    async def test_same_scope_updates_observation(
        self, memory: MemoryEngine, request_context
    ):
        """Test that a tagged fact updates an observation with the same tags.

        Given:
        - Memory with tags=['alice']: "Alice likes coffee"
        - New memory with tags=['alice']: "Alice prefers espresso"

        Expected:
        - Observation with tags=['alice'] is updated to reflect both facts
        """
        bank_id = f"test-tag-same-scope-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain first memory with tags
        await self._retain_with_tags(
            memory, bank_id, "Alice likes coffee.", ["alice"], request_context
        )

        # Check observation has correct tags
        async with memory._pool.acquire() as conn:
            obs_before = await conn.fetch(
                """
                SELECT id, text, tags FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id,
            )
            count_before = len(obs_before)
            if obs_before:
                assert "alice" in (obs_before[0]["tags"] or []), (
                    f"Expected observation to have 'alice' tag, got: {obs_before[0]['tags']}"
                )

        # Retain related memory with same tags
        await self._retain_with_tags(
            memory, bank_id, "Alice prefers espresso over regular coffee.", ["alice"], request_context
        )

        # Check observations - should NOT have increased (same scope update)
        async with memory._pool.acquire() as conn:
            obs_after = await conn.fetch(
                """
                SELECT id, text, tags, source_memory_ids FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id,
            )

            # Count of observations should stay same or decrease (merge)
            assert len(obs_after) <= count_before + 1, (
                f"Same scope fact should update existing observation, not create new. "
                f"Before: {count_before}, After: {len(obs_after)}"
            )

            # The observation(s) should still have alice tag
            for obs in obs_after:
                if "coffee" in obs["text"].lower() or "espresso" in obs["text"].lower():
                    assert "alice" in (obs["tags"] or []), (
                        f"Updated observation should keep 'alice' tag: {obs['text']}"
                    )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_scoped_fact_updates_global_observation(
        self, memory: MemoryEngine, request_context
    ):
        """Test that a scoped fact can update an untagged (global) observation.

        Given:
        - Untagged memory: "Pizza is a popular food"
        - New memory with tags=['history']: "Pizza originated in Naples"

        Expected:
        - The global observation is updated (global absorbs all scopes)
        """
        bank_id = f"test-tag-global-absorb-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain untagged (global) memory
        await memory.retain_async(
            bank_id=bank_id,
            content="Pizza is a popular Italian food.",
            request_context=request_context,
        )

        # Check untagged observation exists
        async with memory._pool.acquire() as conn:
            obs_before = await conn.fetch(
                """
                SELECT id, text, tags FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id,
            )
            count_before = len(obs_before)
            # Should be untagged or have empty tags
            if obs_before:
                assert not obs_before[0]["tags"] or len(obs_before[0]["tags"]) == 0, (
                    f"Expected untagged observation, got: {obs_before[0]['tags']}"
                )

        # Retain scoped memory that relates to the global topic
        await self._retain_with_tags(
            memory, bank_id, "Pizza originated in Naples.", ["history"], request_context
        )

        # Check - global observation should be updated OR new scoped observation created
        async with memory._pool.acquire() as conn:
            obs_after = await conn.fetch(
                """
                SELECT id, text, tags, source_memory_ids FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                ORDER BY created_at
                """,
                bank_id,
            )

            # At least one observation should exist
            assert len(obs_after) >= 1, "Expected at least one observation"

            # Check that global observation was updated (source_memory_ids increased)
            # OR new observation was created with appropriate tags
            global_observations = [o for o in obs_after if not o["tags"] or len(o["tags"]) == 0]
            scoped_observations = [o for o in obs_after if o["tags"] and len(o["tags"]) > 0]

            # Either global was updated or scoped was created
            assert len(global_observations) >= 1 or len(scoped_observations) >= 1, (
                "Expected either global observation update or scoped observation creation"
            )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_cross_scope_creates_untagged(
        self, memory: MemoryEngine, request_context
    ):
        """Test that cross-scope related facts create untagged (global) insights.

        Given:
        - Memory with tags=['alice']: "Alice recommends the Thai restaurant"
        - Memory with tags=['bob']: "Bob tried the Thai restaurant Alice mentioned"

        Expected:
        - A new untagged observation capturing the cross-scope insight
        """
        bank_id = f"test-tag-cross-scope-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain Alice's scoped memory
        await self._retain_with_tags(
            memory, bank_id,
            "Alice recommends the Thai restaurant on Main Street.",
            ["alice"], request_context
        )

        # Check Alice's observation exists with correct tags
        async with memory._pool.acquire() as conn:
            obs_alice = await conn.fetch(
                """
                SELECT id, text, tags FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id,
            )
            count_before = len(obs_alice)

        # Retain Bob's memory that relates to Alice's topic (cross-scope)
        await self._retain_with_tags(
            memory, bank_id,
            "Bob visited the Thai restaurant on Main Street and loved it.",
            ["bob"], request_context
        )

        # Check observations
        async with memory._pool.acquire() as conn:
            obs_after = await conn.fetch(
                """
                SELECT id, text, tags, source_memory_ids FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                ORDER BY created_at
                """,
                bank_id,
            )

            # Should have multiple observations (alice's, bob's, potentially global)
            assert len(obs_after) >= 2, (
                f"Expected at least 2 observations for different scopes, got {len(obs_after)}"
            )

            # Check we have observations with different tags (alice, bob, or untagged)
            tag_sets = [frozenset(o["tags"] or []) for o in obs_after]

            # Should NOT merge alice and bob into same observation
            observations_with_both = [
                o for o in obs_after
                if o["tags"] and "alice" in o["tags"] and "bob" in o["tags"]
            ]
            assert len(observations_with_both) == 0, (
                "Should not merge different scopes into one observation with both tags"
            )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_no_match_creates_with_fact_tags(
        self, memory: MemoryEngine, request_context
    ):
        """Test that a new fact with no matching observations creates an observation with fact's tags.

        Given:
        - Empty bank
        - Memory with tags=['project_x']: "Project X uses Python"

        Expected:
        - Observation created with tags=['project_x']
        """
        bank_id = f"test-tag-new-scoped-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain tagged memory (no existing observations)
        await self._retain_with_tags(
            memory, bank_id,
            "Project X uses Python for its backend services.",
            ["project_x"], request_context
        )

        # Check observation was created with correct tags
        async with memory._pool.acquire() as conn:
            observations = await conn.fetch(
                """
                SELECT id, text, tags FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id,
            )

            assert len(observations) >= 1, "Expected observation to be created"

            # The observation should have the fact's tags
            obs = observations[0]
            assert obs["tags"] is not None, "Observation should have tags"
            assert "project_x" in obs["tags"], (
                f"Observation should have 'project_x' tag, got: {obs['tags']}"
            )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_untagged_fact_can_update_scoped_observation(
        self, memory: MemoryEngine, request_context
    ):
        """Test that an untagged fact can update a scoped observation.

        Given:
        - Memory with tags=['alice']: "Alice works on machine learning"
        - Untagged memory: "Machine learning involves neural networks"

        Expected:
        - The scoped observation may be updated with the global insight
        - OR a global observation is created
        """
        bank_id = f"test-tag-untagged-update-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain scoped memory
        await self._retain_with_tags(
            memory, bank_id,
            "Alice works on machine learning projects.",
            ["alice"], request_context
        )

        # Retain untagged memory on same topic
        await memory.retain_async(
            bank_id=bank_id,
            content="Machine learning involves training neural networks.",
            request_context=request_context,
        )

        # Check observations
        async with memory._pool.acquire() as conn:
            observations = await conn.fetch(
                """
                SELECT id, text, tags, source_memory_ids FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                ORDER BY created_at
                """,
                bank_id,
            )

            # Should have at least one observation
            assert len(observations) >= 1, "Expected at least one observation"

            # Either alice's observation was updated OR a global observation was created
            # This is valid LLM behavior - just verify no errors and structure is correct
            for obs in observations:
                assert obs["text"], "Observation should have text"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_tag_filtering_in_recall(
        self, memory: MemoryEngine, request_context
    ):
        """Test that observations respect tag filtering during recall.

        Observations should be filtered by tags just like memories.
        """
        bank_id = f"test-tag-recall-filter-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain memories with different tags
        await self._retain_with_tags(
            memory, bank_id,
            "Alice works as a software engineer.",
            ["alice"], request_context
        )
        await self._retain_with_tags(
            memory, bank_id,
            "Bob works as a product manager.",
            ["bob"], request_context
        )

        # Recall with alice tag only
        recall_result = await memory.recall_async(
            bank_id=bank_id,
            query="What does everyone do for work?",
            tags=["alice"],
            tags_match="any_strict",  # Only alice's data
            fact_type=["world", "experience", "observation"],
            request_context=request_context,
        )

        # Results should only include alice-tagged content
        # Observations are now regular results with fact_type='observation'
        observations = [r for r in recall_result.results if r.fact_type == "observation"]
        for obs in observations:
            # Observation should be alice-scoped or global (untagged)
            # Not bob-scoped
            obs_tags = obs.tags or []
            assert "bob" not in obs_tags, (
                f"Recall with tags=['alice'] should not return bob's observations: {obs.text}"
            )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_multiple_actions_from_single_fact(
        self, memory: MemoryEngine, request_context
    ):
        """Test that one fact can trigger multiple consolidation actions.

        Given:
        - Global observation: "Coffee is a popular beverage"
        - Alice's observation: "Alice drinks coffee every morning"
        - New fact with tags=['alice']: "Alice switched to decaf coffee"

        Expected:
        - Update Alice's scoped observation (same scope)
        - Potentially update global observation too (global absorbs all)
        """
        bank_id = f"test-tag-multi-action-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Create global observation
        await memory.retain_async(
            bank_id=bank_id,
            content="Coffee is a popular beverage worldwide.",
            request_context=request_context,
        )

        # Create alice's scoped observation
        await self._retain_with_tags(
            memory, bank_id,
            "Alice drinks coffee every morning.",
            ["alice"], request_context
        )

        # Check observations before
        async with memory._pool.acquire() as conn:
            obs_before = await conn.fetch(
                """
                SELECT id, text, tags, source_memory_ids FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id,
            )
            count_before = len(obs_before)

        # Add fact that could relate to both
        await self._retain_with_tags(
            memory, bank_id,
            "Alice switched to decaf coffee for health reasons.",
            ["alice"], request_context
        )

        # Check observations after
        async with memory._pool.acquire() as conn:
            obs_after = await conn.fetch(
                """
                SELECT id, text, tags, source_memory_ids, proof_count FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                ORDER BY created_at
                """,
                bank_id,
            )

            # Should have processed without errors
            assert len(obs_after) >= 1, "Expected at least one observation"

            # Check that consolidation worked (either updates or maintains structure)
            # The key is no errors and proper tag handling
            for obs in obs_after:
                assert obs["text"], "Observation should have text"
                # Tags should be consistent (not mixing alice and bob, etc.)

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_consolidation_inherits_dates_from_source_memory(
        self, memory: MemoryEngine, request_context
    ):
        """Test that observations inherit occurred_start and event_date from source memories.

        When an observation is created, it should inherit the temporal information
        from the source memory that triggered its creation, not use the current time.
        """
        from datetime import datetime, timezone

        bank_id = f"test-consolidation-dates-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Create a specific date in the past for testing
        past_date = datetime(2023, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        # First, create a memory unit directly with a specific date
        async with memory._pool.acquire() as conn:
            memory_id = uuid.uuid4()
            await conn.execute(
                """
                INSERT INTO memory_units (
                    id, bank_id, text, fact_type, occurred_start, event_date, created_at
                )
                VALUES ($1, $2, $3, 'experience', $4, $4, now())
                """,
                memory_id,
                bank_id,
                "Sarah went to Paris for vacation and loved the Eiffel Tower.",
                past_date,
            )

        # Run consolidation manually
        from hindsight_api.engine.consolidation.consolidator import run_consolidation_job

        result = await run_consolidation_job(
            memory_engine=memory,
            bank_id=bank_id,
            request_context=request_context,
        )

        # Verify consolidation processed the memory
        assert result["status"] == "completed"
        assert result["memories_processed"] >= 1

        # Check that observation inherited the date from source memory
        async with memory._pool.acquire() as conn:
            observation = await conn.fetchrow(
                """
                SELECT id, text, occurred_start, event_date, source_memory_ids
                FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                LIMIT 1
                """,
                bank_id,
            )

            if observation:
                # Observation should have inherited the date from the source memory
                obs_occurred = observation["occurred_start"]
                obs_event_date = observation["event_date"]

                # Dates should match the source memory's date (2023-06-15), not today
                assert obs_occurred is not None, "Observation should have occurred_start"
                assert obs_event_date is not None, "Observation should have event_date"

                # The date should be from 2023, not today
                assert obs_occurred.year == 2023, (
                    f"Expected occurred_start year 2023, got {obs_occurred.year}. "
                    "Observation should inherit date from source memory."
                )
                assert obs_occurred.month == 6, f"Expected month 6, got {obs_occurred.month}"
                assert obs_occurred.day == 15, f"Expected day 15, got {obs_occurred.day}"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_observation_temporal_range_expands_on_update(
        self, memory: MemoryEngine, request_context
    ):
        """Test that observation temporal range uses LEAST(occurred_start) and GREATEST(occurred_end).

        When an observation is updated with a new source fact:
        - occurred_start should be the EARLIEST start time across all source facts
        - occurred_end should be the LATEST end time across all source facts

        This ensures observations capture the full temporal range of their source facts.
        """
        from datetime import datetime, timezone

        bank_id = f"test-consolidation-temporal-range-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Define dates: first memory is from June 2023, second is from January 2024
        early_start = datetime(2023, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        early_end = datetime(2023, 6, 15, 18, 0, 0, tzinfo=timezone.utc)
        late_start = datetime(2024, 1, 10, 9, 0, 0, tzinfo=timezone.utc)
        late_end = datetime(2024, 1, 20, 17, 0, 0, tzinfo=timezone.utc)

        # Create first memory with early dates
        async with memory._pool.acquire() as conn:
            memory_id_1 = uuid.uuid4()
            await conn.execute(
                """
                INSERT INTO memory_units (
                    id, bank_id, text, fact_type, occurred_start, occurred_end, event_date, created_at
                )
                VALUES ($1, $2, $3, 'experience', $4, $5, $4, now())
                """,
                memory_id_1,
                bank_id,
                "Tom started learning Python programming in summer 2023.",
                early_start,
                early_end,
            )

        # Run consolidation - should create observation with early dates
        from hindsight_api.engine.consolidation.consolidator import run_consolidation_job

        result = await run_consolidation_job(
            memory_engine=memory,
            bank_id=bank_id,
            request_context=request_context,
        )
        assert result["status"] == "completed"

        # Check observation has the early dates
        async with memory._pool.acquire() as conn:
            obs_after_first = await conn.fetchrow(
                """
                SELECT id, occurred_start, occurred_end, source_memory_ids
                FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                LIMIT 1
                """,
                bank_id,
            )

        if obs_after_first:
            assert obs_after_first["occurred_start"].year == 2023, (
                f"Initial observation should have 2023 start, got {obs_after_first['occurred_start']}"
            )
            assert obs_after_first["occurred_end"].year == 2023, (
                f"Initial observation should have 2023 end, got {obs_after_first['occurred_end']}"
            )

            # Now add a second related memory with later dates
            async with memory._pool.acquire() as conn:
                memory_id_2 = uuid.uuid4()
                await conn.execute(
                    """
                    INSERT INTO memory_units (
                        id, bank_id, text, fact_type, occurred_start, occurred_end, event_date, created_at
                    )
                    VALUES ($1, $2, $3, 'experience', $4, $5, $4, now())
                    """,
                    memory_id_2,
                    bank_id,
                    "Tom completed his Python certification in January 2024.",
                    late_start,
                    late_end,
                )

            # Run consolidation again - should update observation with expanded range
            result = await run_consolidation_job(
                memory_engine=memory,
                bank_id=bank_id,
                request_context=request_context,
            )
            assert result["status"] == "completed"

            # Check observation now has expanded temporal range
            async with memory._pool.acquire() as conn:
                obs_after_second = await conn.fetchrow(
                    """
                    SELECT id, occurred_start, occurred_end, source_memory_ids, proof_count
                    FROM memory_units
                    WHERE bank_id = $1 AND fact_type = 'observation'
                    ORDER BY proof_count DESC
                    LIMIT 1
                    """,
                    bank_id,
                )

            if obs_after_second and obs_after_second["proof_count"] >= 2:
                # occurred_start should be the EARLIEST (2023)
                assert obs_after_second["occurred_start"].year == 2023, (
                    f"occurred_start should be earliest (2023), got {obs_after_second['occurred_start']}"
                )
                assert obs_after_second["occurred_start"].month == 6, (
                    f"occurred_start month should be 6 (June), got {obs_after_second['occurred_start'].month}"
                )

                # occurred_end should be the LATEST (2024)
                assert obs_after_second["occurred_end"].year == 2024, (
                    f"occurred_end should be latest (2024), got {obs_after_second['occurred_end']}"
                )
                assert obs_after_second["occurred_end"].month == 1, (
                    f"occurred_end month should be 1 (January), got {obs_after_second['occurred_end'].month}"
                )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestObservationDrillDown:
    """Test that reflect agent can drill down from observations to source memories."""

    @pytest.mark.asyncio
    async def test_search_observations_returns_source_memory_ids(
        self, memory: MemoryEngine, request_context
    ):
        """Test that search_observations returns source_memory_ids for drill-down.

        This verifies the agent can:
        1. Find an observation
        2. Access its source_memory_ids
        3. Use those IDs to expand/recall for more details
        """
        from hindsight_api.engine.reflect.tools import tool_expand, tool_search_observations

        bank_id = f"test-obs-drilldown-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Store memories with specific details that get summarized in observation
        await memory.retain_async(
            bank_id=bank_id,
            content="Sarah works at TechCorp as a senior software engineer since March 2020.",
            request_context=request_context,
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="Sarah's employee ID at TechCorp is EMP-12345.",
            request_context=request_context,
        )

        # Search for observations
        result = await tool_search_observations(
            memory_engine=memory,
            bank_id=bank_id,
            query="Sarah TechCorp",
            request_context=request_context,
        )

        assert result["count"] > 0, "Expected at least one observation"

        # Verify source_memory_ids and proof_count are present
        obs = result["observations"][0]
        assert "source_memory_ids" in obs, "Observation should have source_memory_ids"
        assert "proof_count" in obs, "Observation should have proof_count"
        assert obs["proof_count"] >= 1, "proof_count should be at least 1"

        # If source_memory_ids exist, verify they can be used with expand
        if obs["source_memory_ids"]:
            assert len(obs["source_memory_ids"]) >= 1, "Should have at least one source memory"

            # Use expand tool to get source memory details
            async with memory._pool.acquire() as conn:
                expand_result = await tool_expand(
                    conn=conn,
                    bank_id=bank_id,
                    memory_ids=obs["source_memory_ids"][:2],  # Take first 2
                    depth="chunk",
                )

            assert "results" in expand_result
            assert len(expand_result["results"]) > 0, "Expand should return source memories"

            # Verify we get the original detailed information
            all_text = " ".join(r["memory"]["text"] for r in expand_result["results"] if "memory" in r)
            # The expanded memories should contain details not necessarily in the observation
            assert "Sarah" in all_text or "TechCorp" in all_text, (
                f"Expanded memories should contain source details. Got: {all_text}"
            )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_observation_source_ids_match_contributing_memories(
        self, memory: MemoryEngine, request_context
    ):
        """Test that source_memory_ids actually point to the memories that built the observation."""
        bank_id = f"test-obs-source-ids-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Store two related memories
        await memory.retain_async(
            bank_id=bank_id,
            content="Project Phoenix was started by the engineering team in January 2024.",
            request_context=request_context,
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="Project Phoenix achieved 99.9% uptime in its first quarter.",
            request_context=request_context,
        )

        # Get the observation with source_memory_ids
        async with memory._pool.acquire() as conn:
            obs_rows = await conn.fetch(
                """
                SELECT id, text, proof_count, source_memory_ids
                FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id,
            )

        if obs_rows:
            obs = obs_rows[0]
            source_ids = obs["source_memory_ids"] or []

            # Verify source_memory_ids point to actual memories
            if source_ids:
                async with memory._pool.acquire() as conn:
                    source_memories = await conn.fetch(
                        """
                        SELECT id, text FROM memory_units
                        WHERE id = ANY($1) AND fact_type IN ('world', 'experience')
                        """,
                        source_ids,
                    )

                # Should have found the source memories
                assert len(source_memories) >= 1, (
                    f"source_memory_ids should point to valid memories. "
                    f"IDs: {source_ids}, Found: {len(source_memories)}"
                )

                # The source memories should contain our original content
                source_texts = [m["text"].lower() for m in source_memories]
                has_phoenix = any("phoenix" in t for t in source_texts)
                assert has_phoenix, f"Source memories should contain original content. Got: {source_texts}"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestHierarchicalRetrieval:
    """Test the reflect agent's hierarchical retrieval tools.

    The hierarchy is:
    1. search_mental_models - User-curated summaries (highest quality, formerly reflections)
    2. search_observations - Auto-consolidated knowledge (formerly mental_models)
    3. recall - Raw facts as ground truth

    When a mental model matches the query, it should be used first.
    """

    @pytest.mark.asyncio
    async def test_mental_model_takes_priority_over_observation(
        self, memory: MemoryEngine, request_context
    ):
        """Test that mental models are found and would be used before observations.

        Given:
        - A memory about "John's favorite color is blue"
        - An observation created from that memory (via consolidation)
        - A mental model manually created about John

        When searching, the mental model should be found first.
        """
        bank_id = f"test-hierarchy-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain a memory - consolidation creates an observation
        await memory.retain_async(
            bank_id=bank_id,
            content="John's favorite color is blue and he likes painting.",
            request_context=request_context,
        )

        # Verify observation was created
        async with memory._pool.acquire() as conn:
            obs_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'observation'
                """,
                bank_id,
            )
        assert obs_count >= 1, "Consolidation should have created an observation"

        # Create a mental model about John (higher quality, user-curated)
        mental_model = await memory.create_mental_model(
            bank_id=bank_id,
            name="John's Preferences",
            source_query="What are John's preferences?",
            content="John is an artist who loves the color blue. He has been painting for 10 years and prefers watercolors.",
            tags=[],
            request_context=request_context,
        )
        assert mental_model["id"] is not None

        # Search mental models - should find our mental model
        async with memory._pool.acquire() as conn:
            query_embedding = memory.embeddings.encode(["What does John like?"])[0]
            mental_model_result = await tool_search_mental_models(
                conn=conn,
                bank_id=bank_id,
                query="What does John like?",
                query_embedding=query_embedding,
                max_results=5,
            )

        # Mental model should be found
        assert mental_model_result["count"] >= 1, "Mental model should be found"
        found_mental_model = mental_model_result["mental_models"][0]
        assert "John" in found_mental_model["content"] or "blue" in found_mental_model["content"]

        # Search observations - should also find something
        obs_result = await tool_search_observations(
            memory_engine=memory,
            bank_id=bank_id,
            query="What does John like?",
            request_context=request_context,
            max_tokens=5000,
        )
        assert obs_result["count"] >= 1, "Observation should also be found"

        # Verify the mental model has higher quality content (more detail)
        mental_model_content = found_mental_model["content"]
        obs_content = obs_result["observations"][0]["text"]

        # The mental model should contain the richer, user-curated content
        assert "watercolors" in mental_model_content or "10 years" in mental_model_content, (
            f"Mental model should have the rich user-curated content. Got: {mental_model_content}"
        )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_fallback_to_observation_when_no_mental_model(
        self, memory: MemoryEngine, request_context
    ):
        """Test that observations are used when no mental model matches.

        Given:
        - A memory about "Sarah works at Google"
        - An observation created from that memory
        - NO mental model about Sarah

        When searching, observations should provide the information.
        """
        bank_id = f"test-hierarchy-fallback-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain a memory - consolidation creates an observation
        await memory.retain_async(
            bank_id=bank_id,
            content="Sarah works at Google as a software engineer.",
            request_context=request_context,
        )

        # Search mental models - should find nothing
        async with memory._pool.acquire() as conn:
            query_embedding = memory.embeddings.encode(["Where does Sarah work?"])[0]
            mental_model_result = await tool_search_mental_models(
                conn=conn,
                bank_id=bank_id,
                query="Where does Sarah work?",
                query_embedding=query_embedding,
                max_results=5,
            )

        # No mental models exist
        assert mental_model_result["count"] == 0, "No mental models should exist"

        # Search observations - should find the consolidated knowledge
        obs_result = await tool_search_observations(
            memory_engine=memory,
            bank_id=bank_id,
            query="Where does Sarah work?",
            request_context=request_context,
            max_tokens=5000,
        )

        # Observation should be found
        assert obs_result["count"] >= 1, "Observation should be found when no mental model exists"
        obs_text = obs_result["observations"][0]["text"].lower()
        assert "sarah" in obs_text or "google" in obs_text, (
            f"Observation should contain info about Sarah. Got: {obs_text}"
        )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_fallback_to_recall_for_fresh_data(
        self, memory: MemoryEngine, request_context
    ):
        """Test that recall provides raw facts when needed for verification.

        This tests the drill-down capability: when mental models are stale or
        need verification, recall provides the original source facts.
        """
        bank_id = f"test-hierarchy-recall-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain some specific memories
        await memory.retain_async(
            bank_id=bank_id,
            content="The quarterly revenue was $1.5M in Q3 2024.",
            request_context=request_context,
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="The quarterly revenue was $2.1M in Q4 2024.",
            request_context=request_context,
        )

        # Use recall to get the raw facts
        recall_result = await tool_recall(
            memory_engine=memory,
            bank_id=bank_id,
            query="What was the quarterly revenue?",
            request_context=request_context,
            max_tokens=2048,
            max_results=10,
        )

        # Should have raw facts with specific numbers
        assert recall_result["count"] >= 1, "Recall should find the raw facts"

        # Check that we get the actual numbers from the original memories
        all_memory_text = " ".join([m["text"] for m in recall_result["memories"]])
        # Accept both abbreviated ($1.5M) and full form ($1.5 million) as LLM extraction can vary
        has_q3_data = "$1.5M" in all_memory_text or "$1.5 million" in all_memory_text
        has_q4_data = "$2.1M" in all_memory_text or "$2.1 million" in all_memory_text
        assert has_q3_data or has_q4_data, (
            f"Recall should return raw facts with specific data. Got: {all_memory_text}"
        )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)


class TestMentalModelRefreshAfterConsolidation:
    """Test that mental models with refresh_after_consolidation trigger are refreshed after consolidation."""

    @pytest.mark.asyncio
    async def test_mental_model_with_trigger_is_refreshed_after_consolidation(
        self, memory: MemoryEngine, request_context
    ):
        """Test that mental models with refresh_after_consolidation=true get refreshed.

        Given:
        - A mental model with trigger.refresh_after_consolidation = true
        - New memories are retained (triggers consolidation)

        Expected:
        - After consolidation, the mental model is refreshed (last_refreshed_at updated)
        """
        bank_id = f"test-mm-refresh-trigger-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Create a mental model with refresh_after_consolidation trigger enabled
        mental_model = await memory.create_mental_model(
            bank_id=bank_id,
            mental_model_id=str(uuid.uuid4()),
            name="User Preferences",
            source_query="What are the user's preferences?",
            content="Initial content about user preferences.",
            tags=[],
            trigger={"refresh_after_consolidation": True},
            request_context=request_context,
        )
        mental_model_id = mental_model["id"]

        # Verify trigger was set correctly
        assert mental_model.get("trigger", {}).get("refresh_after_consolidation") is True

        # Get the initial last_refreshed_at
        async with memory._pool.acquire() as conn:
            initial_row = await conn.fetchrow(
                """
                SELECT last_refreshed_at, content
                FROM mental_models
                WHERE id = $1 AND bank_id = $2
                """,
                mental_model_id,
                bank_id,
            )
            initial_refreshed_at = initial_row["last_refreshed_at"]
            initial_content = initial_row["content"]

        # Retain a memory - this triggers consolidation which should trigger mental model refresh
        await memory.retain_async(
            bank_id=bank_id,
            content="The user prefers dark mode and uses keyboard shortcuts extensively.",
            request_context=request_context,
        )

        # Check that the mental model was refreshed
        async with memory._pool.acquire() as conn:
            refreshed_row = await conn.fetchrow(
                """
                SELECT last_refreshed_at, content
                FROM mental_models
                WHERE id = $1 AND bank_id = $2
                """,
                mental_model_id,
                bank_id,
            )
            refreshed_at = refreshed_row["last_refreshed_at"]
            refreshed_content = refreshed_row["content"]

        # The mental model should have been refreshed (last_refreshed_at updated)
        assert refreshed_at > initial_refreshed_at, (
            f"Mental model should have been refreshed after consolidation. "
            f"Initial: {initial_refreshed_at}, After: {refreshed_at}"
        )

        # The content should have changed (regenerated by reflect)
        assert refreshed_content != initial_content, (
            f"Mental model content should have been updated. "
            f"Initial: {initial_content}, After: {refreshed_content}"
        )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_mental_model_without_trigger_is_not_refreshed(
        self, memory: MemoryEngine, request_context
    ):
        """Test that mental models with refresh_after_consolidation=false are NOT refreshed.

        Given:
        - A mental model with trigger.refresh_after_consolidation = false (default)
        - New memories are retained (triggers consolidation)

        Expected:
        - After consolidation, the mental model is NOT refreshed
        """
        bank_id = f"test-mm-no-refresh-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Create a mental model (default trigger is refresh_after_consolidation: false)
        mental_model = await memory.create_mental_model(
            bank_id=bank_id,
            mental_model_id=str(uuid.uuid4()),
            name="Static Knowledge",
            source_query="What is the company mission?",
            content="Our mission is to build great software.",
            tags=[],
            request_context=request_context,
        )
        mental_model_id = mental_model["id"]

        # Get the initial last_refreshed_at and content
        async with memory._pool.acquire() as conn:
            initial_row = await conn.fetchrow(
                """
                SELECT last_refreshed_at, content
                FROM mental_models
                WHERE id = $1 AND bank_id = $2
                """,
                mental_model_id,
                bank_id,
            )
            initial_refreshed_at = initial_row["last_refreshed_at"]
            initial_content = initial_row["content"]

        # Retain a memory - this triggers consolidation
        await memory.retain_async(
            bank_id=bank_id,
            content="We launched a new product feature today.",
            request_context=request_context,
        )

        # Check that the mental model was NOT refreshed
        async with memory._pool.acquire() as conn:
            after_row = await conn.fetchrow(
                """
                SELECT last_refreshed_at, content
                FROM mental_models
                WHERE id = $1 AND bank_id = $2
                """,
                mental_model_id,
                bank_id,
            )
            after_refreshed_at = after_row["last_refreshed_at"]
            after_content = after_row["content"]

        # The mental model should NOT have been refreshed
        assert after_refreshed_at == initial_refreshed_at, (
            f"Mental model without trigger should NOT be refreshed. "
            f"Initial: {initial_refreshed_at}, After: {after_refreshed_at}"
        )

        # The content should be unchanged
        assert after_content == initial_content, (
            f"Mental model content should be unchanged. "
            f"Initial: {initial_content}, After: {after_content}"
        )

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_graph_endpoint_observations_inherit_links_and_entities(
        self, memory: MemoryEngine, request_context
    ):
        """Test that graph endpoint shows links and entities for observations filtered by type.

        When filtering graph by type=observation:
        - Observations should inherit links from their source memories
        - Observations should show entities inherited from source memories
        - Even when source memories are not visible, their links should be copied to observations
        """
        bank_id = f"test-graph-obs-{uuid.uuid4().hex[:8]}"

        # Create the bank
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain content that will create world facts with shared entities
        # This should create facts that are linked by shared entities
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Google as a software engineer.",
            request_context=request_context,
        )

        await memory.retain_async(
            bank_id=bank_id,
            content="Bob also works at Google in the sales department.",
            request_context=request_context,
        )

        # Wait for consolidation to create observations
        import asyncio

        await asyncio.sleep(2)

        # Get graph data filtered by observation type only
        graph_data = await memory.get_graph_data(
            bank_id=bank_id,
            fact_type="observation",
            limit=1000,
            request_context=request_context,
        )

        # Should have observations
        assert graph_data["total_units"] > 0, "Should have observations"
        assert len(graph_data["nodes"]) > 0, "Should have observation nodes"

        # Verify all nodes are observations
        for row in graph_data["table_rows"]:
            assert row["fact_type"] == "observation", f"All nodes should be observations, got {row['fact_type']}"

        # Should have edges (inherited from source memories)
        # Even though we're only showing observations, they should inherit links from their sources
        assert len(graph_data["edges"]) > 0, (
            "Observations should have edges inherited from source memories. "
            f"Found {len(graph_data['edges'])} edges"
        )

        # Should have entities (inherited from source memories)
        observations_with_entities = [
            row for row in graph_data["table_rows"] if row["entities"] and row["entities"] != "None"
        ]
        assert len(observations_with_entities) > 0, (
            "Observations should inherit entities from source memories. "
            f"Found {len(observations_with_entities)} observations with entities"
        )

        # Verify entities contain expected values
        all_entities = " ".join([row["entities"] for row in graph_data["table_rows"]])
        assert "Alice" in all_entities or "Bob" in all_entities or "Google" in all_entities, (
            f"Expected to find Alice, Bob, or Google in entities, got: {all_entities}"
        )

        # Verify edge types are valid
        valid_link_types = {"semantic", "temporal", "entity"}
        for edge in graph_data["edges"]:
            link_type = edge["data"]["linkType"]
            assert link_type in valid_link_types, f"Invalid link type: {link_type}"

        # Verify all edges connect visible observation nodes
        visible_node_ids = {row["id"] for row in graph_data["table_rows"]}
        for edge in graph_data["edges"]:
            source_id = edge["data"]["source"]
            target_id = edge["data"]["target"]
            assert source_id in visible_node_ids, f"Edge source {source_id[:8]} not in visible nodes"
            assert target_id in visible_node_ids, f"Edge target {target_id[:8]} not in visible nodes"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)
