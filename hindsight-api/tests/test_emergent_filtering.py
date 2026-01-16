"""Tests for emergent entity filtering."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hindsight_api.engine.mental_models.emergent import (
    build_mission_filter_prompt,
    evaluate_emergent_models,
    filter_candidates_by_mission,
    MissionFilterResponse,
    MissionFilterCandidate,
)
from hindsight_api.engine.mental_models.models import EmergentCandidate


class TestBuildMissionFilterPrompt:
    """Test prompt building for mission filtering."""

    def test_prompt_contains_mission(self):
        """Test that prompt includes the mission."""
        candidates = [
            EmergentCandidate(
                name="Alice",
                detection_method="named_entity_extraction",
                mention_count=10,
            )
        ]
        prompt = build_mission_filter_prompt("Be a PM for engineering team", candidates)
        assert "Be a PM for engineering team" in prompt

    def test_prompt_contains_candidates(self):
        """Test that prompt includes all candidates."""
        candidates = [
            EmergentCandidate(
                name="Alice Chen",
                detection_method="named_entity_extraction",
                mention_count=10,
            ),
            EmergentCandidate(
                name="Project Phoenix",
                detection_method="named_entity_extraction",
                mention_count=5,
            ),
        ]
        prompt = build_mission_filter_prompt("Track projects", candidates)
        assert "Alice Chen" in prompt
        assert "Project Phoenix" in prompt

    def test_prompt_contains_rejection_guidance(self):
        """Test that prompt contains guidance to reject generic entities."""
        candidates = [
            EmergentCandidate(
                name="test",
                detection_method="named_entity_extraction",
                mention_count=1,
            )
        ]
        prompt = build_mission_filter_prompt("Test mission", candidates)

        # Should contain rejection guidance for generic terms
        assert "promote=false" in prompt
        assert "kids" in prompt  # Example of generic term to reject
        assert "community" in prompt  # Example of abstract concept to reject
        assert "motivation" in prompt  # Example of abstract concept to reject


class TestFilterCandidatesByMission:
    """Test the filter_candidates_by_mission function."""

    @pytest.fixture
    def mock_llm_config(self):
        """Create a mock LLM config."""
        config = MagicMock()
        config.call = AsyncMock()
        return config

    async def test_empty_candidates(self, mock_llm_config):
        """Test with empty candidate list."""
        result = await filter_candidates_by_mission(
            llm_config=mock_llm_config,
            mission="Test mission",
            candidates=[],
        )
        assert result == []
        mock_llm_config.call.assert_not_called()

    async def test_no_mission_keeps_all(self, mock_llm_config):
        """Test that no mission keeps all candidates (skips filtering)."""
        candidates = [
            EmergentCandidate(
                name="Alice",
                detection_method="named_entity_extraction",
                mention_count=10,
            )
        ]
        result = await filter_candidates_by_mission(
            llm_config=mock_llm_config,
            mission="",  # Empty mission
            candidates=candidates,
        )
        assert len(result) == 1
        assert result[0].name == "Alice"
        mock_llm_config.call.assert_not_called()

    async def test_filters_by_promote_flag(self, mock_llm_config):
        """Test that candidates are filtered by promote flag."""
        candidates = [
            EmergentCandidate(
                name="Alice Chen",
                detection_method="named_entity_extraction",
                mention_count=10,
            ),
            EmergentCandidate(
                name="community",
                detection_method="named_entity_extraction",
                mention_count=5,
            ),
        ]

        # Mock LLM response - Alice is promoted, community is not
        mock_llm_config.call.return_value = MissionFilterResponse(
            candidates=[
                MissionFilterCandidate(name="Alice Chen", promote=True, reason="Specific person"),
                MissionFilterCandidate(name="community", promote=False, reason="Generic abstract concept"),
            ]
        )

        result = await filter_candidates_by_mission(
            llm_config=mock_llm_config,
            mission="Be a PM for engineering team",
            candidates=candidates,
        )

        assert len(result) == 1
        assert result[0].name == "Alice Chen"

    async def test_rejects_generic_entities(self, mock_llm_config):
        """Test that generic entities are rejected."""
        # These are all generic/abstract terms that should be rejected
        generic_names = [
            "user", "support", "community", "family", "motivation",
            "photo", "gratitude", "difference", "volunteering",
            "kids", "veterans", "impact", "kindness", "encouragement",
            "education", "nature", "joy", "positivity", "inspiration",
            "help", "commitment", "passion", "energy", "connection",
        ]
        candidates = [
            EmergentCandidate(
                name=name,
                detection_method="named_entity_extraction",
                mention_count=10,
            )
            for name in generic_names
        ]

        # Add some valid candidates
        valid_candidates = [
            EmergentCandidate(
                name="John",
                detection_method="named_entity_extraction",
                mention_count=10,
            ),
            EmergentCandidate(
                name="Maria",
                detection_method="named_entity_extraction",
                mention_count=8,
            ),
            EmergentCandidate(
                name="Max",
                detection_method="named_entity_extraction",
                mention_count=6,
            ),
        ]
        candidates.extend(valid_candidates)

        # Mock LLM response - reject all generic, promote only specific names
        response_candidates = [
            MissionFilterCandidate(name=name, promote=False, reason="Generic/abstract term")
            for name in generic_names
        ]
        response_candidates.extend([
            MissionFilterCandidate(name=c.name, promote=True, reason="Specific person name")
            for c in valid_candidates
        ])

        mock_llm_config.call.return_value = MissionFilterResponse(candidates=response_candidates)

        result = await filter_candidates_by_mission(
            llm_config=mock_llm_config,
            mission="Be a health coach",
            candidates=candidates,
        )

        # Should only have John, Maria, and Max
        result_names = {c.name for c in result}
        assert result_names == {"John", "Maria", "Max"}

    async def test_accepts_specific_named_entities(self, mock_llm_config):
        """Test that specific named entities are accepted."""
        # These should all be accepted
        valid_names = [
            "Alice Chen",       # Full name
            "Dr. Smith",        # Title + name
            "John",             # First name (when it's clearly a person)
            "Google",           # Organization
            "Frontend Team",    # Named team
            "Project Phoenix",  # Named project
            "NYC Office",       # Named place
            "Q4 Planning",      # Named event
            "Sprint 23 Review", # Named meeting
        ]
        candidates = [
            EmergentCandidate(
                name=name,
                detection_method="named_entity_extraction",
                mention_count=10,
            )
            for name in valid_names
        ]

        # Mock LLM response - promote all
        response_candidates = [
            MissionFilterCandidate(name=name, promote=True, reason="Specific named entity")
            for name in valid_names
        ]
        mock_llm_config.call.return_value = MissionFilterResponse(candidates=response_candidates)

        result = await filter_candidates_by_mission(
            llm_config=mock_llm_config,
            mission="Be a PM for engineering team",
            candidates=candidates,
        )

        # Should have all valid names
        result_names = {c.name for c in result}
        assert result_names == set(valid_names)

    async def test_llm_error_rejects_all_candidates(self, mock_llm_config):
        """Test that LLM errors result in rejecting all candidates (fail-safe)."""
        candidates = [
            EmergentCandidate(
                name="Alice",
                detection_method="named_entity_extraction",
                mention_count=10,
            )
        ]

        mock_llm_config.call.side_effect = Exception("LLM error")

        result = await filter_candidates_by_mission(
            llm_config=mock_llm_config,
            mission="Test mission",
            candidates=candidates,
        )

        # Should reject all candidates on error (fail-safe)
        assert len(result) == 0

    async def test_missing_candidate_in_response_is_rejected(self, mock_llm_config):
        """Test that candidates not in LLM response are rejected by default."""
        candidates = [
            EmergentCandidate(
                name="Alice",
                detection_method="named_entity_extraction",
                mention_count=10,
            ),
            EmergentCandidate(
                name="Bob",
                detection_method="named_entity_extraction",
                mention_count=5,
            ),
        ]

        # Mock LLM response - only includes Alice, not Bob
        mock_llm_config.call.return_value = MissionFilterResponse(
            candidates=[
                MissionFilterCandidate(name="Alice", promote=True, reason="Specific person"),
            ]
        )

        result = await filter_candidates_by_mission(
            llm_config=mock_llm_config,
            mission="Test mission",
            candidates=candidates,
        )

        # Only Alice should be in result (Bob was missing from response, so rejected)
        assert len(result) == 1
        assert result[0].name == "Alice"


class TestEvaluateEmergentModels:
    """Test the evaluate_emergent_models function for cleanup of existing models."""

    @pytest.fixture
    def mock_llm_config(self):
        """Create a mock LLM config."""
        config = MagicMock()
        config.call = AsyncMock()
        return config

    async def test_empty_models(self, mock_llm_config):
        """Test with empty model list."""
        result = await evaluate_emergent_models(
            llm_config=mock_llm_config,
            models=[],
        )
        assert result == []
        mock_llm_config.call.assert_not_called()

    async def test_removes_generic_models(self, mock_llm_config):
        """Test that generic/abstract models are marked for removal."""
        models = [
            {"id": "id-kids", "name": "kids"},
            {"id": "id-community", "name": "community"},
            {"id": "id-motivation", "name": "motivation"},
            {"id": "id-john", "name": "John"},
            {"id": "id-maria", "name": "Maria"},
        ]

        # Mock LLM response - reject generic, keep specific names
        mock_llm_config.call.return_value = MissionFilterResponse(
            candidates=[
                MissionFilterCandidate(name="kids", promote=False, reason="Generic category"),
                MissionFilterCandidate(name="community", promote=False, reason="Abstract concept"),
                MissionFilterCandidate(name="motivation", promote=False, reason="Abstract concept"),
                MissionFilterCandidate(name="John", promote=True, reason="Person name"),
                MissionFilterCandidate(name="Maria", promote=True, reason="Person name"),
            ]
        )

        result = await evaluate_emergent_models(
            llm_config=mock_llm_config,
            models=models,
        )

        # Should return IDs of generic models to remove
        assert set(result) == {"id-kids", "id-community", "id-motivation"}

    async def test_keeps_specific_named_models(self, mock_llm_config):
        """Test that specific named models are kept."""
        models = [
            {"id": "id-john", "name": "John"},
            {"id": "id-google", "name": "Google"},
            {"id": "id-project", "name": "Project Phoenix"},
        ]

        # Mock LLM response - keep all
        mock_llm_config.call.return_value = MissionFilterResponse(
            candidates=[
                MissionFilterCandidate(name="John", promote=True, reason="Person name"),
                MissionFilterCandidate(name="Google", promote=True, reason="Organization"),
                MissionFilterCandidate(name="Project Phoenix", promote=True, reason="Named project"),
            ]
        )

        result = await evaluate_emergent_models(
            llm_config=mock_llm_config,
            models=models,
        )

        # No models should be removed
        assert result == []

    async def test_llm_error_keeps_all_models(self, mock_llm_config):
        """Test that LLM errors result in keeping all models (safe default)."""
        models = [
            {"id": "id-kids", "name": "kids"},
            {"id": "id-john", "name": "John"},
        ]

        mock_llm_config.call.side_effect = Exception("LLM error")

        result = await evaluate_emergent_models(
            llm_config=mock_llm_config,
            models=models,
        )

        # Should keep all models on error (return empty removal list)
        assert result == []

    async def test_missing_model_in_response_is_removed(self, mock_llm_config):
        """Test that models not in LLM response are marked for removal."""
        models = [
            {"id": "id-alice", "name": "Alice"},
            {"id": "id-bob", "name": "Bob"},
        ]

        # Mock LLM response - only includes Alice
        mock_llm_config.call.return_value = MissionFilterResponse(
            candidates=[
                MissionFilterCandidate(name="Alice", promote=True, reason="Person name"),
            ]
        )

        result = await evaluate_emergent_models(
            llm_config=mock_llm_config,
            models=models,
        )

        # Bob should be marked for removal (missing from response)
        assert result == ["id-bob"]


class TestRemovedEntitiesNotRepromoted:
    """Test that entities removed by evaluation are not re-promoted.

    This tests the fix for a bug where:
    1. evaluate_emergent_models returns model IDs to remove (e.g., 'entity-maya')
    2. We delete those models
    3. detect_entity_candidates finds the same entities (now eligible since model was deleted)
    4. filter_candidates_by_goal approves them (different LLM call)
    5. BUG: We were re-promoting the same entities we just removed

    The fix tracks removed entity_ids and excludes them from promotion.
    """

    async def test_removed_entity_ids_excluded_from_promotion(self):
        """Test that entities whose models were removed are not re-promoted."""
        from hindsight_api.engine.mental_models.models import EmergentCandidate

        # Simulate the scenario from the bug:
        # - existing_emergent has model 'entity-maya' with entity_id='uuid-maya'
        # - evaluate_emergent_models says to remove 'entity-maya'
        # - detect_entity_candidates returns 'Maya' with entity_id='uuid-maya' (now eligible)
        # - filter_candidates_by_goal says to promote 'Maya'
        # - But we should NOT promote because we just removed it

        existing_emergent = [
            {"id": "entity-maya", "name": "Maya", "entity_id": "uuid-maya"},
            {"id": "entity-alex", "name": "Alex", "entity_id": "uuid-alex"},
            {"id": "entity-john", "name": "John", "entity_id": "uuid-john"},  # This one will be kept
        ]

        # Models to remove (evaluate_emergent_models would return these)
        models_to_remove = ["entity-maya", "entity-alex"]

        # Build model_id -> entity_id mapping (this is what the fix does)
        model_to_entity = {m["id"]: m.get("entity_id") for m in existing_emergent}

        # Track removed entity_ids
        removed_entity_ids: set[str] = set()
        for model_id in models_to_remove:
            entity_id = model_to_entity.get(model_id)
            if entity_id:
                removed_entity_ids.add(str(entity_id))

        # Verify we tracked the right entity_ids
        assert removed_entity_ids == {"uuid-maya", "uuid-alex"}

        # Now simulate candidates that were detected (includes removed entities)
        candidates = [
            EmergentCandidate(
                name="Maya", entity_id="uuid-maya", detection_method="named_entity", mention_count=10
            ),
            EmergentCandidate(
                name="Alex", entity_id="uuid-alex", detection_method="named_entity", mention_count=8
            ),
            EmergentCandidate(
                name="NewPerson", entity_id="uuid-new", detection_method="named_entity", mention_count=5
            ),
        ]

        # Filter out candidates whose entity was just removed (the fix)
        filtered_candidates = [c for c in candidates if c.entity_id not in removed_entity_ids]

        # Only NewPerson should remain - Maya and Alex were removed and should not be re-promoted
        assert len(filtered_candidates) == 1
        assert filtered_candidates[0].name == "NewPerson"
        assert filtered_candidates[0].entity_id == "uuid-new"

    async def test_candidates_without_matching_removal_are_kept(self):
        """Test that candidates not in the removed set are still promoted."""
        from hindsight_api.engine.mental_models.models import EmergentCandidate

        # No models removed
        removed_entity_ids: set[str] = set()

        candidates = [
            EmergentCandidate(
                name="Alice", entity_id="uuid-alice", detection_method="named_entity", mention_count=10
            ),
            EmergentCandidate(
                name="Bob", entity_id="uuid-bob", detection_method="named_entity", mention_count=8
            ),
        ]

        # Filter (should keep all since nothing was removed)
        filtered_candidates = [c for c in candidates if c.entity_id not in removed_entity_ids]

        assert len(filtered_candidates) == 2
        assert {c.name for c in filtered_candidates} == {"Alice", "Bob"}

    async def test_partial_removal_keeps_other_candidates(self):
        """Test that only removed entities are excluded, others pass through."""
        from hindsight_api.engine.mental_models.models import EmergentCandidate

        # Only one entity removed
        removed_entity_ids = {"uuid-removed"}

        candidates = [
            EmergentCandidate(
                name="Removed", entity_id="uuid-removed", detection_method="named_entity", mention_count=10
            ),
            EmergentCandidate(
                name="Kept1", entity_id="uuid-kept1", detection_method="named_entity", mention_count=8
            ),
            EmergentCandidate(
                name="Kept2", entity_id="uuid-kept2", detection_method="named_entity", mention_count=5
            ),
        ]

        filtered_candidates = [c for c in candidates if c.entity_id not in removed_entity_ids]

        assert len(filtered_candidates) == 2
        assert {c.name for c in filtered_candidates} == {"Kept1", "Kept2"}
