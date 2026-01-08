"""Tests for emergent entity filtering."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hindsight_api.engine.mental_models.emergent import (
    build_goal_filter_prompt,
    evaluate_emergent_models,
    filter_candidates_by_goal,
    GoalFilterResponse,
    GoalFilterCandidate,
)
from hindsight_api.engine.mental_models.models import EmergentCandidate, MentalModelType


class TestBuildGoalFilterPrompt:
    """Test prompt building for goal filtering."""

    def test_prompt_contains_goal(self):
        """Test that prompt includes the goal."""
        candidates = [
            EmergentCandidate(
                name="Alice",
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=10,
            )
        ]
        prompt = build_goal_filter_prompt("Be a PM for engineering team", candidates)
        assert "Be a PM for engineering team" in prompt

    def test_prompt_contains_candidates(self):
        """Test that prompt includes all candidates."""
        candidates = [
            EmergentCandidate(
                name="Alice Chen",
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=10,
            ),
            EmergentCandidate(
                name="Project Phoenix",
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=5,
            ),
        ]
        prompt = build_goal_filter_prompt("Track projects", candidates)
        assert "Alice Chen" in prompt
        assert "Project Phoenix" in prompt

    def test_prompt_contains_rejection_guidance(self):
        """Test that prompt contains guidance to reject generic entities."""
        candidates = [
            EmergentCandidate(
                name="test",
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=1,
            )
        ]
        prompt = build_goal_filter_prompt("Test goal", candidates)

        # Should contain rejection guidance for generic terms
        assert "promote=false" in prompt
        assert "kids" in prompt  # Example of generic term to reject
        assert "community" in prompt  # Example of abstract concept to reject
        assert "motivation" in prompt  # Example of abstract concept to reject


class TestFilterCandidatesByGoal:
    """Test the filter_candidates_by_goal function."""

    @pytest.fixture
    def mock_llm_config(self):
        """Create a mock LLM config."""
        config = MagicMock()
        config.call = AsyncMock()
        return config

    async def test_empty_candidates(self, mock_llm_config):
        """Test with empty candidate list."""
        result = await filter_candidates_by_goal(
            llm_config=mock_llm_config,
            goal="Test goal",
            candidates=[],
        )
        assert result == []
        mock_llm_config.call.assert_not_called()

    async def test_no_goal_keeps_all(self, mock_llm_config):
        """Test that no goal keeps all candidates (skips filtering)."""
        candidates = [
            EmergentCandidate(
                name="Alice",
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=10,
            )
        ]
        result = await filter_candidates_by_goal(
            llm_config=mock_llm_config,
            goal="",  # Empty goal
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
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=10,
            ),
            EmergentCandidate(
                name="community",
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=5,
            ),
        ]

        # Mock LLM response - Alice is promoted, community is not
        mock_llm_config.call.return_value = GoalFilterResponse(
            candidates=[
                GoalFilterCandidate(name="Alice Chen", promote=True, reason="Specific person"),
                GoalFilterCandidate(name="community", promote=False, reason="Generic abstract concept"),
            ]
        )

        result = await filter_candidates_by_goal(
            llm_config=mock_llm_config,
            goal="Be a PM for engineering team",
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
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=10,
            )
            for name in generic_names
        ]

        # Add some valid candidates
        valid_candidates = [
            EmergentCandidate(
                name="John",
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=10,
            ),
            EmergentCandidate(
                name="Maria",
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=8,
            ),
            EmergentCandidate(
                name="Max",
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=6,
            ),
        ]
        candidates.extend(valid_candidates)

        # Mock LLM response - reject all generic, promote only specific names
        response_candidates = [
            GoalFilterCandidate(name=name, promote=False, reason="Generic/abstract term")
            for name in generic_names
        ]
        response_candidates.extend([
            GoalFilterCandidate(name=c.name, promote=True, reason="Specific person name")
            for c in valid_candidates
        ])

        mock_llm_config.call.return_value = GoalFilterResponse(candidates=response_candidates)

        result = await filter_candidates_by_goal(
            llm_config=mock_llm_config,
            goal="Be a health coach",
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
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=10,
            )
            for name in valid_names
        ]

        # Mock LLM response - promote all
        response_candidates = [
            GoalFilterCandidate(name=name, promote=True, reason="Specific named entity")
            for name in valid_names
        ]
        mock_llm_config.call.return_value = GoalFilterResponse(candidates=response_candidates)

        result = await filter_candidates_by_goal(
            llm_config=mock_llm_config,
            goal="Be a PM for engineering team",
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
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=10,
            )
        ]

        mock_llm_config.call.side_effect = Exception("LLM error")

        result = await filter_candidates_by_goal(
            llm_config=mock_llm_config,
            goal="Test goal",
            candidates=candidates,
        )

        # Should reject all candidates on error (fail-safe)
        assert len(result) == 0

    async def test_missing_candidate_in_response_is_rejected(self, mock_llm_config):
        """Test that candidates not in LLM response are rejected by default."""
        candidates = [
            EmergentCandidate(
                name="Alice",
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=10,
            ),
            EmergentCandidate(
                name="Bob",
                type=MentalModelType.ENTITY,
                detection_method="named_entity_extraction",
                mention_count=5,
            ),
        ]

        # Mock LLM response - only includes Alice, not Bob
        mock_llm_config.call.return_value = GoalFilterResponse(
            candidates=[
                GoalFilterCandidate(name="Alice", promote=True, reason="Specific person"),
            ]
        )

        result = await filter_candidates_by_goal(
            llm_config=mock_llm_config,
            goal="Test goal",
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
        mock_llm_config.call.return_value = GoalFilterResponse(
            candidates=[
                GoalFilterCandidate(name="kids", promote=False, reason="Generic category"),
                GoalFilterCandidate(name="community", promote=False, reason="Abstract concept"),
                GoalFilterCandidate(name="motivation", promote=False, reason="Abstract concept"),
                GoalFilterCandidate(name="John", promote=True, reason="Person name"),
                GoalFilterCandidate(name="Maria", promote=True, reason="Person name"),
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
        mock_llm_config.call.return_value = GoalFilterResponse(
            candidates=[
                GoalFilterCandidate(name="John", promote=True, reason="Person name"),
                GoalFilterCandidate(name="Google", promote=True, reason="Organization"),
                GoalFilterCandidate(name="Project Phoenix", promote=True, reason="Named project"),
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
        mock_llm_config.call.return_value = GoalFilterResponse(
            candidates=[
                GoalFilterCandidate(name="Alice", promote=True, reason="Person name"),
            ]
        )

        result = await evaluate_emergent_models(
            llm_config=mock_llm_config,
            models=models,
        )

        # Bob should be marked for removal (missing from response)
        assert result == ["id-bob"]
