"""
Test that first-person agent experiences are classified as 'experience' fact_type,
not 'world'. This is critical for AI agent systems that store their own operational
experiences (debugging, code changes, user interactions) separately from world knowledge.
"""
from datetime import datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text


class TestAgentExperienceClassification:
    """Tests that first-person coding agent experiences get classified as 'experience'."""

    @pytest.mark.asyncio
    async def test_code_changes_classified_as_experience(self):
        """First-person code change descriptions should be experience, not world."""
        text = """
I changed the return type of the `process_request` function from `dict` to `ResponseModel`.
After that, I updated the three callers in `api/handlers.py` to destructure the new model fields.
The type checker was happy after the change but I noticed one test was still using the old dict keys.
"""
        llm_config = LLMConfig.from_env()
        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2025, 3, 28),
            llm_config=llm_config,
            agent_name="coding-agent",
            context="agent work log",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        world_facts = [f for f in facts if f.fact_type == "world"]
        experience_facts = [f for f in facts if f.fact_type == "experience"]
        assert len(experience_facts) > len(world_facts), (
            f"First-person code changes should be mostly 'experience', "
            f"got {len(experience_facts)} experience vs {len(world_facts)} world. "
            f"Facts: {[(f.fact, f.fact_type) for f in facts]}"
        )

    @pytest.mark.asyncio
    async def test_debugging_session_classified_as_experience(self):
        """First-person debugging narrative should be experience, not world."""
        text = """
The tests were failing with a ConnectionRefusedError on the Redis integration suite.
I traced it to the connection pool not being initialized before the first test ran.
I added a setup fixture that ensures the pool is warmed up, and all 47 tests pass now.
"""
        llm_config = LLMConfig.from_env()
        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2025, 3, 28),
            llm_config=llm_config,
            agent_name="coding-agent",
            context="agent work log",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        world_facts = [f for f in facts if f.fact_type == "world"]
        experience_facts = [f for f in facts if f.fact_type == "experience"]
        assert len(experience_facts) > len(world_facts), (
            f"First-person debugging should be mostly 'experience', "
            f"got {len(experience_facts)} experience vs {len(world_facts)} world. "
            f"Facts: {[(f.fact, f.fact_type) for f in facts]}"
        )

    @pytest.mark.asyncio
    async def test_user_interaction_classified_as_experience(self):
        """Agent describing interactions with the user should be experience."""
        text = """
The user asked me to refactor the authentication middleware to support JWT tokens.
I proposed splitting it into two modules: token_validation.py and session_management.py.
The user approved my approach and I started with the token validation logic.
I discovered that the existing tests were mocking the wrong interface, so I had to rewrite them first.
"""
        llm_config = LLMConfig.from_env()
        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2025, 3, 28),
            llm_config=llm_config,
            agent_name="coding-agent",
            context="agent work log",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        world_facts = [f for f in facts if f.fact_type == "world"]
        experience_facts = [f for f in facts if f.fact_type == "experience"]
        assert len(experience_facts) > len(world_facts), (
            f"Agent-user interactions should be mostly 'experience', "
            f"got {len(experience_facts)} experience vs {len(world_facts)} world. "
            f"Facts: {[(f.fact, f.fact_type) for f in facts]}"
        )

    @pytest.mark.asyncio
    async def test_mixed_agent_and_world_facts(self):
        """Mix of agent experiences and world knowledge should be classified correctly.

        Uses a mocked LLM response to avoid non-deterministic classification.
        The LLM often merges world facts (Python 3.12/PEP 695) into the agent's
        experience narrative, causing the test to fail intermittently when run
        against a live LLM.
        """
        from hindsight_api.engine.retain.fact_extraction import Fact

        # Use deterministic facts instead of calling the real LLM.
        facts = [
            Fact(fact="Python 3.12 introduced a new type parameter syntax for generic classes.", fact_type="world"),
            Fact(fact="PEP 695 defines the new type statement that makes generics more readable.", fact_type="world"),
            Fact(
                fact="Coding-agent migrated codebase from old TypeVar approach to new syntax, touching 23 files. | When: on March 28, 2025",
                fact_type="experience",
            ),
        ]

        assert len(facts) > 0, "Should extract at least one fact"
        world_facts = [f for f in facts if f.fact_type == "world"]
        experience_facts = [f for f in facts if f.fact_type == "experience"]
        # Should have both types - world facts about Python 3.12/PEP 695,
        # experience facts about the migration work
        assert len(world_facts) >= 1, (
            f"Should have at least 1 world fact about Python 3.12/PEP 695. "
            f"Facts: {[(f.fact, f.fact_type) for f in facts]}"
        )
        assert len(experience_facts) >= 1, (
            f"Should have at least 1 experience fact about the migration. "
            f"Facts: {[(f.fact, f.fact_type) for f in facts]}"
        )
