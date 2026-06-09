"""
Test suite for causal relations extraction and validation.

Tests that:
1. Causal relations only reference previous facts (target_index < current fact index)
2. Invalid causal relation indices are rejected
3. The new per-fact causal relations schema works correctly
"""

from datetime import datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text

pytestmark = pytest.mark.hs_llm_core


class TestCausalRelationsValidation:
    """Tests for causal relations index validation."""

    @pytest.mark.asyncio
    async def test_causal_relations_only_reference_previous_facts(self):
        """
        Test that causal relations can only reference facts that appear before them.

        This test verifies the new schema that prevents hallucination of invalid
        fact indices by constraining target_index to be less than the current fact's index.
        """
        # Text with clear causal chain
        text = """
        I lost my job in January due to company layoffs.
        Because I lost my job, I couldn't pay my rent.
        Since I couldn't afford rent, I had to move to a cheaper apartment.
        After moving, I started looking for a new job.
        """

        context = "Personal life update"
        llm_config = LLMConfig.from_env()
        event_date = datetime(2024, 3, 15)

        facts, _, usage = await extract_facts_from_text(
            text=text,
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"

        # Verify all causal relations reference valid previous facts
        for i, fact in enumerate(facts):
            if fact.causal_relations:
                for rel in fact.causal_relations:
                    assert rel.target_fact_index < i, (
                        f"Fact {i} has causal relation to fact {rel.target_fact_index}, "
                        f"but target_index must be < current index ({i})"
                    )
                    assert rel.target_fact_index >= 0, (
                        f"Fact {i} has negative causal relation index: {rel.target_fact_index}"
                    )
                    assert rel.relation_type in ["caused_by", "enabled_by", "prevented_by"], (
                        f"Invalid relation_type: {rel.relation_type}"
                    )

    @pytest.mark.asyncio
    async def test_first_fact_has_no_causal_relations(self):
        """
        Test that the first fact (index 0) cannot have causal relations.

        Since causal relations can only reference previous facts,
        and there are no facts before index 0, the first fact should
        have no causal relations.
        """
        text = """
        The user started a new machine learning project.
        The project requires learning TensorFlow.
        Learning TensorFlow is challenging but rewarding.
        """

        context = "Project update"
        llm_config = LLMConfig.from_env()
        event_date = datetime(2024, 6, 1)

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"

        # First fact should have no causal relations (nothing to reference)
        if facts[0].causal_relations:
            # If there are causal relations on the first fact, they should be empty
            # or the validation should have filtered them out
            for rel in facts[0].causal_relations:
                # This should never happen due to validation
                assert False, (
                    f"First fact should not have causal relations, but found: target_index={rel.target_fact_index}"
                )

    @pytest.mark.asyncio
    async def test_causal_chain_extraction(self):
        """
        Test that a clear causal chain is extracted with valid relations.
        """
        text = """
        Emily got promoted to senior engineer last month.
        Because of her promotion, she received a significant salary increase.
        With the extra money, she decided to buy a new car.
        """

        context = "Personal achievement story"
        llm_config = LLMConfig.from_env()
        event_date = datetime(2024, 7, 15)

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract facts about the causal chain"

        # Collect all causal relations
        all_relations = []
        for i, fact in enumerate(facts):
            if fact.causal_relations:
                for rel in fact.causal_relations:
                    all_relations.append(
                        {
                            "from_fact": i,
                            "to_fact": rel.target_fact_index,
                            "type": rel.relation_type,
                        }
                    )

        # If causal relations were extracted, verify they form a valid chain
        if all_relations:
            for rel in all_relations:
                assert rel["to_fact"] < rel["from_fact"], (
                    f"Causal relation from fact {rel['from_fact']} to fact {rel['to_fact']} "
                    f"is invalid (target must be < source)"
                )

    @pytest.mark.asyncio
    async def test_token_efficiency_with_causal_relations(self):
        """
        Test that causal relations don't cause excessive output tokens.

        This test verifies that the new schema (per-fact causal relations
        with index constraints) doesn't waste tokens on invalid relations.
        """
        text = """
        The company announced budget cuts in Q1.
        Due to the budget cuts, the marketing team was reduced.
        The reduced team meant fewer campaigns could be run.
        With fewer campaigns, lead generation dropped.
        Lower leads resulted in decreased sales.
        """

        context = "Business impact analysis"
        llm_config = LLMConfig.from_env()
        event_date = datetime(2024, 4, 1)

        facts, _, usage = await extract_facts_from_text(
            text=text,
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract facts"

        # Calculate output/input ratio
        if usage.input_tokens > 0:
            ratio = usage.output_tokens / usage.input_tokens
            # The ratio should be reasonable (< 5x) with the new schema
            # Previously it could be 7-10x due to hallucinated indices
            assert ratio < 6, (
                f"Output/input token ratio {ratio:.2f}x is too high. "
                f"Input: {usage.input_tokens}, Output: {usage.output_tokens}"
            )

    @pytest.mark.asyncio
    async def test_relation_types_are_backward_looking(self):
        """
        Test that all relation types describe how the current fact
        relates to a previous fact (caused_by, enabled_by, prevented_by).
        """
        text = """
        Alice learned Python programming.
        Because she knew Python, she got a job as a data scientist.
        Her data science skills enabled her to lead the analytics team.
        """

        context = "Career progression"
        llm_config = LLMConfig.from_env()
        event_date = datetime(2024, 5, 1)

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        # Verify relation types are all backward-looking
        valid_types = {"caused_by", "enabled_by", "prevented_by"}

        for i, fact in enumerate(facts):
            if fact.causal_relations:
                for rel in fact.causal_relations:
                    assert rel.relation_type in valid_types, (
                        f"Invalid relation_type '{rel.relation_type}'. Must be one of: {valid_types}"
                    )
