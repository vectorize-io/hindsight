"""
Test suite for causal relationship extraction.

Tests that the fact extraction system correctly identifies and validates
causal relationships between facts, with valid indices.
"""

from datetime import datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text


class TestCausalRelationships:
    """Tests for causal relationship extraction and validation."""

    @pytest.mark.asyncio
    async def test_causal_chain_extraction(self):
        """
        Test that a clear causal chain is extracted with valid relationships.

        Story: Lost job -> couldn't pay rent -> had to move -> found new apartment

        This is a 4-fact causal chain where each fact causes the next.
        The extracted causal relations should have valid indices (0-3).
        """
        text = """
I lost my job at the tech company in January because of layoffs.
Because I lost my job, I couldn't pay my rent anymore.
Since I couldn't afford rent, I had to move out of my apartment.
After searching for weeks, I finally found a cheaper apartment in Brooklyn.
"""

        context = "Personal story about housing change"
        llm_config = LLMConfig.for_memory()

        facts, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 3, 15),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser"
        )

        assert len(facts) >= 3, f"Should extract at least 3 facts from the causal chain. Got {len(facts)}"

        # Collect all causal relations from all facts
        all_causal_relations = []
        for i, fact in enumerate(facts):
            if fact.causal_relations:
                for rel in fact.causal_relations:
                    all_causal_relations.append({
                        "from_fact_index": i,
                        "to_fact_index": rel.target_fact_index,
                        "relation_type": rel.relation_type,
                        "strength": rel.strength,
                        "from_fact_text": fact.fact[:50],
                    })

        # Verify that ALL causal relation indices are valid
        num_facts = len(facts)
        invalid_relations = []
        for rel in all_causal_relations:
            if rel["to_fact_index"] < 0 or rel["to_fact_index"] >= num_facts:
                invalid_relations.append(rel)

        assert len(invalid_relations) == 0, (
            f"Found {len(invalid_relations)} causal relations with invalid indices! "
            f"Valid range is 0-{num_facts - 1}. "
            f"Invalid relations: {invalid_relations}"
        )

        # Should have at least some causal relations extracted
        assert len(all_causal_relations) >= 2, (
            f"Should extract at least 2 causal relationships from this clear chain. "
            f"Got {len(all_causal_relations)}: {all_causal_relations}"
        )

        # Verify relation types are valid
        valid_types = {"causes", "caused_by", "enables", "prevents"}
        for rel in all_causal_relations:
            assert rel["relation_type"] in valid_types, (
                f"Invalid relation_type '{rel['relation_type']}'. Must be one of {valid_types}"
            )

    @pytest.mark.asyncio
    async def test_complex_causal_web(self):
        """
        Test a more complex scenario with multiple interconnected causes.

        This tests the LLM's ability to identify multiple causal links and
        ensure all referenced indices exist.
        """
        text = """
The heavy rain caused flooding in the basement.
The flooding damaged the electrical system.
Because of the electrical damage, we had to call an electrician.
The electrician found that the wiring was old and needed replacement.
We decided to renovate the entire basement while fixing the wiring.
The renovation took three months and cost $15,000.
"""

        context = "Home repair story"
        llm_config = LLMConfig.for_memory()

        facts, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 6, 1),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser"
        )

        assert len(facts) >= 4, f"Should extract at least 4 facts. Got {len(facts)}"

        # Validate all causal relation indices
        num_facts = len(facts)
        for i, fact in enumerate(facts):
            if fact.causal_relations:
                for rel in fact.causal_relations:
                    assert 0 <= rel.target_fact_index < num_facts, (
                        f"Fact {i} has causal relation to invalid index {rel.target_fact_index}. "
                        f"Valid range is 0-{num_facts - 1}. "
                        f"Fact text: {fact.fact[:80]}..."
                    )

    @pytest.mark.asyncio
    async def test_no_self_referencing_causal_relations(self):
        """
        Test that facts don't have causal relations pointing to themselves.
        """
        text = """
I started learning Python because I wanted to automate my work tasks.
Learning Python led me to discover machine learning.
Machine learning fascinated me so much that I changed my career to data science.
"""

        context = "Career change story"
        llm_config = LLMConfig.for_memory()

        facts, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 1, 1),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser"
        )

        # Check no fact references itself
        for i, fact in enumerate(facts):
            if fact.causal_relations:
                for rel in fact.causal_relations:
                    assert rel.target_fact_index != i, (
                        f"Fact {i} has a self-referencing causal relation! "
                        f"Fact text: {fact.fact}"
                    )

    @pytest.mark.asyncio
    async def test_bidirectional_causal_relationships(self):
        """
        Test that bidirectional causal relationships (causes and caused_by)
        are handled correctly.
        """
        text = """
My promotion at work caused me to move to New York.
Moving to New York was caused by my promotion at work.
The new role enabled me to lead a team of engineers.
"""

        context = "Work promotion story"
        llm_config = LLMConfig.for_memory()

        facts, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 2, 15),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser"
        )

        num_facts = len(facts)

        # Validate all indices
        for i, fact in enumerate(facts):
            if fact.causal_relations:
                for rel in fact.causal_relations:
                    assert 0 <= rel.target_fact_index < num_facts, (
                        f"Invalid target_fact_index {rel.target_fact_index} in fact {i}. "
                        f"Valid range: 0-{num_facts - 1}"
                    )

    @pytest.mark.asyncio
    async def test_causal_relation_strength_values(self):
        """
        Test that causal relation strength values are within valid range [0.0, 1.0].
        """
        text = """
The stock market crash directly caused the company to lay off employees.
The layoffs indirectly led to reduced consumer spending in the area.
Reduced spending somewhat affected local businesses.
"""

        context = "Economic impact story"
        llm_config = LLMConfig.for_memory()

        facts, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 4, 1),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser"
        )

        for i, fact in enumerate(facts):
            if fact.causal_relations:
                for rel in fact.causal_relations:
                    assert 0.0 <= rel.strength <= 1.0, (
                        f"Causal relation strength {rel.strength} is outside valid range [0.0, 1.0]. "
                        f"Fact {i}: {fact.fact[:50]}..."
                    )
