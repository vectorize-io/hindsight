"""Context overrides the narrator for speaker attribution (issue #1680 / #1095).

The Narrator line primes "first-person → the agent". When the Context says the
first-person speaker is actually the *user* (a transcript / customer log), the
prompt instructs the model to prefer the Context: such statements are facts
ABOUT the user ("world"), not the agent's own experiences.

Real-LLM test — the narrator/context tension cannot be simulated with MockLLM.
"""

from datetime import datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


class TestContextOverridesNarrator:
    """When Context names the user as the first-person speaker, first-person facts
    are attributed to the user, not the narrator agent."""

    @pytest.mark.asyncio
    async def test_user_first_person_attributed_to_user_not_agent(self):
        # Narrator is a named agent (so the Narrator line IS injected), but the
        # Context makes clear the first-person speaker is the customer, not the agent.
        agent_name = "SupportBot"
        context = (
            "Transcript of a customer named Maria describing her situation to the "
            "support agent. Maria is the one speaking in the first person here; the "
            "support agent is not speaking."
        )
        text = (
            "I just bought a Tesla Model 3 last week. I live in Berlin and I commute "
            "about 40 miles a day to my job at Acme Corp."
        )

        llm_config = LLMConfig.from_env()
        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 6, 1),
            llm_config=llm_config,
            agent_name=agent_name,
            context=context,
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"

        # Use the LLM judge for correctness — both the attribution and the
        # world/experience classification are non-deterministic, so a direct
        # `fact_type == "world"` assert would flake (see test_fact_extraction_
        # agent_experience.py for the same judge-the-classification pattern).
        # The summary carries each fact's type so the judge can evaluate it.
        facts_summary = "\n".join(f"- [{f.fact_type}] {f.fact}" for f in facts)
        await assert_meets_criteria(
            response=facts_summary,
            criteria=(
                "Buying the Tesla, living in Berlin, and the Acme Corp commute are attributed to "
                "Maria / the customer / the user — NOT to the support agent or 'SupportBot' (the "
                "agent did not buy a Tesla, does not live in Berlin, does not commute to Acme Corp). "
                "Because the Context says the user is the one speaking, these are facts ABOUT the "
                "user and are classified 'world', not the agent's own 'experience'."
            ),
            context=(
                "Maria (a customer) said in first person that she bought a Tesla Model 3, lives in "
                "Berlin, and commutes ~40 miles/day to Acme Corp. The narrator agent is 'SupportBot', "
                "who was NOT speaking. Each fact is shown as '- [fact_type] text'."
            ),
            msg=f"User first-person statements must be 'world' attributed to the user, not the agent. Facts: {[(f.fact, f.fact_type) for f in facts]}",
        )
