"""
Real-LLM check that the model actually populates `entities` (#2749).

The bug was behavioural, not structural: the prompt's few-shot examples taught a
flat string array while the schema demanded objects, so models that follow the
prompt literally returned strings or omitted the field entirely and every entity
was dropped. MockLLM cannot reproduce that — it takes a real model reading the
real prompt, so this runs the actual extraction pipeline.
"""

from datetime import datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text

pytestmark = pytest.mark.hs_llm_core


@pytest.mark.asyncio
async def test_extraction_populates_entities():
    text = (
        "Alice has 5 years of Kubernetes experience and holds a CKA certification. "
        "She's been leading the infrastructure team at Acme Corp since March."
    )

    facts, _, _ = await extract_facts_from_text(
        text=text,
        event_date=datetime(2025, 3, 28),
        llm_config=LLMConfig.from_env(),
        agent_name="test-agent",
        context="professional background",
        config=_get_raw_config(),
    )

    assert len(facts) > 0, "Should extract at least one fact"

    all_entities = [e for f in facts for e in (f.entities or [])]
    # The regression: this list was empty because the model returned strings for a
    # field the schema declared as objects, and the entities were dropped.
    assert all_entities, f"No entities extracted. Facts: {[(f.fact, f.entities) for f in facts]}"
    assert all(isinstance(e, str) and e.strip() for e in all_entities), (
        f"Entities must be non-empty strings, got {all_entities}"
    )
    # "Alice" is named unambiguously and repeatedly; any model that populates the
    # field at all should surface her. Matched case-insensitively / as a substring
    # since models legitimately vary on casing and qualifiers ("Alice (team lead)").
    assert any("alice" in e.lower() for e in all_entities), f"Expected 'Alice' among entities, got {all_entities}"
