"""Real-model resolution and leakage check for the counterbalanced #2551 fixture."""

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["A", "B"])
async def test_previous_options_resolve_target_without_prior_only_facts(scenario: str) -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures/retain_context_options.json").read_text())[scenario]
    config = replace(
        _get_raw_config(),
        retain_chunk_size=800,
        retain_structured_chunk_size=None,
        retain_context_chars=800,
        retain_extraction_mode="concise",
        retain_extract_causal_links=False,
        retain_mission="Extract concrete decisions and commitments. Distinguish proposals from approved decisions.",
    )
    facts, chunks, _ = await extract_facts_from_text(
        text=json.dumps([fixture["prior"], fixture["target"]]),
        event_date=datetime(2026, 9, 24, 9, 1, tzinfo=timezone.utc),
        llm_config=LLMConfig.from_env(),
        config=config,
        context="Mira is the human decision maker. Rowan is the assistant proposing alternatives. Each turn names its speaker.",
    )
    assert len(chunks) == 2
    target_facts = facts[chunks[0][1] :]
    await assert_meets_criteria(
        response="\n".join(f"- [{f.fact_type}] {f.fact}" for f in target_facts),
        criteria=(
            f"Mira approved this plan: {fixture['expected']} The approval links the correct day AND owner. "
            f"The alternative ({fixture['rejected']}) is not reported as approved. "
            "Mira's commitment to circulate the decision after lunch is preserved. "
            "There is no standalone fact asserting that the Cedar room is booked on Tuesday, "
            "and no standalone enumeration of the earlier proposals. A fact recording Mira's "
            "instruction to keep the room booking separate is allowed."
        ),
        context="Only facts from Mira's target turn are being judged. Prior source: " + json.dumps(fixture["prior"]),
    )
