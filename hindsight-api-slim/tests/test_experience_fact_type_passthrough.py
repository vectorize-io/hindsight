"""
Regression test for experience fact_type preservation.

The LLM extraction layer normalizes raw "assistant" → "experience" early in parsing.
The subsequent conversion to ExtractedFactType must pass through the already-normalized
fact_type rather than re-checking for "assistant" (which would remap experience → world).

See: https://github.com/vectorize-io/hindsight/pull/839
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain.fact_extraction import (
    Fact,
    RetainContent,
    extract_facts_from_contents,
    extract_facts_from_contents_batch_api,
)


@pytest.mark.asyncio
async def test_extract_facts_preserves_experience_type():
    """
    When extract_facts_from_text returns a Fact with fact_type="experience",
    extract_facts_from_contents must preserve it (not remap to "world").
    """
    contents = [
        RetainContent(
            content="I fixed the failing tests after discovering they mocked the wrong interface.",
            event_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
            context="assistant work log",
        )
    ]
    extracted_fact = Fact(
        fact="Fixed the failing tests after discovering they mocked the wrong interface.",
        fact_type="experience",
    )

    with patch(
        "hindsight_api.engine.retain.fact_extraction.extract_facts_from_text",
        new=AsyncMock(return_value=([extracted_fact], [(contents[0].content, 1)], TokenUsage())),
    ):
        facts, _chunks, _usage = await extract_facts_from_contents(
            contents=contents,
            llm_config=None,
            agent_name="TestAgent",
            config=_get_raw_config(),
        )

    assert len(facts) == 1
    assert facts[0].fact_type == "experience", (
        f"Expected 'experience' but got '{facts[0].fact_type}' — "
        f"the conversion layer is remapping the already-normalized fact_type"
    )
