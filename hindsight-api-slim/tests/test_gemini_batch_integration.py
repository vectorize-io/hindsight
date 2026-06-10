"""Live integration test for the Gemini Batch API provider.

This makes REAL calls to the Gemini Batch API and runs the full retain fact
extraction pipeline end-to-end (translate -> upload JSONL -> batches.create ->
poll -> download -> normalize -> parse facts). It is the only test that
validates the one assumption the unit tests (which use a fake genai client)
cannot: that Gemini's real batch output JSONL shape matches what
``_normalize_output_line`` produces and what ``fact_extraction`` consumes. If
the shape is wrong, this returns zero facts.

This is an explicit opt-in test. It is gated on a dedicated flag rather than
just "a Gemini API key exists" because CI always has a Gemini key (Gemini is the
LLM-as-judge / core-LLM provider) — keying off the API key alone would let this
slow batch job run in the standard CI shard and blow the 300s pytest timeout. To
run it:

    export HINDSIGHT_API_GEMINI_BATCH_LIVE_TEST=1
    export GEMINI_API_KEY=...                          # or HINDSIGHT_API_GEMINI_API_KEY
    # optional: override the model
    export HINDSIGHT_API_GEMINI_TEST_MODEL=gemini-2.5-flash
    uv run pytest tests/test_gemini_batch_integration.py -v -s

It is slow (typically minutes, but Gemini's batch queue can take far longer; the
SLA is up to 24h) and costs money, so it never runs in CI. No database is
required — it calls the extraction function directly with ``pool=None``.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from dotenv import load_dotenv

from hindsight_api.config import HindsightConfig, clear_config_cache
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.retain.fact_extraction import (
    RetainContent,
    extract_facts_from_contents_batch_api,
)

logger = logging.getLogger(__name__)

load_dotenv()

_DEFAULT_TEST_MODEL = "gemini-2.5-flash"


@dataclass
class GeminiTestEnv:
    api_key: str
    model: str


@pytest.fixture
def gemini_env() -> GeminiTestEnv:
    # Opt-in flag, NOT just key presence: CI always has GEMINI_API_KEY (judge /
    # core-LLM provider), so gating on the key alone runs this slow batch job in
    # the standard CI shard and times out. Require an explicit flag CI never sets.
    if os.getenv("HINDSIGHT_API_GEMINI_BATCH_LIVE_TEST", "").lower() not in ("1", "true", "yes"):
        pytest.skip("Set HINDSIGHT_API_GEMINI_BATCH_LIVE_TEST=1 (and GEMINI_API_KEY) to run the live Gemini batch test")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("HINDSIGHT_API_GEMINI_API_KEY")
    if not api_key:
        pytest.skip("Set GEMINI_API_KEY (or HINDSIGHT_API_GEMINI_API_KEY) to run the live Gemini batch test")

    clear_config_cache()

    return GeminiTestEnv(
        api_key=api_key,
        model=os.getenv("HINDSIGHT_API_GEMINI_TEST_MODEL", _DEFAULT_TEST_MODEL),
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_gemini_batch_end_to_end(gemini_env):
    config = HindsightConfig.from_env()
    config.retain_batch_enabled = True
    config.retain_batch_poll_interval_seconds = 30
    config.retain_chunk_size = 4000
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = False

    llm_config = LLMProvider(
        provider="gemini",
        api_key=gemini_env.api_key,
        base_url="",
        model=gemini_env.model,
        reasoning_effort="low",
    )
    assert await llm_config._provider_impl.supports_batch_api() is True

    contents = [
        RetainContent(
            content=(
                "Alice is a senior software engineer at TechCorp. She specializes in "
                "distributed systems and graduated from MIT in 2015."
            ),
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            context="team member profile",
        )
    ]

    logger.info("Submitting a real Gemini batch (this can take several minutes)...")
    facts, chunks, usage = await extract_facts_from_contents_batch_api(
        contents=contents,
        llm_config=llm_config,
        agent_name="test_agent",
        config=config,
        pool=None,
        operation_id=None,
        schema=None,
    )

    # The end-to-end proof: if the real output shape doesn't match the normalizer,
    # the consumer extracts nothing and this is empty.
    assert len(facts) > 0, (
        "Gemini batch returned no facts. The live output JSONL shape likely "
        "differs from what _normalize_output_line produces — inspect a raw output "
        "line and adjust the normalizer."
    )
    assert any("Alice" in fact.fact_text for fact in facts)
    # Token usage must be threaded from Gemini's usageMetadata into the batch
    # result body — otherwise the consumer reports zero (the bug this guards).
    assert usage.total_tokens > 0, "Gemini batch reported zero token usage — usageMetadata translation is broken"
    assert usage.input_tokens > 0 and usage.output_tokens > 0
    logger.info(
        f"Extracted {len(facts)} facts; usage in={usage.input_tokens} out={usage.output_tokens} "
        f"total={usage.total_tokens} tokens"
    )
