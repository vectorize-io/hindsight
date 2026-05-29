"""Live integration test for the Fireworks AI batch-inference provider.

This makes REAL calls to Fireworks' batch API and runs the full retain fact
extraction pipeline end-to-end (submit -> poll -> download -> normalize ->
parse facts). It is the only test that validates the one assumption the unit
tests (which use mocked httpx responses) cannot: that Fireworks' real output
JSONL shape matches what ``_normalize_output_line`` produces and what
``fact_extraction`` consumes. If the shape is wrong, this returns zero facts.

Skipped automatically unless credentials are present. To run:

    export HINDSIGHT_API_FIREWORKS_API_KEY=fw_xxx     # or FIREWORKS_API_KEY
    export HINDSIGHT_API_FIREWORKS_ACCOUNT_ID=your-account-id
    # optional: override the (must be batch-eligible) model
    export HINDSIGHT_API_FIREWORKS_TEST_MODEL=accounts/fireworks/models/llama-v3p1-8b-instruct
    uv run pytest tests/test_fireworks_batch_integration.py -v -s

It is slow (minutes, depending on Fireworks' queue) and costs money, so it does
not run in CI (no key there). No database is required — it calls the extraction
function directly with ``pool=None``.
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

_DEFAULT_TEST_MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"


@dataclass
class FireworksTestEnv:
    api_key: str
    account_id: str
    model: str


@pytest.fixture
def fireworks_env(monkeypatch) -> FireworksTestEnv:
    api_key = os.getenv("HINDSIGHT_API_FIREWORKS_API_KEY") or os.getenv("FIREWORKS_API_KEY")
    account_id = os.getenv("HINDSIGHT_API_FIREWORKS_ACCOUNT_ID")
    if not api_key or not account_id:
        pytest.skip(
            "Set HINDSIGHT_API_FIREWORKS_API_KEY (or FIREWORKS_API_KEY) and "
            "HINDSIGHT_API_FIREWORKS_ACCOUNT_ID to run the live Fireworks batch test"
        )

    # FireworksLLM resolves the account id from global config, so make sure the
    # cached config picks it up for this run.
    monkeypatch.setenv("HINDSIGHT_API_FIREWORKS_ACCOUNT_ID", account_id)
    clear_config_cache()

    return FireworksTestEnv(
        api_key=api_key,
        account_id=account_id,
        model=os.getenv("HINDSIGHT_API_FIREWORKS_TEST_MODEL", _DEFAULT_TEST_MODEL),
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_fireworks_batch_end_to_end(fireworks_env):
    config = HindsightConfig.from_env()
    config.retain_batch_enabled = True
    config.retain_batch_poll_interval_seconds = 30
    config.retain_chunk_size = 4000
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = False

    llm_config = LLMProvider(
        provider="fireworks",
        api_key=fireworks_env.api_key,
        base_url="",  # defaults to the Fireworks inference host
        model=fireworks_env.model,
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

    logger.info("Submitting a real Fireworks batch (this can take several minutes)...")
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
        "Fireworks batch returned no facts. The live output JSONL shape likely "
        "differs from what _normalize_output_line produces — inspect a raw output "
        "line and adjust the normalizer."
    )
    assert any("Alice" in fact.fact_text for fact in facts)
    logger.info(f"Extracted {len(facts)} facts; usage={usage.total_tokens} tokens")
