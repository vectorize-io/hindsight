"""
Real integration test for OpenAI Batch API.

This test makes REAL API calls to OpenAI and measures actual timing.
It will be slow (minutes to hours) depending on OpenAI's queue.

To run:
    pytest tests/test_batch_api_integration.py -v -s

To skip in CI:
    Add @pytest.mark.skip at the test level
"""
import pytest
import os
import asyncio
import logging
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from hindsight_api import RequestContext
from hindsight_api.engine.retain.fact_extraction import (
    extract_facts_from_contents_batch_api,
    RetainContent,
)
from hindsight_api.config import HindsightConfig
from hindsight_api.engine.llm_wrapper import LLMProvider

logger = logging.getLogger(__name__)

# Load .env file for API keys
load_dotenv()


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    # Try both current and commented keys from .env
    api_key = os.getenv("HINDSIGHT_API_LLM_API_KEY")

    # Check if it's an OpenAI key (starts with sk-proj- or sk-)
    if not api_key or not api_key.startswith("sk-"):
        # Try the OpenAI-specific env var (if set separately)
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or not api_key.startswith("sk-"):
        pytest.skip("OpenAI API key not found in environment. Set OPENAI_API_KEY or uncomment OpenAI config in .env")

    return api_key


@pytest.fixture
def real_llm_config(openai_api_key):
    """Create real LLM config for OpenAI."""
    # Create config with OpenAI settings
    config = HindsightConfig.from_env()

    # Use LLMProvider wrapper (which creates _provider_impl internally)
    llm_config = LLMProvider(
        provider="openai",
        api_key=openai_api_key,
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",  # Fast, cheap model for testing
        reasoning_effort="medium",  # Required parameter
    )

    return llm_config


@pytest.fixture
def test_contents_real():
    """Create realistic test content for fact extraction."""
    return [
        RetainContent(
            content="""
            Alice is a senior software engineer at TechCorp, where she has been working for 5 years.
            She specializes in distributed systems and microservices architecture. Alice graduated
            from MIT with a degree in Computer Science in 2015. She is known for writing clean,
            well-documented code and mentoring junior developers.
            """,
            event_date=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            context="team member profile",
        ),
        RetainContent(
            content="""
            Bob joined TechCorp last month as a junior developer. He is learning React and Node.js
            and recently completed his first feature, which was a user authentication flow. Bob
            graduated from Berkeley with a degree in Computer Science in 2023. He is enthusiastic
            and asks great questions during code reviews.
            """,
            event_date=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            context="team member profile",
        ),
        RetainContent(
            content="""
            The team uses Kubernetes for container orchestration and deploys to AWS. They follow
            agile methodologies with two-week sprints. Code reviews are mandatory before merging
            any pull request. The team meets every morning for a 15-minute standup to discuss
            progress and blockers.
            """,
            event_date=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            context="team processes",
        ),
    ]


@pytest.fixture
def integration_config():
    """Create config for integration test."""
    config = HindsightConfig.from_env()
    config.retain_use_batch_api = True
    config.retain_batch_poll_interval = 30  # Poll every 30 seconds (reasonable for real API)
    config.retain_chunk_size = 4000
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = False
    return config


@pytest.mark.skip(reason="Real API test - takes minutes and costs money. Run manually with: pytest tests/test_batch_api_integration.py::test_real_openai_batch_api -v -s")
@pytest.mark.integration  # Mark as integration test
@pytest.mark.slow  # Mark as slow test
@pytest.mark.asyncio
async def test_real_openai_batch_api(real_llm_config, test_contents_real, integration_config, memory, request_context):
    """
    REAL integration test: Submit actual batch to OpenAI and measure timing.

    WARNING: This test:
    - Makes real API calls to OpenAI
    - Will take minutes to hours to complete
    - Costs money (though very little with gpt-4o-mini)
    - Requires valid OpenAI API key

    To skip this test:
        pytest tests/test_batch_api_integration.py --skip-integration
    """
    bank_id = f"test_real_batch_{datetime.now(timezone.utc).timestamp()}"

    logger.info("=" * 80)
    logger.info("STARTING REAL OPENAI BATCH API INTEGRATION TEST")
    logger.info("=" * 80)
    logger.info(f"Test contents: {len(test_contents_real)} items")
    logger.info(f"Poll interval: {integration_config.retain_batch_poll_interval}s")
    logger.info(f"Model: {real_llm_config.model}")
    logger.info("This may take several minutes to hours depending on OpenAI's queue...")
    logger.info("=" * 80)

    try:
        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Get database pool and schema for crash recovery testing
        pool = memory._pool
        schema = request_context.tenant_id

        # Track overall timing
        test_start_time = time.time()

        # Call REAL batch API extraction
        logger.info("\n📤 Submitting batch to OpenAI...")

        facts, chunks, usage = await extract_facts_from_contents_batch_api(
            contents=test_contents_real,
            llm_config=real_llm_config,
            agent_name="test_agent",
            config=integration_config,
            pool=pool,
            operation_id=None,  # No crash recovery for this test
            schema=schema,
        )

        test_end_time = time.time()
        total_duration = test_end_time - test_start_time

        # Log results
        logger.info("\n" + "=" * 80)
        logger.info("✅ BATCH COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        logger.info(f"Facts extracted: {len(facts)}")
        logger.info(f"Chunks processed: {len(chunks)}")
        logger.info(f"Token usage: {usage.input_tokens} input + {usage.output_tokens} output = {usage.total_tokens} total")
        logger.info(f"Estimated cost: ${(usage.input_tokens * 0.00015 / 1000 + usage.output_tokens * 0.0006 / 1000):.4f}")
        logger.info("=" * 80)

        # Log sample facts
        logger.info("\n📋 Sample extracted facts:")
        for i, fact in enumerate(facts[:5]):  # Show first 5 facts
            logger.info(f"\nFact {i+1}:")
            logger.info(f"  Type: {fact.fact_type}")
            logger.info(f"  Text: {fact.fact_text[:100]}...")
            logger.info(f"  Entities: {fact.entities}")

        # Verify results
        assert len(facts) > 0, "Should extract at least some facts"
        assert len(chunks) == len(test_contents_real), f"Should have {len(test_contents_real)} chunks"
        assert usage.total_tokens > 0, "Should have token usage"

        # Verify fact structure
        for fact in facts:
            assert hasattr(fact, "fact_text"), "Fact should have fact_text"
            assert hasattr(fact, "fact_type"), "Fact should have fact_type"
            assert fact.fact_type in ["world", "experience", "opinion"], f"Invalid fact_type: {fact.fact_type}"

        logger.info("\n✅ All assertions passed!")

        # Write timing report to file for later analysis
        report_path = "/tmp/openai_batch_api_timing_report.txt"
        with open(report_path, "w") as f:
            f.write(f"OpenAI Batch API Integration Test Report\n")
            f.write(f"={'=' * 60}\n\n")
            f.write(f"Test Date: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"Model: {real_llm_config.model}\n")
            f.write(f"Contents: {len(test_contents_real)} items\n")
            f.write(f"Poll Interval: {integration_config.retain_batch_poll_interval}s\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)\n")
            f.write(f"  Facts Extracted: {len(facts)}\n")
            f.write(f"  Chunks Processed: {len(chunks)}\n")
            f.write(f"  Token Usage: {usage.total_tokens} ({usage.input_tokens} in + {usage.output_tokens} out)\n")
            f.write(f"  Estimated Cost: ${(usage.input_tokens * 0.00015 / 1000 + usage.output_tokens * 0.0006 / 1000):.4f}\n")

        logger.info(f"\n📄 Timing report written to: {report_path}")

    finally:
        # Cleanup
        try:
            await memory.delete_bank(bank_id, request_context=request_context)
            logger.info(f"\n🧹 Cleaned up test bank: {bank_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup bank: {e}")


@pytest.mark.skip(reason="Real API test - requires Groq API key. Run manually if needed.")
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_batch_supports_groq(integration_config):
    """
    Test that Groq also supports batch API (if configured).

    Groq has the same batch API interface as OpenAI.
    """
    groq_api_key = os.getenv("HINDSIGHT_API_LLM_API_KEY")

    if not groq_api_key or not groq_api_key.startswith("gsk_"):
        pytest.skip("Groq API key not found in environment")

    llm_config = LLMProvider(
        provider="groq",
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        reasoning_effort="medium",
    )

    # Check if Groq supports batch API
    supports_batch = await llm_config._provider_impl.supports_batch_api()

    logger.info(f"Groq batch API support: {supports_batch}")

    # Groq should support batch API (same interface as OpenAI)
    assert supports_batch, "Groq should support batch API"

    logger.info("✅ Groq batch API support confirmed")
