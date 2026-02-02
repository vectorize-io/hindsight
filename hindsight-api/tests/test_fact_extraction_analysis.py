"""
Test to analyze fact extraction token usage and identify optimization opportunities.
"""
import asyncio
import logging
import time
from datetime import datetime

import pytest

from hindsight_api.config import get_config, clear_config_cache
from hindsight_api.engine.llm_wrapper import LLMConfig
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def llm_config():
    """Create LLM config from environment."""
    clear_config_cache()
    config = get_config()
    return LLMConfig(
        provider=config.retain_llm_provider or config.llm_provider,
        api_key=config.retain_llm_api_key or config.llm_api_key,
        model=config.retain_llm_model or config.llm_model,
        base_url=config.retain_llm_base_url or config.llm_base_url,
    )


@pytest.mark.asyncio
async def test_fact_extraction_basic_analysis(llm_config):
    """
    Test fact extraction and analyze token usage with sample content.

    This test helps identify:
    1. How many facts are extracted
    2. Token usage (input/output ratio)
    3. Types of facts being extracted
    """
    content = """
    Alice is a senior software engineer at TechCorp with 8 years of experience.
    She has a Kubernetes certification (CKA) and leads the platform team.
    Bob is her colleague who works on the frontend. He's been at the company for 3 years.
    They're working on a new microservices migration project together.
    The deadline for the first milestone is end of Q2.
    Alice prefers to use Go for backend services while Bob advocates for TypeScript.
    """

    logger.info(f"Content length: {len(content)} chars (~{len(content) // 4} tokens)")

    start_time = time.time()

    facts, chunks, usage = await extract_facts_from_text(
        text=content,
        event_date=datetime.now(),
        llm_config=llm_config,
        agent_name="test-agent",
        context="Friday Standup meeting",
    )

    duration = time.time() - start_time

    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Duration: {duration:.2f}s")
    logger.info(f"Chunks: {len(chunks)}")
    logger.info(f"Facts extracted: {len(facts)}")
    logger.info(f"Input tokens: {usage.input_tokens}")
    logger.info(f"Output tokens: {usage.output_tokens}")
    logger.info(f"Token ratio (out/in): {usage.output_tokens / max(1, usage.input_tokens):.2f}")

    # Analyze facts by type
    fact_types = {}
    for fact in facts:
        ft = fact.fact_type
        fact_types[ft] = fact_types.get(ft, 0) + 1

    logger.info(f"\nFacts by type:")
    for ft, count in sorted(fact_types.items()):
        logger.info(f"  {ft}: {count}")

    # Show sample facts
    logger.info(f"\nSample facts (first 10):")
    for i, fact in enumerate(facts[:10]):
        logger.info(f"\n  [{i+1}] {fact.fact_type}: {fact.fact[:150]}...")

    # Show facts containing key terms
    key_terms = ["kubernetes", "k8s", "CKA", "certification", "Alice"]
    logger.info(f"\n{'='*60}")
    logger.info(f"FACTS CONTAINING KEY TERMS")
    logger.info(f"{'='*60}")

    for term in key_terms:
        matching = [f for f in facts if term.lower() in f.fact.lower()]
        logger.info(f"\n'{term}' ({len(matching)} facts):")
        for fact in matching[:3]:
            logger.info(f"  - {fact.fact[:200]}...")

    assert len(facts) > 0, "Should extract at least one fact"
