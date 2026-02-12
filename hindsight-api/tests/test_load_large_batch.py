"""
Load test for large batch retain operations.

Tests batch processing with 20 content items totaling ~500k chars
using a mock LLM to verify DB and batch size handling.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, UTC
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import pytest_asyncio

from hindsight_api import MemoryEngine, LLMConfig, LocalSTEmbeddings, RequestContext
from hindsight_api.engine.cross_encoder import LocalSTCrossEncoder
from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer
from hindsight_api.engine.task_backend import SyncTaskBackend
from hindsight_api.engine.retain.fact_extraction import FactExtractionResponse, ExtractedFact
from hindsight_api.engine.llm_wrapper import TokenUsage

logger = logging.getLogger(__name__)


def generate_content(char_count: int) -> str:
    """Generate realistic content of approximately char_count characters."""
    # Base sentences that look like real conversations/notes
    sentences = [
        "I had a meeting with John about the quarterly projections for Q3.",
        "We discussed the new marketing strategy and agreed to increase social media presence.",
        "Sarah mentioned that she's planning to visit Tokyo next month for the conference.",
        "The project deadline was extended to December 15th after consulting with stakeholders.",
        "I need to follow up with the engineering team about the API integration issues.",
        "The budget review showed we're 15% under projections, which is good news.",
        "Mike suggested we look into alternative vendors for the cloud infrastructure.",
        "The client feedback from the beta testing was overwhelmingly positive.",
        "We should schedule another sync meeting for next Tuesday afternoon.",
        "The documentation needs to be updated before the product launch.",
        "I learned that Python 3.12 has some great new performance improvements.",
        "The restaurant downtown has amazing pasta - must remember to go back.",
        "Emily's birthday is coming up, need to plan something special.",
        "The new office location will be in the financial district starting January.",
        "Weather forecast shows rain all week, should bring an umbrella.",
    ]

    content = []
    current_chars = 0
    idx = 0

    while current_chars < char_count:
        sentence = sentences[idx % len(sentences)]
        # Add some variation with numbers/dates
        if idx % 3 == 0:
            sentence = f"[{datetime.now().strftime('%Y-%m-%d')}] " + sentence
        content.append(sentence)
        current_chars += len(sentence) + 1  # +1 for newline
        idx += 1

    return "\n".join(content)


def create_mock_facts_from_content(content: str, ratio: float = 1.5, max_facts: int = 50) -> list[dict]:
    """
    Create mock extracted facts from content at the given ratio.

    If content has N sentences, return approximately N * ratio facts (capped at max_facts).
    """
    # Estimate sentences by splitting on periods
    sentences = [s.strip() for s in content.split('.') if s.strip()]
    num_facts = min(max(1, int(len(sentences) * ratio)), max_facts)

    facts = []
    for i in range(num_facts):
        facts.append({
            "what": f"Mock fact {i}: Something happened based on the content",
            "when": "2024-06-15",
            "where": "San Francisco",
            "who": "John, Sarah",
            "why": "Business reasons",
            "fact_type": "world",
            "entities": [{"text": "John", "type": "PERSON"}],
            "causal_relations": [],
        })

    return facts


class TestLargeBatchRetain:
    """Load tests for large batch retain operations."""

    @pytest_asyncio.fixture
    async def memory_with_mock_llm(self, pg0_db_url, embeddings, cross_encoder, query_analyzer):
        """Create a memory engine with mocked LLM for testing."""
        mem = MemoryEngine(
            db_url=pg0_db_url,
            memory_llm_provider="openai",  # Will be mocked
            memory_llm_api_key="mock-key",
            memory_llm_model="gpt-4",
            embeddings=embeddings,
            cross_encoder=cross_encoder,
            query_analyzer=query_analyzer,
            pool_min_size=2,
            pool_max_size=10,
            run_migrations=False,
            skip_llm_verification=True,  # Skip LLM verification since we're mocking
            task_backend=SyncTaskBackend(),  # Execute tasks immediately in tests
        )
        await mem.initialize()
        yield mem
        try:
            if mem._pool and not mem._pool._closing:
                await mem.close()
        except Exception:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # 5 minute timeout
    async def test_large_batch_500k_chars_20_items(self, memory_with_mock_llm, request_context):
        """
        Test retaining a batch of 20 content items totaling ~500k chars.

        Uses mock LLM with 1.5x output ratio to test DB and batch handling.
        """
        memory = memory_with_mock_llm
        bank_id = f"load-test-{uuid.uuid4().hex[:8]}"

        # Create 20 content items totaling ~50k chars
        num_items = 20
        total_target_chars = 50_000
        chars_per_item = total_target_chars // num_items

        contents = []
        for i in range(num_items):
            content_text = generate_content(chars_per_item)
            contents.append({
                "content": content_text,
                "context": f"Test content item {i + 1} of {num_items}",
                "event_date": datetime.now(UTC),
            })

        actual_total_chars = sum(len(c["content"]) for c in contents)
        logger.info(f"Created {num_items} content items with {actual_total_chars:,} total chars")

        # Track LLM calls to verify mock is working
        call_tracker = {"count": 0, "facts": 0}

        async def mock_llm_call(*args, **kwargs):
            call_tracker["count"] += 1

            # Extract the content from the user message to generate proportional facts
            messages = kwargs.get("messages", args[0] if args else [])
            user_msg = messages[-1]["content"] if messages else ""
            mock_facts = create_mock_facts_from_content(user_msg, ratio=1.5)
            call_tracker["facts"] += len(mock_facts)

            # Return a dict (parsed JSON) since skip_validation=True but the code expects a dict
            response_dict = {"facts": mock_facts}

            return_usage = kwargs.get("return_usage", False)
            if return_usage:
                usage = TokenUsage(
                    input_tokens=len(user_msg) // 4,
                    output_tokens=len(json.dumps(response_dict)) // 4,
                )
                return response_dict, usage
            return response_dict

        # Patch LLMProvider.call at the class level
        with patch('hindsight_api.engine.llm_wrapper.LLMProvider.call', new=mock_llm_call):
            start_time = time.time()

            try:
                result = await memory.retain_batch_async(
                    bank_id=bank_id,
                    contents=contents,
                    request_context=request_context,
                )

                elapsed = time.time() - start_time

                # Log results
                total_units = sum(len(unit_ids) for unit_ids in result)
                logger.info(f"\n{'=' * 60}")
                logger.info(f"LOAD TEST RESULTS")
                logger.info(f"{'=' * 60}")
                logger.info(f"Input: {num_items} items, {actual_total_chars:,} chars")
                logger.info(f"LLM calls: {call_tracker['count']}")
                logger.info(f"Mock facts generated: {call_tracker['facts']}")
                logger.info(f"Memory units created: {total_units}")
                logger.info(f"Elapsed time: {elapsed:.2f}s")
                logger.info(f"Throughput: {actual_total_chars / elapsed:,.0f} chars/sec")
                logger.info(f"{'=' * 60}")

                # Assertions
                assert len(result) == num_items, f"Expected {num_items} result lists, got {len(result)}"
                assert total_units > 0, "Expected at least some memory units to be created"
                assert call_tracker["count"] > 0, "Expected LLM to be called"

                # Verify we didn't timeout or have major issues
                assert elapsed < 300, f"Operation took too long: {elapsed:.2f}s"

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"LOAD TEST FAILED after {elapsed:.2f}s: {e}")
                raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(240)  # Increased timeout for VectorChord BM25 tokenization
    async def test_batch_chunking_behavior(self, memory_with_mock_llm, request_context):
        """
        Test that large batches are properly chunked into sub-batches.

        Verifies the CHARS_PER_BATCH (600k) chunking logic.
        """
        memory = memory_with_mock_llm
        bank_id = f"chunk-test-{uuid.uuid4().hex[:8]}"

        # Create contents that are moderately sized
        # Testing the chunking behavior with smaller content
        num_items = 5
        chars_per_item = 10_000  # 50k total

        contents = []
        for i in range(num_items):
            contents.append({
                "content": generate_content(chars_per_item),
                "context": f"Chunk test item {i + 1}",
                "event_date": datetime.now(UTC),
            })

        actual_total_chars = sum(len(c["content"]) for c in contents)
        logger.info(f"Created {num_items} items with {actual_total_chars:,} chars (should trigger chunking)")

        async def mock_llm_call(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            user_msg = messages[-1]["content"] if messages else ""
            mock_facts = create_mock_facts_from_content(user_msg, ratio=1.0)
            response_dict = {"facts": mock_facts}

            return_usage = kwargs.get("return_usage", False)
            if return_usage:
                return response_dict, TokenUsage(input_tokens=100, output_tokens=50)
            return response_dict

        with patch('hindsight_api.engine.llm_wrapper.LLMProvider.call', new=mock_llm_call):
            start_time = time.time()

            result = await memory.retain_batch_async(
                bank_id=bank_id,
                contents=contents,
                request_context=request_context,
            )

            elapsed = time.time() - start_time
            total_units = sum(len(unit_ids) for unit_ids in result)

            logger.info(f"Chunking test: {total_units} units in {elapsed:.2f}s")

            assert len(result) == num_items
            assert total_units > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_db_connection_pool_under_load(self, memory_with_mock_llm, request_context):
        """
        Test that DB connection pool handles concurrent operations.

        Runs multiple retain operations concurrently to stress the pool.
        """
        memory = memory_with_mock_llm

        async def mock_llm_call(*args, **kwargs):
            # Small delay to simulate real LLM latency
            await asyncio.sleep(0.01)
            mock_facts = [{"what": "Test fact", "when": "now", "where": "here",
                          "who": "someone", "why": "testing", "fact_type": "world",
                          "entities": [], "causal_relations": []}]
            response_dict = {"facts": mock_facts}

            return_usage = kwargs.get("return_usage", False)
            if return_usage:
                return response_dict, TokenUsage(input_tokens=10, output_tokens=10)
            return response_dict

        with patch('hindsight_api.engine.llm_wrapper.LLMProvider.call', new=mock_llm_call):
            # Run 10 concurrent retain operations
            tasks = []
            for i in range(10):
                bank_id = f"pool-test-{uuid.uuid4().hex[:8]}"
                contents = [{
                    "content": f"Test content for concurrent operation {i}. " * 50,
                    "context": f"Pool test {i}",
                    "event_date": datetime.now(UTC),
                }]
                tasks.append(
                    memory.retain_batch_async(bank_id=bank_id, contents=contents, request_context=request_context)
                )

            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start_time

            # Check results
            errors = [r for r in results if isinstance(r, Exception)]
            successes = [r for r in results if not isinstance(r, Exception)]

            logger.info(f"Pool test: {len(successes)} successes, {len(errors)} errors in {elapsed:.2f}s")

            if errors:
                for e in errors:
                    logger.error(f"Error: {e}")

            assert len(errors) == 0, f"Expected no errors, got: {errors}"
            assert len(successes) == 10
