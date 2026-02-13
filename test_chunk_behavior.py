#!/usr/bin/env python3
"""
Quick test to verify the new chunk behavior:
- Chunks are fetched independently of max_tokens
- max_tokens=0 returns 0 facts but can return chunks
- Number of chunks is limited
"""

import asyncio
import os
import sys
from pathlib import Path

# Add hindsight-api to path
sys.path.insert(0, str(Path(__file__).parent / "hindsight-api"))

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.config import HindsightConfig
from hindsight_api.extensions import RequestContext


async def test_chunk_independence():
    """Test that chunks are fetched independently of max_tokens"""

    # Create a minimal config for testing
    config = HindsightConfig.from_env()

    # Create memory engine
    engine = MemoryEngine(config)

    # Create test bank
    bank_id = "test-chunks-independence"
    request_context = RequestContext()

    try:
        # Create bank
        await engine.create_bank(
            bank_id=bank_id,
            request_context=request_context,
        )

        # Retain some test data
        await engine.retain_async(
            bank_id=bank_id,
            content="The quick brown fox jumps over the lazy dog. " * 50,  # Create some content
            request_context=request_context,
        )

        # Test 1: Normal recall with facts and chunks
        print("\n=== Test 1: Normal recall (max_tokens=4096) ===")
        result1 = await engine.recall_async(
            bank_id=bank_id,
            query="fox",
            max_tokens=4096,
            include_chunks=True,
            max_chunk_tokens=1000,
            request_context=request_context,
        )
        print(f"Facts returned: {len(result1.results)}")
        print(f"Chunks returned: {len(result1.chunks or {})}")

        # Test 2: Recall with max_tokens=0 but include_chunks=True
        print("\n=== Test 2: Recall with max_tokens=0 but chunks enabled ===")
        result2 = await engine.recall_async(
            bank_id=bank_id,
            query="fox",
            max_tokens=0,  # Should return 0 facts
            include_chunks=True,  # But should still return chunks
            max_chunk_tokens=1000,
            request_context=request_context,
        )
        print(f"Facts returned: {len(result2.results)}")
        print(f"Chunks returned: {len(result2.chunks or {})}")

        if len(result2.results) == 0 and len(result2.chunks or {}) > 0:
            print("\n✅ SUCCESS: max_tokens=0 returns 0 facts but still returns chunks!")
        else:
            print(f"\n❌ FAILED: Expected 0 facts and >0 chunks, got {len(result2.results)} facts and {len(result2.chunks or {})} chunks")

    finally:
        # Cleanup
        await engine.delete_bank(bank_id=bank_id, request_context=request_context)
        await engine.close()


if __name__ == "__main__":
    # Check for required env vars
    if not os.getenv("HINDSIGHT_API_LLM_API_KEY"):
        print("❌ Error: HINDSIGHT_API_LLM_API_KEY not set")
        print("Please set your LLM API key in .env file")
        sys.exit(1)

    asyncio.run(test_chunk_independence())
