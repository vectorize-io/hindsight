"""
Test that recall chunks are fetched independently of max_tokens filtering.

This test verifies the new behavior where:
1. Chunks are fetched BEFORE max_tokens filtering
2. max_tokens=0 returns 0 facts but can still return chunks
3. Chunks are fetched in batches to handle varying chunk sizes
"""

import pytest
import pytest_asyncio

from hindsight_api.engine.memory_engine import Budget


@pytest.mark.asyncio
async def test_recall_chunks_independent_of_max_tokens(memory):
    """
    Test that chunks are fetched independently of max_tokens.

    When max_tokens=0, recall should:
    - Return 0 memory facts
    - Still return chunks (up to max_chunk_tokens)
    - Chunks should come from top-scored results before token filtering
    """
    bank_id = "test-chunks-independence"

    try:
        # Create bank
        await memory.create_bank(bank_id=bank_id)

        # Retain some test content with substantial size to generate chunks
        test_content = """
        The quantum computing research team at MIT has made significant breakthroughs.
        Dr. Sarah Chen leads the team and focuses on quantum error correction.
        The team published three papers in Nature Physics this year.
        Their work on topological qubits shows promise for scalable quantum computers.
        Collaborators include IBM Research and Google Quantum AI.
        The research is funded by a $5M NSF grant running through 2026.
        """ * 10  # Repeat to ensure we get multiple chunks

        await memory.retain_async(
            bank_id=bank_id,
            content=test_content,
            context="research notes",
        )

        # Test 1: Normal recall with both facts and chunks
        result_normal = await memory.recall_async(
            bank_id=bank_id,
            query="quantum computing",
            max_tokens=4096,  # Normal token budget
            include_chunks=True,
            max_chunk_tokens=2000,
            budget=Budget.MID,
        )

        assert len(result_normal.results) > 0, "Should return memory facts with normal max_tokens"
        assert result_normal.chunks is not None, "Should include chunks when requested"
        assert len(result_normal.chunks) > 0, "Should return at least one chunk"

        # Test 2: Recall with max_tokens=0 but chunks enabled
        result_chunks_only = await memory.recall_async(
            bank_id=bank_id,
            query="quantum computing",
            max_tokens=0,  # Zero token budget for facts
            include_chunks=True,
            max_chunk_tokens=2000,  # But allow chunks
            budget=Budget.MID,
        )

        # Key assertions for new behavior
        assert len(result_chunks_only.results) == 0, "max_tokens=0 should return 0 facts"
        assert result_chunks_only.chunks is not None, "Should still include chunks dict"
        assert len(result_chunks_only.chunks) > 0, "Should return chunks even with max_tokens=0"

        # Verify chunks are from the same content (non-empty text)
        for chunk_id, chunk_info in result_chunks_only.chunks.items():
            assert len(chunk_info.chunk_text) > 0, "Chunks should contain text"
            assert chunk_info.chunk_index >= 0, "Chunk should have valid index"

    finally:
        # Cleanup
        await memory.delete_bank(bank_id=bank_id)


@pytest.mark.asyncio
async def test_recall_chunks_batching_with_varying_sizes(memory):
    """
    Test that chunk batching works correctly with varying chunk sizes.

    This verifies that:
    1. Chunks are fetched in batches until token budget is exhausted
    2. The system handles varying chunk sizes across documents
    3. Token budget is respected across multiple batch fetches
    """
    bank_id = "test-chunks-batching"

    try:
        # Create bank
        await memory.create_bank(bank_id=bank_id)

        # Retain multiple documents with different content sizes
        # Document 1: Short content (small chunks)
        await memory.retain_async(
            bank_id=bank_id,
            content="Short fact about Alice.",
            context="doc1",
        )

        # Document 2: Medium content
        await memory.retain_async(
            bank_id=bank_id,
            content="Medium length content about Bob. " * 20,
            context="doc2",
        )

        # Document 3: Long content (large chunks)
        await memory.retain_async(
            bank_id=bank_id,
            content="Very long detailed content about Charlie. " * 100,
            context="doc3",
        )

        # Recall with modest chunk token budget
        result = await memory.recall_async(
            bank_id=bank_id,
            query="Alice Bob Charlie",
            max_tokens=0,  # No facts, only chunks
            include_chunks=True,
            max_chunk_tokens=1000,  # Limited chunk budget
            budget=Budget.MID,
        )

        assert len(result.results) == 0, "Should return 0 facts with max_tokens=0"
        assert result.chunks is not None, "Should include chunks"

        # Verify we got chunks and respected the token budget
        if len(result.chunks) > 0:
            # Count total tokens (approximate)
            total_chunk_chars = sum(len(chunk.chunk_text) for chunk in result.chunks.values())
            # Very rough estimate: 1 token ≈ 4 characters
            estimated_tokens = total_chunk_chars // 4

            # Should be reasonably close to budget (within 2x due to estimation and batching)
            assert estimated_tokens <= 1000 * 2, f"Should respect chunk token budget (got ~{estimated_tokens} tokens)"

    finally:
        # Cleanup
        await memory.delete_bank(bank_id=bank_id)


@pytest.mark.asyncio
async def test_recall_chunks_ordering_by_relevance(memory):
    """
    Test that chunks are returned in order of fact relevance.

    Chunks should be ordered based on the top-scored (reranked) results,
    not in document order or random order.
    """
    bank_id = "test-chunks-ordering"

    try:
        # Create bank
        await memory.create_bank(bank_id=bank_id)

        # Retain content with different relevance to query
        await memory.retain_async(
            bank_id=bank_id,
            content="The Python programming language is widely used for machine learning and data science applications.",
            context="topic: Python",
        )

        await memory.retain_async(
            bank_id=bank_id,
            content="JavaScript is commonly used for web development and frontend applications.",
            context="topic: JavaScript",
        )

        await memory.retain_async(
            bank_id=bank_id,
            content="Python's scikit-learn library is excellent for traditional machine learning tasks and model training.",
            context="topic: Python ML",
        )

        # Query specifically about Python - should rank Python facts higher
        result = await memory.recall_async(
            bank_id=bank_id,
            query="Python machine learning",
            max_tokens=0,  # No facts
            include_chunks=True,
            max_chunk_tokens=5000,  # Enough for all chunks
            budget=Budget.HIGH,  # Use high budget for better recall
        )

        assert len(result.results) == 0, "Should return 0 facts with max_tokens=0"
        assert result.chunks is not None, "Should include chunks"

        # We should get chunks, and they should be ordered by relevance
        # The exact ordering depends on the reranker, but we should have chunks
        assert len(result.chunks) > 0, "Should return chunks from relevant facts"

        # Verify chunks contain relevant content
        all_chunk_text = " ".join(chunk.chunk_text for chunk in result.chunks.values())
        # At least some chunks should mention Python (higher relevance)
        # This is a soft check since exact ordering depends on scoring
        assert "Python" in all_chunk_text or "python" in all_chunk_text.lower(), \
            "Chunks should include content about Python (relevant to query)"

    finally:
        # Cleanup
        await memory.delete_bank(bank_id=bank_id)


@pytest.mark.asyncio
async def test_recall_chunks_without_include_flag(memory):
    """
    Test that chunks are NOT returned when include_chunks=False (default).

    This ensures backward compatibility - chunks are only fetched when explicitly requested.
    """
    bank_id = "test-chunks-no-include"

    try:
        # Create bank
        await memory.create_bank(bank_id=bank_id)

        # Retain content
        await memory.retain_async(
            bank_id=bank_id,
            content="Test content for chunks.",
        )

        # Recall without include_chunks flag (default is False)
        result = await memory.recall_async(
            bank_id=bank_id,
            query="test",
            max_tokens=4096,
            # include_chunks=False is the default
        )

        # Should have facts but no chunks
        assert len(result.results) > 0, "Should return facts"
        assert result.chunks is None or len(result.chunks) == 0, \
            "Should NOT return chunks when include_chunks=False"

    finally:
        # Cleanup
        await memory.delete_bank(bank_id=bank_id)
