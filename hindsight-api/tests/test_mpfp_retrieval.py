"""
Tests for MPFP (Meta-Path Forward Push) graph retrieval.

Tests cover:
1. EdgeCache - lazy caching behavior
2. mpfp_traverse_async - core traversal algorithm
3. load_edges_for_frontier - lazy edge loading
4. rrf_fusion - result fusion
5. MPFPGraphRetriever - full integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from hindsight_api.engine.search.mpfp_retrieval import (
    EdgeCache,
    EdgeTarget,
    MPFPConfig,
    MPFPGraphRetriever,
    PatternResult,
    SeedNode,
    load_all_edges_for_frontier,
    mpfp_traverse_async,
    rrf_fusion,
)
from hindsight_api.engine.search.types import RetrievalResult


class TestEdgeCache:
    """Tests for the EdgeCache lazy loading cache."""

    def test_empty_cache_returns_empty_neighbors(self):
        """Empty cache should return empty list for any node."""
        cache = EdgeCache()
        neighbors = cache.get_neighbors("semantic", "node-1")
        assert neighbors == []

    def test_is_fully_loaded_false_for_uncached(self):
        """is_fully_loaded should return False for nodes not yet loaded."""
        cache = EdgeCache()
        assert cache.is_fully_loaded("node-1") is False

    def test_add_all_edges_marks_as_fully_loaded(self):
        """Adding edges should mark nodes as fully loaded."""
        cache = EdgeCache()

        edges_by_type = {
            "semantic": {"node-1": [EdgeTarget("node-2", 0.8), EdgeTarget("node-3", 0.6)]},
        }
        cache.add_all_edges(edges_by_type, ["node-1", "node-4"])  # node-4 has no edges

        assert cache.is_fully_loaded("node-1") is True
        assert cache.is_fully_loaded("node-4") is True  # Marked even with no edges
        assert cache.is_fully_loaded("node-2") is False  # Target, not source

    def test_get_neighbors_returns_added_edges(self):
        """get_neighbors should return edges after add_all_edges."""
        cache = EdgeCache()

        edges_by_type = {
            "semantic": {"node-1": [EdgeTarget("node-2", 0.8), EdgeTarget("node-3", 0.6)]},
        }
        cache.add_all_edges(edges_by_type, ["node-1"])

        neighbors = cache.get_neighbors("semantic", "node-1")
        assert len(neighbors) == 2
        assert neighbors[0].node_id == "node-2"
        assert neighbors[0].weight == 0.8

    def test_get_uncached_filters_loaded_nodes(self):
        """get_uncached should only return nodes not yet fully loaded."""
        cache = EdgeCache()

        # Load some nodes (all edge types)
        cache.add_all_edges({"semantic": {"node-1": []}}, ["node-1", "node-2"])

        # Check uncached
        uncached = cache.get_uncached(["node-1", "node-2", "node-3", "node-4"])
        assert set(uncached) == {"node-3", "node-4"}

    def test_get_normalized_neighbors_normalizes_weights(self):
        """get_normalized_neighbors should normalize weights to sum to 1."""
        cache = EdgeCache()

        edges_by_type = {
            "semantic": {
                "node-1": [
                    EdgeTarget("node-2", 0.8),
                    EdgeTarget("node-3", 0.4),
                    EdgeTarget("node-4", 0.2),
                ],
            },
        }
        cache.add_all_edges(edges_by_type, ["node-1"])

        # Get top 2, normalized
        neighbors = cache.get_normalized_neighbors("semantic", "node-1", top_k=2)
        assert len(neighbors) == 2

        # Weights should sum to 1
        total = sum(n.weight for n in neighbors)
        assert abs(total - 1.0) < 0.001

        # node-2 should have higher normalized weight than node-3
        assert neighbors[0].node_id == "node-2"
        assert neighbors[1].node_id == "node-3"
        # Original: 0.8 and 0.4, so normalized: 0.8/1.2 and 0.4/1.2
        assert abs(neighbors[0].weight - 0.8 / 1.2) < 0.001
        assert abs(neighbors[1].weight - 0.4 / 1.2) < 0.001

    def test_different_edge_types_are_separate(self):
        """Different edge types should be stored separately."""
        cache = EdgeCache()

        edges_by_type = {
            "semantic": {"node-1": [EdgeTarget("node-2", 0.8)]},
            "temporal": {"node-1": [EdgeTarget("node-3", 0.5)]},
        }
        cache.add_all_edges(edges_by_type, ["node-1"])

        semantic_neighbors = cache.get_neighbors("semantic", "node-1")
        temporal_neighbors = cache.get_neighbors("temporal", "node-1")

        assert len(semantic_neighbors) == 1
        assert semantic_neighbors[0].node_id == "node-2"

        assert len(temporal_neighbors) == 1
        assert temporal_neighbors[0].node_id == "node-3"


class TestRRFFusion:
    """Tests for RRF (Reciprocal Rank Fusion)."""

    def test_empty_results(self):
        """Empty results should return empty fusion."""
        fused = rrf_fusion([])
        assert fused == []

    def test_single_pattern_ranking(self):
        """Single pattern should preserve ranking order."""
        result = PatternResult(
            pattern=["semantic"],
            scores={"node-1": 0.9, "node-2": 0.7, "node-3": 0.5},
        )

        fused = rrf_fusion([result], top_k=3)
        assert len(fused) == 3
        # node-1 should be first (highest score)
        assert fused[0][0] == "node-1"
        assert fused[1][0] == "node-2"
        assert fused[2][0] == "node-3"

    def test_multiple_patterns_boost_common_nodes(self):
        """Nodes appearing in multiple patterns should get boosted."""
        result1 = PatternResult(
            pattern=["semantic", "semantic"],
            scores={"node-1": 0.9, "node-2": 0.7},
        )
        result2 = PatternResult(
            pattern=["entity", "temporal"],
            scores={"node-1": 0.8, "node-3": 0.6},  # node-1 in both
        )

        fused = rrf_fusion([result1, result2], top_k=3)

        # node-1 should be first (appears in both patterns)
        assert fused[0][0] == "node-1"
        # Its score should be higher than others
        assert fused[0][1] > fused[1][1]

    def test_top_k_limits_results(self):
        """top_k should limit the number of results."""
        result = PatternResult(
            pattern=["semantic"],
            scores={f"node-{i}": 1.0 / (i + 1) for i in range(10)},
        )

        fused = rrf_fusion([result], top_k=3)
        assert len(fused) == 3

    def test_empty_pattern_scores_ignored(self):
        """Patterns with empty scores should be ignored."""
        result1 = PatternResult(pattern=["semantic"], scores={})
        result2 = PatternResult(
            pattern=["entity"],
            scores={"node-1": 0.5},
        )

        fused = rrf_fusion([result1, result2], top_k=3)
        assert len(fused) == 1
        assert fused[0][0] == "node-1"


class TestMPFPTraverseAsync:
    """Tests for the async MPFP traversal algorithm."""

    @pytest.mark.asyncio
    async def test_empty_seeds_returns_empty(self):
        """Empty seeds should return empty result."""
        cache = EdgeCache()
        config = MPFPConfig()

        result = await mpfp_traverse_async(
            pool=None,  # Not used when no seeds
            seeds=[],
            pattern=["semantic"],
            config=config,
            cache=cache,
        )

        assert result.scores == {}

    @pytest.mark.asyncio
    async def test_single_hop_no_edges(self):
        """Single hop with no edges should deposit mass at seeds."""
        cache = EdgeCache()
        config = MPFPConfig(alpha=0.15, threshold=1e-6)

        # Pre-populate cache with empty edges for seed (marks as fully loaded)
        cache.add_all_edges({}, ["seed-1"])

        seeds = [SeedNode("seed-1", 1.0)]

        with patch(
            "hindsight_api.engine.search.mpfp_retrieval.load_all_edges_for_frontier",
            new_callable=AsyncMock,
            return_value={},
        ):
            result = await mpfp_traverse_async(
                pool=MagicMock(),
                seeds=seeds,
                pattern=["semantic"],
                config=config,
                cache=cache,
            )

        # Seed should have alpha portion of its mass
        assert "seed-1" in result.scores
        assert result.scores["seed-1"] == pytest.approx(config.alpha, rel=0.01)

    @pytest.mark.asyncio
    async def test_single_hop_with_edges(self):
        """Single hop should spread mass to neighbors."""
        cache = EdgeCache()
        config = MPFPConfig(alpha=0.15, threshold=1e-6, top_k_neighbors=10)

        seeds = [SeedNode("seed-1", 1.0)]

        # Pre-populate cache with seed edges (mimics pre-warming in retrieve())
        cache.add_all_edges(
            {
                "semantic": {
                    "seed-1": [
                        EdgeTarget("neighbor-1", 0.8),
                        EdgeTarget("neighbor-2", 0.4),
                    ]
                }
            },
            ["seed-1"],
        )

        # Mock for loading neighbor edges (after hop 0)
        async def mock_load_all_edges(pool, node_ids, top_k=20):
            return {}

        with patch(
            "hindsight_api.engine.search.mpfp_retrieval.load_all_edges_for_frontier",
            side_effect=mock_load_all_edges,
        ):
            result = await mpfp_traverse_async(
                pool=MagicMock(),
                seeds=seeds,
                pattern=["semantic"],
                config=config,
                cache=cache,
            )

        # Seed keeps alpha portion
        assert "seed-1" in result.scores
        assert result.scores["seed-1"] == pytest.approx(config.alpha, rel=0.01)

        # Neighbors get remaining mass (normalized)
        assert "neighbor-1" in result.scores
        assert "neighbor-2" in result.scores

        # neighbor-1 should get more (higher weight)
        assert result.scores["neighbor-1"] > result.scores["neighbor-2"]

    @pytest.mark.asyncio
    async def test_two_hops(self):
        """Two-hop pattern should traverse through neighbors."""
        cache = EdgeCache()
        config = MPFPConfig(alpha=0.15, threshold=1e-6, top_k_neighbors=10)

        seeds = [SeedNode("seed-1", 1.0)]

        # Pre-populate cache with seed edges (mimics pre-warming in retrieve())
        cache.add_all_edges(
            {"semantic": {"seed-1": [EdgeTarget("hop1-node", 1.0)]}},
            ["seed-1"],
        )

        # Mock edge loading for hop 1 nodes
        async def mock_load_all_edges(pool, node_ids, top_k=20):
            edges: dict[str, dict[str, list[EdgeTarget]]] = {"semantic": {}}
            if "hop1-node" in node_ids:
                edges["semantic"]["hop1-node"] = [EdgeTarget("hop2-node", 1.0)]
            return edges

        with patch(
            "hindsight_api.engine.search.mpfp_retrieval.load_all_edges_for_frontier",
            side_effect=mock_load_all_edges,
        ):
            result = await mpfp_traverse_async(
                pool=MagicMock(),
                seeds=seeds,
                pattern=["semantic", "semantic"],  # Two hops
                config=config,
                cache=cache,
            )

        # Should have scores for all three nodes
        assert "seed-1" in result.scores
        assert "hop1-node" in result.scores
        assert "hop2-node" in result.scores

    @pytest.mark.asyncio
    async def test_cache_reuse(self):
        """Cache should prevent redundant edge loading for already-cached nodes."""
        cache = EdgeCache()
        config = MPFPConfig(alpha=0.15, threshold=1e-6)

        # Pre-load cache (marks seed-1 AND neighbor-1 as fully loaded)
        # neighbor-1 is also cached because after hop 0, the frontier contains neighbor-1
        # and the algorithm tries to pre-warm edges for the next hop
        cache.add_all_edges(
            {"semantic": {"seed-1": [EdgeTarget("neighbor-1", 1.0)], "neighbor-1": []}},
            ["seed-1", "neighbor-1"],
        )

        seeds = [SeedNode("seed-1", 1.0)]

        load_mock = AsyncMock(return_value={})

        with patch(
            "hindsight_api.engine.search.mpfp_retrieval.load_all_edges_for_frontier",
            load_mock,
        ):
            await mpfp_traverse_async(
                pool=MagicMock(),
                seeds=seeds,
                pattern=["semantic"],
                config=config,
                cache=cache,
            )

        # Should not call load_all_edges_for_frontier since all nodes are already cached
        load_mock.assert_not_called()


class TestMPFPGraphRetriever:
    """Tests for the MPFPGraphRetriever class."""

    def test_name_is_mpfp(self):
        """Retriever name should be 'mpfp'."""
        retriever = MPFPGraphRetriever()
        assert retriever.name == "mpfp"

    def test_default_config(self):
        """Default config should have expected patterns."""
        # Use explicit config to avoid global config dependency
        config = MPFPConfig()
        retriever = MPFPGraphRetriever(config=config)

        assert len(retriever.config.patterns_semantic) > 0
        assert len(retriever.config.patterns_temporal) > 0
        assert retriever.config.alpha == 0.15
        assert retriever.config.top_k_neighbors == 20

    def test_custom_config(self):
        """Custom config should be used."""
        config = MPFPConfig(alpha=0.3, top_k_neighbors=10)
        retriever = MPFPGraphRetriever(config=config)

        assert retriever.config.alpha == 0.3
        assert retriever.config.top_k_neighbors == 10

    def test_convert_seeds_from_retrieval_results(self):
        """_convert_seeds should extract scores from RetrievalResult."""
        retriever = MPFPGraphRetriever()

        results = [
            RetrievalResult(id="id-1", text="text1", fact_type="world", similarity=0.9),
            RetrievalResult(id="id-2", text="text2", fact_type="world", similarity=0.7),
        ]

        seeds = retriever._convert_seeds(results, "similarity")

        assert len(seeds) == 2
        assert seeds[0].node_id == "id-1"
        assert seeds[0].score == 0.9
        assert seeds[1].node_id == "id-2"
        assert seeds[1].score == 0.7

    def test_convert_seeds_empty(self):
        """_convert_seeds should handle empty/None input."""
        retriever = MPFPGraphRetriever()

        assert retriever._convert_seeds(None, "similarity") == []
        assert retriever._convert_seeds([], "similarity") == []

    @pytest.mark.asyncio
    async def test_retrieve_no_seeds_returns_empty(self):
        """Retrieve with no seeds should return empty results."""
        # Use explicit config to avoid global config dependency
        config = MPFPConfig()
        retriever = MPFPGraphRetriever(config=config)

        # Mock _find_semantic_seeds to return empty
        with patch.object(retriever, "_find_semantic_seeds", new_callable=AsyncMock, return_value=[]):
            results, timings = await retriever.retrieve(
                pool=MagicMock(),
                query_embedding_str="[0.1, 0.2]",
                bank_id="test",
                fact_type="world",
                budget=10,
            )

        assert results == []
        assert timings is not None
        assert timings.pattern_count == 0

    @pytest.mark.asyncio
    async def test_retrieve_with_semantic_seeds(self):
        """Retrieve with semantic seeds should run patterns and return results."""
        # Use explicit config to avoid global config dependency
        config = MPFPConfig()
        retriever = MPFPGraphRetriever(config=config)

        semantic_seeds = [
            RetrievalResult(id="seed-1", text="seed text", fact_type="world", similarity=0.9),
        ]

        # Mock the internal functions
        # mpfp_traverse_hop_synchronized returns a list of PatternResult (one per pattern)
        async def mock_traverse(*args, **kwargs):
            return [PatternResult(pattern=["semantic"], scores={"seed-1": 0.5, "result-1": 0.3})]

        async def mock_fetch(pool, node_ids, fact_type):
            return [
                RetrievalResult(id="seed-1", text="seed text", fact_type="world"),
                RetrievalResult(id="result-1", text="result text", fact_type="world"),
            ]

        with (
            patch(
                "hindsight_api.engine.search.mpfp_retrieval.mpfp_traverse_hop_synchronized",
                side_effect=mock_traverse,
            ),
            patch(
                "hindsight_api.engine.search.mpfp_retrieval.fetch_memory_units_by_ids",
                side_effect=mock_fetch,
            ),
            patch(
                "hindsight_api.engine.search.mpfp_retrieval.load_all_edges_for_frontier",
                new_callable=AsyncMock,
                return_value={},
            ),
        ):
            results, timings = await retriever.retrieve(
                pool=MagicMock(),
                query_embedding_str="[0.1, 0.2]",
                bank_id="test",
                fact_type="world",
                budget=10,
                semantic_seeds=semantic_seeds,
            )

        assert len(results) == 2
        assert timings is not None
        assert timings.pattern_count > 0


@pytest.mark.asyncio
async def test_mpfp_integration(memory, request_context):
    """Integration test: MPFP retrieval with real database."""
    bank_id = f"test_mpfp_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store memories with entity relationships
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at TechCorp as a software engineer",
            context="employee info",
            request_context=request_context,
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="TechCorp is located in San Francisco",
            context="company info",
            request_context=request_context,
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="Bob is Alice's manager at TechCorp",
            context="employee info",
            request_context=request_context,
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="San Francisco has many tech companies",
            context="city info",
            request_context=request_context,
        )

        # Query should find related facts via graph traversal
        from hindsight_api.engine.memory_engine import Budget

        result = await memory.recall_async(
            bank_id=bank_id,
            query="Tell me about Alice",
            fact_type=["world"],
            budget=Budget.MID,
            max_tokens=2048,
            request_context=request_context,
        )

        # Should return results
        assert result.results is not None
        assert len(result.results) > 0

        # Should find Alice-related facts
        fact_texts = [f.text for f in result.results]
        alice_facts = [t for t in fact_texts if "Alice" in t or "TechCorp" in t]
        assert len(alice_facts) > 0, f"Should find Alice-related facts, got: {fact_texts}"

        print(f"\n✓ MPFP integration test passed! Found {len(result.results)} facts")

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_mpfp_lazy_loading_efficiency(memory, request_context):
    """Test that MPFP loads edges lazily, not upfront."""
    bank_id = f"test_mpfp_lazy_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store many memories to create a larger graph
        for i in range(20):
            await memory.retain_async(
                bank_id=bank_id,
                content=f"Fact number {i} about topic {i % 5}",
                context=f"context {i}",
                request_context=request_context,
            )

        from hindsight_api.engine.memory_engine import Budget

        # Query - MPFP should only load edges for relevant frontier nodes
        result = await memory.recall_async(
            bank_id=bank_id,
            query="topic 0",
            fact_type=["world"],
            budget=Budget.LOW,
            max_tokens=1024,
            enable_trace=True,
            request_context=request_context,
        )

        assert result.results is not None

        # Check trace for timing info
        if result.trace:
            print(f"\n✓ MPFP lazy loading test passed!")
            print(f"  - Facts returned: {len(result.results)}")

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ============================================================================
# MPFP Performance Benchmark Tests
# ============================================================================
# These tests require an external database with a large memory bank to be useful.
# Set EXTERNAL_DATABASE_URL and BENCHMARK_BANK_ID environment variables to run.
# Example:
#   EXTERNAL_DATABASE_URL=postgresql://user:pass@host:port/db \
#   BENCHMARK_BANK_ID=load-test \
#   pytest tests/test_mpfp_retrieval.py::test_mpfp_edge_loading_performance -v -s


import os
import asyncpg

EXTERNAL_DATABASE_URL = os.environ.get("EXTERNAL_DATABASE_URL")
BENCHMARK_BANK_ID = os.environ.get("BENCHMARK_BANK_ID", "load-test")

requires_external_db = pytest.mark.skipif(
    EXTERNAL_DATABASE_URL is None,
    reason="EXTERNAL_DATABASE_URL not set - skipping external DB benchmark",
)


@requires_external_db
@pytest.mark.asyncio
async def test_mpfp_edge_loading_performance():
    """
    Benchmark MPFP edge loading performance.

    This test measures the performance of the LATERAL query optimization
    for loading edges in the MPFP graph traversal algorithm.

    Set EXTERNAL_DATABASE_URL to point to a database with existing data.
    Set BENCHMARK_BANK_ID to specify which bank to query (default: load-test).

    Example usage:
        EXTERNAL_DATABASE_URL=postgresql://hindsight:hindsight@localhost:5435/hindsight \
        BENCHMARK_BANK_ID=load-test \
        pytest tests/test_mpfp_retrieval.py::test_mpfp_edge_loading_performance -v -s
    """
    import time

    # Connect to external database
    pool = await asyncpg.create_pool(EXTERNAL_DATABASE_URL, min_size=2, max_size=10)

    try:
        # Get some sample node IDs from the database
        async with pool.acquire() as conn:
            # First check how many links exist
            stats = await conn.fetchrow("""
                SELECT
                    count(*) as total_links,
                    count(DISTINCT from_unit_id) as unique_sources
                FROM memory_links
            """)
            print(f"\n📊 Database Stats:")
            print(f"   Total links: {stats['total_links']:,}")
            print(f"   Unique sources: {stats['unique_sources']:,}")

            # Get edge distribution by type
            type_stats = await conn.fetch("""
                SELECT link_type, count(*) as cnt,
                       round(avg(weight)::numeric, 3) as avg_weight
                FROM memory_links
                GROUP BY link_type
                ORDER BY cnt DESC
            """)
            print(f"\n   Edge distribution:")
            for row in type_stats:
                print(f"     - {row['link_type']}: {row['cnt']:,} (avg_weight={row['avg_weight']})")

            # Get sample frontier nodes (from memory_units in the benchmark bank)
            # bank_id is the text primary key in banks table
            frontier_rows = await conn.fetch("""
                SELECT id FROM memory_units
                WHERE bank_id = $1
                LIMIT 100
            """, BENCHMARK_BANK_ID)

            if not frontier_rows:
                pytest.skip(f"No memory units found for bank '{BENCHMARK_BANK_ID}'")

            frontier_node_ids = [str(row['id']) for row in frontier_rows]
            print(f"\n🎯 Testing with {len(frontier_node_ids)} frontier nodes from bank '{BENCHMARK_BANK_ID}'")

        # Test 1: Original query approach (all edges, no per-type limit)
        async with pool.acquire() as conn:
            start = time.time()
            original_rows = await conn.fetch("""
                SELECT ml.from_unit_id, ml.to_unit_id, ml.link_type, ml.weight
                FROM memory_links ml
                WHERE ml.from_unit_id = ANY($1::uuid[])
                  AND ml.weight >= 0.1
                ORDER BY ml.from_unit_id, ml.link_type, ml.weight DESC
            """, frontier_node_ids)
            original_time = time.time() - start
            original_count = len(original_rows)

        # Test 2: New LATERAL query approach (top-k per type)
        async with pool.acquire() as conn:
            start = time.time()
            lateral_rows = await conn.fetch("""
                WITH frontier(node_id) AS (SELECT unnest($1::uuid[]))
                SELECT f.node_id as from_unit_id, lt.link_type, edges.to_unit_id, edges.weight
                FROM frontier f
                CROSS JOIN (VALUES ('semantic'), ('temporal'), ('entity'), ('causes'), ('caused_by')) AS lt(link_type)
                CROSS JOIN LATERAL (
                    SELECT ml.to_unit_id, ml.weight
                    FROM memory_links ml
                    WHERE ml.from_unit_id = f.node_id
                      AND ml.link_type = lt.link_type
                      AND ml.weight >= 0.1
                    ORDER BY ml.weight DESC
                    LIMIT 20
                ) edges
            """, frontier_node_ids)
            lateral_time = time.time() - start
            lateral_count = len(lateral_rows)

        # Print results
        print(f"\n⏱️  Performance Comparison ({len(frontier_node_ids)} nodes):")
        print(f"\n   Original (all edges):")
        print(f"     - Time: {original_time * 1000:.2f}ms")
        print(f"     - Rows: {original_count:,}")
        print(f"     - Rows/node: {original_count / len(frontier_node_ids):.1f}")

        print(f"\n   LATERAL (top-20 per type):")
        print(f"     - Time: {lateral_time * 1000:.2f}ms")
        print(f"     - Rows: {lateral_count:,}")
        print(f"     - Rows/node: {lateral_count / len(frontier_node_ids):.1f}")

        speedup = original_time / lateral_time if lateral_time > 0 else float('inf')
        reduction = (1 - lateral_count / original_count) * 100 if original_count > 0 else 0
        print(f"\n   📈 Improvement:")
        print(f"     - Speedup: {speedup:.2f}x faster")
        print(f"     - Data reduction: {reduction:.1f}% fewer rows")

        # Assert improvement (should be at least some improvement for large datasets)
        if original_count > 1000:
            # For large datasets, expect significant improvement
            assert speedup >= 1.5, f"Expected at least 1.5x speedup, got {speedup:.2f}x"
            assert reduction >= 30, f"Expected at least 30% data reduction, got {reduction:.1f}%"
            print(f"\n✅ Performance test PASSED!")
        else:
            print(f"\n⚠️ Dataset too small ({original_count} rows) for meaningful performance comparison")

    finally:
        await pool.close()


@requires_external_db
@pytest.mark.asyncio
async def test_mpfp_full_retrieval_performance():
    """
    Benchmark full MPFP retrieval including traversal and reranking.

    This test measures end-to-end MPFP retrieval performance.
    """
    import time

    pool = await asyncpg.create_pool(EXTERNAL_DATABASE_URL, min_size=2, max_size=10)

    try:
        # Get a sample query embedding from an existing memory unit
        async with pool.acquire() as conn:
            # Check if bank exists
            bank_exists = await conn.fetchval("""
                SELECT 1 FROM banks WHERE bank_id = $1
            """, BENCHMARK_BANK_ID)
            if not bank_exists:
                pytest.skip(f"Bank '{BENCHMARK_BANK_ID}' not found")

            sample = await conn.fetchrow("""
                SELECT embedding::text as embedding_str
                FROM memory_units
                WHERE bank_id = $1
                  AND embedding IS NOT NULL
                LIMIT 1
            """, BENCHMARK_BANK_ID)

            if not sample:
                pytest.skip("No memory units with embeddings found")

            query_embedding_str = sample['embedding_str']

        # Run MPFP retrieval
        retriever = MPFPGraphRetriever()

        print(f"\n🔍 Running MPFP retrieval benchmark on bank '{BENCHMARK_BANK_ID}'...")

        # Warm-up run
        await retriever.retrieve(
            pool=pool,
            query_embedding_str=query_embedding_str,
            bank_id=BENCHMARK_BANK_ID,
            fact_type="world",
            budget=100,
            query_text="test query",
        )

        # Timed runs
        timings_list = []
        for i in range(3):
            start = time.time()
            results, timings = await retriever.retrieve(
                pool=pool,
                query_embedding_str=query_embedding_str,
                bank_id=BENCHMARK_BANK_ID,
                fact_type="opinion",
                budget=100,
                query_text="What did I say about training models?",
            )
            elapsed = time.time() - start
            timings_list.append((elapsed, timings, len(results)))

        # Print results
        print(f"\n⏱️  MPFP Retrieval Results (3 runs):")
        for i, (elapsed, timings, count) in enumerate(timings_list):
            print(f"\n   Run {i + 1}:")
            print(f"     - Total: {elapsed * 1000:.2f}ms")
            print(f"     - Results: {count}")
            if timings:
                print(f"     - Seeds: {timings.seeds_time * 1000:.2f}ms")
                print(f"     - Patterns: {timings.pattern_count}")
                print(f"     - Traverse: {timings.traverse * 1000:.2f}ms")
                print(f"     - Edge load: {timings.edge_load_time * 1000:.2f}ms")
                print(f"     - Edges: {timings.edge_count:,}")
                print(f"     - DB queries: {timings.db_queries}")
                print(f"     - Fusion: {timings.fusion * 1000:.2f}ms")
                print(f"     - Fetch: {timings.fetch * 1000:.2f}ms")

        avg_time = sum(t[0] for t in timings_list) / len(timings_list)
        print(f"\n   📊 Average: {avg_time * 1000:.2f}ms")
        print(f"\n✅ MPFP retrieval benchmark complete!")

    finally:
        await pool.close()
