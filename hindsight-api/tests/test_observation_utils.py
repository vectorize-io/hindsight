"""
Unit tests for the 4-step entity summary generation pipeline.

Tests the individual components of observation_utils.py:
1. Facet Discovery
2. Parallel Retrieval
3. Diversity Selection (Clustering + MMR)
4. Synthesis
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from hindsight_api.engine.search.observation_utils import (
    # Step 1
    build_facet_discovery_prompt,
    get_facet_discovery_system_message,
    discover_facets,
    FacetProbesResponse,
    # Step 2
    retrieve_facets_parallel,
    deduplicate_facets,
    Facet,
    # Step 3
    cluster_memories,
    mmr_selection,
    select_diverse_memories,
    # Step 4
    format_memories_for_synthesis,
    build_synthesis_prompt,
    synthesize_summary,
    EntitySummaryResponse,
    # Main pipeline
    generate_entity_summary,
    SummaryResult,
    # Constants
    MAX_PROBES,
    MAX_CLUSTERS,
    MIN_CLUSTERS,
)
from hindsight_api.engine.response_models import MemoryFact


# =============================================================================
# Test Fixtures
# =============================================================================


def create_memory_fact(id: str, text: str, fact_type: str = "world", occurred_start: str = None) -> MemoryFact:
    """Create a MemoryFact for testing."""
    return MemoryFact(
        id=id,
        text=text,
        fact_type=fact_type,
        occurred_start=occurred_start,
    )


def create_random_embedding(dim: int = 384, seed: int = None) -> list[float]:
    """Create a random embedding for testing."""
    if seed is not None:
        np.random.seed(seed)
    emb = np.random.randn(dim).astype(np.float32)
    emb = emb / np.linalg.norm(emb)  # Normalize
    return emb.tolist()


# =============================================================================
# Step 1: Facet Discovery Tests
# =============================================================================


class TestFacetDiscovery:
    """Tests for Step 1: Facet Discovery."""

    def test_build_facet_discovery_prompt(self):
        """Test that the prompt includes the entity name."""
        prompt = build_facet_discovery_prompt("John Doe")
        assert "John Doe" in prompt
        assert "15-20" in prompt  # Should request multiple probes
        assert "diverse" in prompt.lower()

    def test_get_facet_discovery_system_message(self):
        """Test that system message is appropriate."""
        msg = get_facet_discovery_system_message()
        assert "search" in msg.lower() or "query" in msg.lower()
        assert "JSON" in msg

    @pytest.mark.asyncio
    async def test_discover_facets_limits_probes(self):
        """Test that discover_facets limits to MAX_PROBES."""
        mock_llm = MagicMock()

        # Create more probes than MAX_PROBES
        many_probes = [f"probe_{i}" for i in range(30)]
        mock_llm.call = AsyncMock(return_value=FacetProbesResponse(probes=many_probes))

        result = await discover_facets(mock_llm, "Test Entity")

        assert len(result) <= MAX_PROBES
        mock_llm.call.assert_called_once()


# =============================================================================
# Step 2: Parallel Retrieval Tests
# =============================================================================


class TestParallelRetrieval:
    """Tests for Step 2: Parallel Retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_facets_parallel_basic(self):
        """Test basic parallel retrieval."""
        probes = ["probe1", "probe2", "probe3"]

        async def mock_recall(query: str, max_tokens: int):
            # Return different memories for each probe
            mem = create_memory_fact(f"id_{query}", f"Memory for {query}")
            emb = create_random_embedding(seed=hash(query) % 1000)
            return [mem], [emb]

        facets = await retrieve_facets_parallel(probes, mock_recall, total_token_budget=10000)

        assert len(facets) == 3
        for facet in facets:
            assert len(facet.memories) == 1
            assert len(facet.embeddings) == 1

    @pytest.mark.asyncio
    async def test_retrieve_facets_filters_empty(self):
        """Test that empty facets are filtered out."""
        probes = ["probe1", "probe2", "probe3"]

        async def mock_recall(query: str, max_tokens: int):
            if query == "probe2":
                return [], []  # Empty for probe2
            mem = create_memory_fact(f"id_{query}", f"Memory for {query}")
            emb = create_random_embedding(seed=hash(query) % 1000)
            return [mem], [emb]

        facets = await retrieve_facets_parallel(probes, mock_recall, total_token_budget=10000)

        assert len(facets) == 2  # probe2 should be filtered

    @pytest.mark.asyncio
    async def test_retrieve_handles_errors_gracefully(self):
        """Test that errors in individual probes don't fail the whole operation."""
        probes = ["probe1", "probe2"]

        async def mock_recall(query: str, max_tokens: int):
            if query == "probe1":
                raise Exception("Simulated error")
            mem = create_memory_fact(f"id_{query}", f"Memory for {query}")
            emb = create_random_embedding(seed=hash(query) % 1000)
            return [mem], [emb]

        facets = await retrieve_facets_parallel(probes, mock_recall, total_token_budget=10000)

        assert len(facets) == 1
        assert facets[0].probe == "probe2"


class TestDeduplicateFacets:
    """Tests for facet deduplication."""

    def test_deduplicate_merges_overlapping_facets(self):
        """Test that highly overlapping facets are merged."""
        mem1 = create_memory_fact("id1", "Memory 1")
        mem2 = create_memory_fact("id2", "Memory 2")
        mem3 = create_memory_fact("id3", "Memory 3")

        emb1 = create_random_embedding(seed=1)
        emb2 = create_random_embedding(seed=2)
        emb3 = create_random_embedding(seed=3)

        # Facets with high overlap (share mem1 and mem2)
        facets = [
            Facet(probe="probe1", memories=[mem1, mem2], embeddings=[emb1, emb2]),
            Facet(probe="probe2", memories=[mem1, mem2, mem3], embeddings=[emb1, emb2, emb3]),
        ]

        result = deduplicate_facets(facets, overlap_threshold=0.6)

        # Should be merged into 1 facet
        assert len(result) == 1
        # Merged facet should have all 3 unique memories
        assert len(result[0].memories) == 3

    def test_deduplicate_keeps_distinct_facets(self):
        """Test that distinct facets are kept separate."""
        # Create memories with unique IDs
        facet1_mems = [create_memory_fact(f"f1_m{i}", f"Facet1 Memory {i}") for i in range(3)]
        facet2_mems = [create_memory_fact(f"f2_m{i}", f"Facet2 Memory {i}") for i in range(3)]

        facet1_embs = [create_random_embedding(seed=i) for i in range(3)]
        facet2_embs = [create_random_embedding(seed=i + 100) for i in range(3)]

        facets = [
            Facet(probe="probe1", memories=facet1_mems, embeddings=facet1_embs),
            Facet(probe="probe2", memories=facet2_mems, embeddings=facet2_embs),
        ]

        result = deduplicate_facets(facets, overlap_threshold=0.7)

        # Should keep both facets
        assert len(result) == 2

    def test_deduplicate_empty_list(self):
        """Test deduplication with empty list."""
        result = deduplicate_facets([])
        assert result == []


# =============================================================================
# Step 3: Diversity Selection Tests
# =============================================================================


class TestClusterMemories:
    """Tests for clustering."""

    def test_cluster_small_set_returns_all(self):
        """Test that small sets are returned without clustering."""
        memories = [create_memory_fact(f"id{i}", f"Memory {i}") for i in range(3)]
        embeddings = [create_random_embedding(seed=i) for i in range(3)]

        result = cluster_memories(memories, embeddings, max_clusters=50, min_clusters=5)

        # With only 3 memories (less than min_clusters), should return all
        assert len(result) == 3

    def test_cluster_reduces_large_set(self):
        """Test that large sets are reduced via clustering."""
        # Create 100 memories
        memories = [create_memory_fact(f"id{i}", f"Memory {i}") for i in range(100)]
        embeddings = [create_random_embedding(seed=i) for i in range(100)]

        result = cluster_memories(memories, embeddings, max_clusters=20, min_clusters=5)

        # Should reduce to at most max_clusters
        assert len(result) <= 20
        assert len(result) >= 5

    def test_cluster_handles_empty_input(self):
        """Test handling of empty input."""
        result = cluster_memories([], [], max_clusters=50)
        assert result == []

    def test_cluster_handles_mismatched_lengths(self):
        """Test handling of mismatched memory/embedding lengths."""
        memories = [create_memory_fact(f"id{i}", f"Memory {i}") for i in range(10)]
        embeddings = [create_random_embedding(seed=i) for i in range(5)]  # Fewer embeddings

        result = cluster_memories(memories, embeddings, max_clusters=50, min_clusters=5)

        # Should handle gracefully
        assert len(result) <= 5  # Limited by embeddings


class TestMMRSelection:
    """Tests for MMR diversity selection."""

    def test_mmr_basic_selection(self):
        """Test basic MMR selection."""
        memories = [create_memory_fact(f"id{i}", f"Memory {i}") for i in range(10)]
        embeddings = [create_random_embedding(seed=i) for i in range(10)]
        candidates = list(zip(memories, embeddings))
        relevance_scores = [1.0] * 10

        result = mmr_selection(candidates, relevance_scores, target_count=5, lambda_param=0.5)

        assert len(result) == 5
        # Verify all results are MemoryFact objects
        for mem in result:
            assert isinstance(mem, MemoryFact)

    def test_mmr_returns_all_if_target_exceeds_candidates(self):
        """Test that MMR returns all if target > candidates."""
        memories = [create_memory_fact(f"id{i}", f"Memory {i}") for i in range(3)]
        embeddings = [create_random_embedding(seed=i) for i in range(3)]
        candidates = list(zip(memories, embeddings))
        relevance_scores = [1.0] * 3

        result = mmr_selection(candidates, relevance_scores, target_count=10)

        assert len(result) == 3

    def test_mmr_empty_input(self):
        """Test MMR with empty input."""
        result = mmr_selection([], [], target_count=5)
        assert result == []

    def test_mmr_diversity(self):
        """Test that MMR selects diverse items."""
        # Create similar embeddings (low diversity should be penalized)
        base_emb = create_random_embedding(seed=42)

        memories = [create_memory_fact(f"id{i}", f"Memory {i}") for i in range(5)]
        embeddings = []
        for i in range(5):
            if i < 3:
                # First 3 are similar to base
                noise = np.random.randn(384) * 0.1
                emb = np.array(base_emb) + noise
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb.tolist())
            else:
                # Last 2 are different
                embeddings.append(create_random_embedding(seed=i * 100))

        candidates = list(zip(memories, embeddings))
        relevance_scores = [1.0] * 5

        # With lambda=0.3 (favoring diversity), should select more diverse items
        result = mmr_selection(candidates, relevance_scores, target_count=3, lambda_param=0.3)

        assert len(result) == 3


class TestSelectDiverseMemories:
    """Tests for the main diversity selection function."""

    def test_select_diverse_empty_facets(self):
        """Test with empty facets."""
        result = select_diverse_memories([], target_token_budget=10000)
        assert result == {}

    def test_select_diverse_basic(self):
        """Test basic diverse memory selection."""
        mem1 = create_memory_fact("id1", "Memory 1")
        mem2 = create_memory_fact("id2", "Memory 2")
        emb1 = create_random_embedding(seed=1)
        emb2 = create_random_embedding(seed=2)

        facets = [
            Facet(probe="probe1", memories=[mem1, mem2], embeddings=[emb1, emb2])
        ]

        result = select_diverse_memories(facets, target_token_budget=10000)

        assert "probe1" in result
        assert len(result["probe1"]) <= 2


# =============================================================================
# Step 4: Synthesis Tests
# =============================================================================


class TestFormatMemoriesForSynthesis:
    """Tests for memory formatting."""

    def test_format_basic(self):
        """Test basic formatting."""
        mem1 = create_memory_fact("id1", "John works at Google", occurred_start="2024-01-15")
        mem2 = create_memory_fact("id2", "John likes hiking")

        facet_memories = {
            "work info": [mem1],
            "hobbies": [mem2],
        }

        result = format_memories_for_synthesis("John", facet_memories)

        assert "Entity: John" in result
        assert "## Facet: work info" in result
        assert "## Facet: hobbies" in result
        assert "John works at Google" in result
        assert "(when: 2024-01-15)" in result  # Temporal info
        assert "John likes hiking" in result

    def test_format_empty_facets(self):
        """Test formatting with some empty facets."""
        mem1 = create_memory_fact("id1", "Memory 1")

        facet_memories = {
            "facet1": [mem1],
            "facet2": [],  # Empty
        }

        result = format_memories_for_synthesis("Entity", facet_memories)

        assert "facet1" in result
        assert "facet2" not in result  # Empty facet should be skipped


class TestBuildSynthesisPrompt:
    """Tests for synthesis prompt building."""

    def test_build_prompt(self):
        """Test synthesis prompt."""
        prompt = build_synthesis_prompt("John Doe", "some formatted memories", 2000)

        assert "John Doe" in prompt
        assert "some formatted memories" in prompt
        assert "2000" in prompt  # Target tokens


class TestSynthesizeSummary:
    """Tests for summary synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_basic(self):
        """Test basic synthesis."""
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=EntitySummaryResponse(summary="Test summary about John."))

        mem1 = create_memory_fact("id1", "John works at Google")
        facet_memories = {"work": [mem1]}

        result = await synthesize_summary(mock_llm, "John", facet_memories, target_tokens=2000)

        assert result == "Test summary about John."
        mock_llm.call.assert_called_once()


# =============================================================================
# Main Pipeline Tests
# =============================================================================


class TestGenerateEntitySummary:
    """Tests for the main pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_no_probes(self):
        """Test that pipeline returns None if no probes generated."""
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=FacetProbesResponse(probes=[]))

        async def mock_recall(query: str, max_tokens: int):
            return [], []

        result = await generate_entity_summary(mock_llm, "Entity", mock_recall)

        assert isinstance(result, SummaryResult)
        assert result.summary is None
        assert len(result.log_buffer) > 0  # Should have log output

    @pytest.mark.asyncio
    async def test_pipeline_no_memories(self):
        """Test that pipeline returns None if no memories found."""
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=FacetProbesResponse(probes=["probe1", "probe2"]))

        async def mock_recall(query: str, max_tokens: int):
            return [], []  # Always empty

        result = await generate_entity_summary(mock_llm, "Entity", mock_recall)

        assert isinstance(result, SummaryResult)
        assert result.summary is None
        assert len(result.log_buffer) > 0

    @pytest.mark.asyncio
    async def test_pipeline_full(self):
        """Test the full pipeline end-to-end."""
        # Create mock memories with embeddings
        memories = {
            "probe1": [create_memory_fact("id1", "John works at Google")],
            "probe2": [create_memory_fact("id2", "John likes hiking")],
        }
        embeddings = {
            "probe1": [create_random_embedding(seed=1)],
            "probe2": [create_random_embedding(seed=2)],
        }

        mock_llm = MagicMock()

        # First call: facet discovery
        # Second call: synthesis
        mock_llm.call = AsyncMock(side_effect=[
            FacetProbesResponse(probes=["probe1", "probe2"]),
            EntitySummaryResponse(summary="John is a Google employee who enjoys hiking."),
        ])

        async def mock_recall(query: str, max_tokens: int):
            return memories.get(query, []), embeddings.get(query, [])

        result = await generate_entity_summary(mock_llm, "John", mock_recall)

        assert isinstance(result, SummaryResult)
        assert result.summary is not None
        assert "John" in result.summary or "Google" in result.summary or "hiking" in result.summary
        # Should have called LLM twice (discovery + synthesis)
        assert mock_llm.call.call_count == 2
        # Should have log buffer with timing info
        assert len(result.log_buffer) > 0
        assert result.duration > 0

    @pytest.mark.asyncio
    async def test_pipeline_handles_llm_error(self):
        """Test that pipeline handles LLM errors."""
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(side_effect=Exception("LLM error"))

        async def mock_recall(query: str, max_tokens: int):
            mem = create_memory_fact("id1", "Memory 1")
            emb = create_random_embedding(seed=1)
            return [mem], [emb]

        result = await generate_entity_summary(mock_llm, "Entity", mock_recall)

        assert isinstance(result, SummaryResult)
        assert result.summary is None  # Should return None on error, not raise
        assert "ERROR" in "\n".join(result.log_buffer)  # Should log error
