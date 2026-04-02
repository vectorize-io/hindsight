"""
Graph retrieval strategies for memory recall.

This module provides an abstraction for graph-based memory retrieval,
allowing different algorithms to be swapped without changing the rest
of the recall pipeline.
"""

import logging
from abc import ABC, abstractmethod

from .tags import TagGroup, TagsMatch
from .types import GraphRetrievalTimings, RetrievalResult

logger = logging.getLogger(__name__)


class GraphRetriever(ABC):
    """
    Abstract base class for graph-based memory retrieval.

    Implementations traverse the memory graph (entity links, temporal links,
    causal links) to find relevant facts that might not be found by
    semantic or keyword search alone.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return identifier for this retrieval strategy (e.g., 'link_expansion')."""
        pass

    @abstractmethod
    async def retrieve(
        self,
        pool,
        query_embedding_str: str,
        bank_id: str,
        fact_type: str,
        budget: int,
        query_text: str | None = None,
        semantic_seeds: list[RetrievalResult] | None = None,
        temporal_seeds: list[RetrievalResult] | None = None,
        adjacency=None,  # TypedAdjacency, optional pre-loaded graph
        tags: list[str] | None = None,  # Visibility scope tags for filtering
        tags_match: TagsMatch = "any",  # How to match tags: 'any' (OR) or 'all' (AND)
        tag_groups: list[TagGroup] | None = None,  # Compound boolean tag filter groups
    ) -> tuple[list[RetrievalResult], GraphRetrievalTimings | None]:
        """
        Retrieve relevant facts via graph traversal.

        Args:
            pool: Database connection pool
            query_embedding_str: Query embedding as string (for finding entry points)
            bank_id: Memory bank identifier
            fact_type: Fact type to filter ('world', 'experience', 'opinion', 'observation')
            budget: Maximum number of nodes to explore/return
            query_text: Original query text (optional, for some strategies)
            semantic_seeds: Pre-computed semantic entry points (from semantic retrieval)
            temporal_seeds: Pre-computed temporal entry points (from temporal retrieval)
            adjacency: Pre-loaded typed adjacency graph (optional)
            tags: Optional list of tags for visibility filtering (OR matching)

        Returns:
            Tuple of (List of RetrievalResult with activation scores, optional timing info)
        """
        pass
