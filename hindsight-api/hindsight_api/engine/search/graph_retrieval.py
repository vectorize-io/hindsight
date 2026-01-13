"""
Graph retrieval strategies for memory recall.

This module provides an abstraction for graph-based memory retrieval,
allowing different algorithms (BFS spreading activation, PPR, etc.) to be
swapped without changing the rest of the recall pipeline.
"""

import logging
from abc import ABC, abstractmethod

from ..db_utils import acquire_with_retry
from ..memory_engine import fq_table
from .tags import TagsMatch, filter_results_by_tags
from .types import MPFPTimings, RetrievalResult

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
        """Return identifier for this retrieval strategy (e.g., 'bfs', 'mpfp')."""
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
    ) -> tuple[list[RetrievalResult], MPFPTimings | None]:
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
            adjacency: Pre-loaded typed adjacency graph (optional, for MPFP)
            tags: Optional list of tags for visibility filtering (OR matching)

        Returns:
            Tuple of (List of RetrievalResult with activation scores, optional timing info)
        """
        pass


class BFSGraphRetriever(GraphRetriever):
    """
    Graph retrieval using BFS-style spreading activation.

    Starting from semantic entry points, spreads activation through
    the memory graph (entity, temporal, causal links) using breadth-first
    traversal with decaying activation.

    This is the original Hindsight graph retrieval algorithm.
    """

    def __init__(
        self,
        entry_point_limit: int = 5,
        entry_point_threshold: float = 0.5,
        activation_decay: float = 0.8,
        min_activation: float = 0.1,
        batch_size: int = 20,
    ):
        """
        Initialize BFS graph retriever.

        Args:
            entry_point_limit: Maximum number of entry points to start from
            entry_point_threshold: Minimum semantic similarity for entry points
            activation_decay: Decay factor per hop (activation *= decay)
            min_activation: Minimum activation to continue spreading
            batch_size: Number of nodes to process per batch (for neighbor fetching)
        """
        self.entry_point_limit = entry_point_limit
        self.entry_point_threshold = entry_point_threshold
        self.activation_decay = activation_decay
        self.min_activation = min_activation
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return "bfs"

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
        adjacency=None,  # Not used by BFS
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
    ) -> tuple[list[RetrievalResult], MPFPTimings | None]:
        """
        Retrieve facts using BFS spreading activation.

        Algorithm:
        1. Find entry points (top semantic matches above threshold)
        2. BFS traversal: visit neighbors, propagate decaying activation
        3. Boost causal links (causes, enables, prevents)
        4. Return visited nodes up to budget

        Note: BFS finds its own entry points via embedding search.
        The semantic_seeds, temporal_seeds, and adjacency parameters are accepted
        for interface compatibility but not used.
        """
        async with acquire_with_retry(pool) as conn:
            results = await self._retrieve_with_conn(
                conn, query_embedding_str, bank_id, fact_type, budget, tags=tags, tags_match=tags_match
            )
            return results, None

    async def _retrieve_with_conn(
        self,
        conn,
        query_embedding_str: str,
        bank_id: str,
        fact_type: str,
        budget: int,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
    ) -> list[RetrievalResult]:
        """Internal implementation with connection."""
        from .tags import build_tags_where_clause_simple

        tags_clause = build_tags_where_clause_simple(tags, 6, match=tags_match)
        params = [query_embedding_str, bank_id, fact_type, self.entry_point_threshold, self.entry_point_limit]
        if tags:
            params.append(tags)

        # Step 1: Find entry points
        entry_points = await conn.fetch(
            f"""
            SELECT id, text, context, event_date, occurred_start, occurred_end,
                   mentioned_at, access_count, embedding, fact_type, document_id, chunk_id, tags,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM {fq_table("memory_units")}
            WHERE bank_id = $2
              AND embedding IS NOT NULL
              AND fact_type = $3
              AND (1 - (embedding <=> $1::vector)) >= $4
              {tags_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $5
            """,
            *params,
        )

        if not entry_points:
            logger.debug(
                f"[BFS] No entry points found for fact_type={fact_type} (tags={tags}, tags_match={tags_match})"
            )
            return []

        logger.debug(
            f"[BFS] Found {len(entry_points)} entry points for fact_type={fact_type} "
            f"(tags={tags}, tags_match={tags_match})"
        )

        # Step 2: BFS spreading activation
        visited = set()
        results = []
        queue = [(RetrievalResult.from_db_row(dict(r)), r["similarity"]) for r in entry_points]
        budget_remaining = budget

        while queue and budget_remaining > 0:
            # Collect a batch of nodes to process
            batch_nodes = []
            batch_activations = {}

            while queue and len(batch_nodes) < self.batch_size and budget_remaining > 0:
                current, activation = queue.pop(0)
                unit_id = current.id

                if unit_id not in visited:
                    visited.add(unit_id)
                    budget_remaining -= 1
                    current.activation = activation
                    results.append(current)
                    batch_nodes.append(current.id)
                    batch_activations[unit_id] = activation

            # Batch fetch neighbors
            if batch_nodes and budget_remaining > 0:
                max_neighbors = len(batch_nodes) * 20
                neighbors = await conn.fetch(
                    f"""
                    SELECT mu.id, mu.text, mu.context, mu.occurred_start, mu.occurred_end,
                           mu.mentioned_at, mu.access_count, mu.embedding, mu.fact_type,
                           mu.document_id, mu.chunk_id, mu.tags,
                           ml.weight, ml.link_type, ml.from_unit_id
                    FROM {fq_table("memory_links")} ml
                    JOIN {fq_table("memory_units")} mu ON ml.to_unit_id = mu.id
                    WHERE ml.from_unit_id = ANY($1::uuid[])
                      AND ml.weight >= $2
                      AND mu.fact_type = $3
                    ORDER BY ml.weight DESC
                    LIMIT $4
                    """,
                    batch_nodes,
                    self.min_activation,
                    fact_type,
                    max_neighbors,
                )

                for n in neighbors:
                    neighbor_id = str(n["id"])
                    if neighbor_id not in visited:
                        parent_id = str(n["from_unit_id"])
                        parent_activation = batch_activations.get(parent_id, 0.5)

                        # Boost causal links
                        link_type = n["link_type"]
                        base_weight = n["weight"]

                        if link_type in ("causes", "caused_by"):
                            causal_boost = 2.0
                        elif link_type in ("enables", "prevents"):
                            causal_boost = 1.5
                        else:
                            causal_boost = 1.0

                        effective_weight = base_weight * causal_boost
                        new_activation = parent_activation * effective_weight * self.activation_decay

                        if new_activation > self.min_activation:
                            neighbor_result = RetrievalResult.from_db_row(dict(n))
                            queue.append((neighbor_result, new_activation))

        # Apply tags filtering (BFS may traverse into memories that don't match tags criteria)
        if tags:
            results = filter_results_by_tags(results, tags, match=tags_match)

        return results
