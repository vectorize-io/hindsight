"""
Search tracer for collecting detailed search execution traces.

The SearchTracer collects comprehensive information about each step
of the spreading activation search process for debugging and visualization.
"""

import time
from datetime import UTC, datetime
from typing import Any, Literal

from .trace import (
    EntryPoint,
    LinkInfo,
    NodeVisit,
    PruningDecision,
    QueryInfo,
    RerankedResult,
    RetrievalMethodResults,
    RetrievalResult,
    RRFMergeResult,
    SearchPhaseMetrics,
    SearchSummary,
    SearchTrace,
    WeightComponents,
)


class SearchTracer:
    """
    Tracer for collecting detailed search execution information.

    Usage:
        tracer = SearchTracer(query="Who is Alice?", budget=50, max_tokens=4096)
        tracer.start()

        # During search...
        tracer.record_query_embedding(embedding)
        tracer.add_entry_point(node_id, text, similarity, rank)
        tracer.visit_node(...)
        tracer.prune_node(...)

        # After search...
        trace = tracer.finalize(final_results)
        json_output = trace.to_json()
    """

    def __init__(self, query: str, budget: int, max_tokens: int):
        """
        Initialize tracer.

        Args:
            query: Search query text
            budget: Maximum nodes to explore
            max_tokens: Maximum tokens to return in results
        """
        self.query_text = query
        self.budget = budget
        self.max_tokens = max_tokens

        # Trace data
        self.query_embedding: list[float] | None = None
        self.start_time: float | None = None
        self.entry_points: list[EntryPoint] = []
        self.visits: list[NodeVisit] = []
        self.pruned: list[PruningDecision] = []
        self.phase_metrics: list[SearchPhaseMetrics] = []

        # New 4-way retrieval tracking
        self.retrieval_results: list[RetrievalMethodResults] = []
        self.rrf_merged: list[RRFMergeResult] = []
        self.reranked: list[RerankedResult] = []

        # Tracking state
        self.current_step = 0
        self.nodes_visited_set = set()  # For quick lookups

        # Link statistics
        self.temporal_links_followed = 0
        self.semantic_links_followed = 0
        self.entity_links_followed = 0

    def start(self):
        """Start timing the search."""
        self.start_time = time.time()

    def record_query_embedding(self, embedding: list[float]):
        """Record the query embedding."""
        self.query_embedding = embedding

    def add_entry_point(self, node_id: str, text: str, similarity: float, rank: int):
        """
        Record an entry point.

        Args:
            node_id: Memory unit ID
            text: Memory unit text
            similarity: Cosine similarity to query
            rank: Rank among entry points (1-based)
        """
        # Clamp similarity to [0.0, 1.0] to handle floating-point precision
        similarity = min(1.0, max(0.0, similarity))

        self.entry_points.append(
            EntryPoint(
                node_id=node_id,
                text=text,
                similarity_score=similarity,
                rank=rank,
            )
        )

    def visit_node(
        self,
        node_id: str,
        text: str,
        context: str,
        event_date: datetime | None,
        access_count: int,
        is_entry_point: bool,
        parent_node_id: str | None,
        link_type: Literal["temporal", "semantic", "entity"] | None,
        link_weight: float | None,
        activation: float,
        semantic_similarity: float,
        recency: float,
        frequency: float,
        final_weight: float,
    ):
        """
        Record visiting a node.

        Args:
            node_id: Memory unit ID
            text: Memory unit text
            context: Memory unit context
            event_date: When the memory occurred
            access_count: Access count before this search
            is_entry_point: Whether this is an entry point
            parent_node_id: Node that led here (None for entry points)
            link_type: Type of link from parent
            link_weight: Weight of link from parent
            activation: Activation score
            semantic_similarity: Semantic similarity to query
            recency: Recency weight
            frequency: Frequency weight
            final_weight: Combined final weight
        """
        self.current_step += 1
        self.nodes_visited_set.add(node_id)

        # Clamp values to handle floating-point precision issues
        # (sometimes normalization produces values like 1.0000005 instead of 1.0)
        semantic_similarity = min(1.0, max(0.0, semantic_similarity))
        recency = min(1.0, max(0.0, recency))
        frequency = min(1.0, max(0.0, frequency))

        # Calculate weight contributions for transparency
        weights = WeightComponents(
            activation=activation,
            semantic_similarity=semantic_similarity,
            recency=recency,
            frequency=frequency,
            final_weight=final_weight,
            activation_contribution=0.3 * activation,
            semantic_contribution=0.3 * semantic_similarity,
            recency_contribution=0.25 * recency,
            frequency_contribution=0.15 * frequency,
        )

        visit = NodeVisit(
            step=self.current_step,
            node_id=node_id,
            text=text,
            context=context,
            event_date=event_date,
            access_count=access_count,
            is_entry_point=is_entry_point,
            parent_node_id=parent_node_id,
            link_type=link_type,
            link_weight=link_weight,
            weights=weights,
            neighbors_explored=[],
            final_rank=None,  # Will be set later
        )

        self.visits.append(visit)

        # Track link statistics
        if link_type == "temporal":
            self.temporal_links_followed += 1
        elif link_type == "semantic":
            self.semantic_links_followed += 1
        elif link_type == "entity":
            self.entity_links_followed += 1

    def add_neighbor_link(
        self,
        from_node_id: str,
        to_node_id: str,
        link_type: Literal["temporal", "semantic", "entity"],
        link_weight: float,
        entity_id: str | None,
        new_activation: float | None,
        followed: bool,
        prune_reason: str | None = None,
        is_supplementary: bool = False,
    ):
        """
        Record a link to a neighbor (whether followed or not).

        Args:
            from_node_id: Source node
            to_node_id: Target node
            link_type: Type of link
            link_weight: Weight of link
            entity_id: Entity ID if link is entity-based
            new_activation: Activation passed to neighbor (None for supplementary links)
            followed: Whether link was followed
            prune_reason: Why link was not followed (if not followed)
            is_supplementary: Whether this is a supplementary link (multiple connections)
        """
        # Find the visit for the source node
        visit = None
        for v in self.visits:
            if v.node_id == from_node_id:
                visit = v
                break

        if visit is None:
            # Node not found, skip
            return

        link_info = LinkInfo(
            to_node_id=to_node_id,
            link_type=link_type,
            link_weight=link_weight,
            entity_id=entity_id,
            new_activation=new_activation,
            followed=followed,
            prune_reason=prune_reason,
            is_supplementary=is_supplementary,
        )

        visit.neighbors_explored.append(link_info)

    def prune_node(
        self,
        node_id: str,
        reason: Literal["already_visited", "activation_too_low", "budget_exhausted"],
        activation: float,
    ):
        """
        Record a node being pruned (not visited).

        Args:
            node_id: Node that was pruned
            reason: Why it was pruned
            activation: Activation value when pruned
        """
        self.pruned.append(
            PruningDecision(
                node_id=node_id,
                reason=reason,
                activation=activation,
                would_have_been_step=self.current_step + 1,
            )
        )

    def add_phase_metric(self, phase_name: str, duration_seconds: float, details: dict[str, Any] | None = None):
        """
        Record metrics for a search phase.

        Args:
            phase_name: Name of the phase
            duration_seconds: Time taken
            details: Additional phase-specific details
        """
        self.phase_metrics.append(
            SearchPhaseMetrics(
                phase_name=phase_name,
                duration_seconds=duration_seconds,
                details=details or {},
            )
        )

    def add_retrieval_results(
        self,
        method_name: Literal["semantic", "bm25", "graph", "temporal"],
        results: list[tuple],  # List of (doc_id, data) tuples
        duration_seconds: float,
        score_field: str,  # e.g., "similarity", "bm25_score"
        metadata: dict[str, Any] | None = None,
        fact_type: str | None = None,
    ):
        """
        Record results from a single retrieval method.

        Args:
            method_name: Name of the retrieval method
            results: List of (doc_id, data) tuples from retrieval
            duration_seconds: Time taken for this retrieval
            score_field: Field name containing the score in data dict
            metadata: Optional metadata about this retrieval method
            fact_type: Fact type this retrieval was for (world, experience, opinion)
        """
        retrieval_results = []
        for rank, (doc_id, data) in enumerate(results, start=1):
            score = data.get(score_field)
            if score is None:
                score = 0.0
            retrieval_results.append(
                RetrievalResult(
                    rank=rank,
                    node_id=doc_id,
                    text=data.get("text", ""),
                    context=data.get("context", ""),
                    event_date=data.get("event_date"),
                    fact_type=data.get("fact_type") or fact_type,
                    score=score,
                    score_name=score_field,
                )
            )

        self.retrieval_results.append(
            RetrievalMethodResults(
                method_name=method_name,
                fact_type=fact_type,
                results=retrieval_results,
                duration_seconds=duration_seconds,
                metadata=metadata or {},
            )
        )

    def add_rrf_merged(self, merged_results: list[tuple]):
        """
        Record RRF merged results.

        Args:
            merged_results: List of (doc_id, data, rrf_meta) tuples from RRF merge
        """
        self.rrf_merged = []
        for rank, (doc_id, data, rrf_meta) in enumerate(merged_results, start=1):
            self.rrf_merged.append(
                RRFMergeResult(
                    node_id=doc_id,
                    text=data.get("text", ""),
                    rrf_score=rrf_meta.get("rrf_score", 0.0),
                    source_ranks=rrf_meta.get("source_ranks", {}),
                    final_rrf_rank=rank,
                )
            )

    def add_reranked(self, reranked_results: list[dict[str, Any]], rrf_merged: list):
        """
        Record reranked results.

        Args:
            reranked_results: List of result dicts after reranking
            rrf_merged: Original RRF merged results for comparison
        """
        # Build map of node_id -> rrf_rank
        rrf_rank_map = {}
        for item in self.rrf_merged:
            rrf_rank_map[item.node_id] = item.final_rrf_rank

        self.reranked = []
        for rank, result in enumerate(reranked_results, start=1):
            node_id = result["id"]
            rrf_rank = rrf_rank_map.get(node_id, len(rrf_merged) + 1)
            rank_change = rrf_rank - rank  # Positive = moved up

            # Extract score components (only include non-None values)
            # Keys from ScoredResult.to_dict(): cross_encoder_score, cross_encoder_score_normalized,
            # rrf_normalized, temporal, recency, combined_score, weight
            score_components = {}
            for key in [
                "cross_encoder_score",
                "cross_encoder_score_normalized",
                "rrf_score",
                "rrf_normalized",
                "temporal",
                "recency",
                "combined_score",
            ]:
                if key in result and result[key] is not None:
                    score_components[key] = result[key]

            self.reranked.append(
                RerankedResult(
                    node_id=node_id,
                    text=result.get("text", ""),
                    rerank_score=result.get("weight", 0.0),
                    rerank_rank=rank,
                    rrf_rank=rrf_rank,
                    rank_change=rank_change,
                    score_components=score_components,
                )
            )

    def finalize(self, final_results: list[dict[str, Any]]) -> SearchTrace:
        """
        Finalize the trace and return the complete SearchTrace object.

        Args:
            final_results: Final ranked results returned to user

        Returns:
            Complete SearchTrace object
        """
        if self.start_time is None:
            raise ValueError("Tracer not started - call start() first")

        total_duration = time.time() - self.start_time

        # Set final ranks on visits based on results
        for rank, result in enumerate(final_results, 1):
            result_node_id = result["id"]
            for visit in self.visits:
                if visit.node_id == result_node_id:
                    visit.final_rank = rank
                    break

        # Create query info
        query_info = QueryInfo(
            query_text=self.query_text,
            query_embedding=self.query_embedding or [],
            timestamp=datetime.now(UTC),
            budget=self.budget,
            max_tokens=self.max_tokens,
        )

        # Create summary
        summary = SearchSummary(
            total_nodes_visited=len(self.visits),
            total_nodes_pruned=len(self.pruned),
            entry_points_found=len(self.entry_points),
            budget_used=len(self.visits),
            budget_remaining=self.budget - len(self.visits),
            total_duration_seconds=total_duration,
            results_returned=len(final_results),
            temporal_links_followed=self.temporal_links_followed,
            semantic_links_followed=self.semantic_links_followed,
            entity_links_followed=self.entity_links_followed,
            phase_metrics=self.phase_metrics,
        )

        # Create complete trace
        trace = SearchTrace(
            query=query_info,
            retrieval_results=self.retrieval_results,
            rrf_merged=self.rrf_merged,
            reranked=self.reranked,
            entry_points=self.entry_points,
            visits=self.visits,
            pruned=self.pruned,
            summary=summary,
            final_results=final_results,
        )

        return trace
