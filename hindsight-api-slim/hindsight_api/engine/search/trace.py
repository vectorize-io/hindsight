"""
Search trace models for debugging and visualization.

These Pydantic models define the structure of search traces, capturing
every step of the spreading activation search process for analysis.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TemporalConstraint(BaseModel):
    """Detected temporal constraint from query analysis."""

    start: datetime | None = Field(default=None, description="Start of temporal range")
    end: datetime | None = Field(default=None, description="End of temporal range")


class QueryInfo(BaseModel):
    """Information about the search query."""

    query_text: str = Field(description="Original query text")
    query_embedding: list[float] = Field(description="Generated query embedding vector")
    timestamp: datetime = Field(description="When the query was executed")
    budget: int = Field(description="Maximum nodes to explore")
    max_tokens: int = Field(description="Maximum tokens to return in results")
    tags: list[str] | None = Field(default=None, description="Tags filter applied to recall")
    tags_match: str | None = Field(default=None, description="Tags matching mode: any, all, any_strict, all_strict")
    temporal_constraint: TemporalConstraint | None = Field(
        default=None, description="Detected temporal range from query"
    )


class EntryPoint(BaseModel):
    """An entry point node selected for search."""

    node_id: str = Field(description="Memory unit ID")
    text: str = Field(description="Memory unit text content")
    similarity_score: float = Field(description="Cosine similarity to query", ge=0.0, le=1.0)
    rank: int = Field(description="Rank among entry points (1-based)")


class WeightComponents(BaseModel):
    """Breakdown of weight calculation components."""

    activation: float = Field(description="Activation from spreading (can exceed 1.0 through accumulation)", ge=0.0)
    semantic_similarity: float = Field(description="Semantic similarity to query", ge=0.0, le=1.0)
    recency: float = Field(description="Recency weight", ge=0.0, le=1.0)
    frequency: float = Field(description="Normalized frequency weight", ge=0.0, le=1.0)
    final_weight: float = Field(description="Combined final weight")

    # Weight formula components (for transparency)
    activation_contribution: float = Field(description="0.3 * activation")
    semantic_contribution: float = Field(description="0.3 * semantic_similarity")
    recency_contribution: float = Field(description="0.25 * recency")
    frequency_contribution: float = Field(description="0.15 * frequency")


class NodeVisit(BaseModel):
    """Information about visiting a node during search."""

    step: int = Field(description="Step number in search (1-based)")
    node_id: str = Field(description="Memory unit ID")
    text: str = Field(description="Memory unit text content")
    context: str = Field(description="Memory unit context")
    event_date: datetime | None = Field(default=None, description="When the memory occurred")

    # How this node was reached
    is_entry_point: bool = Field(description="Whether this is an entry point")

    # Weights
    weights: WeightComponents = Field(description="Weight calculation breakdown")

    # Ranking
    final_rank: int | None = Field(default=None, description="Final rank in results (1-based, None if not in top-k)")


class SearchPhaseMetrics(BaseModel):
    """Performance metrics for a search phase."""

    phase_name: str = Field(description="Name of the phase")
    duration_seconds: float = Field(description="Time taken in seconds")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional phase-specific metrics")


class RetrievalResult(BaseModel):
    """A single result from a retrieval method."""

    rank: int = Field(description="Rank in this retrieval method (1-based)")
    node_id: str = Field(description="Memory unit ID")
    text: str = Field(description="Memory unit text content")
    context: str = Field(default="", description="Memory unit context")
    event_date: datetime | None = Field(default=None, description="When the memory occurred")
    fact_type: str | None = Field(default=None, description="Fact type (world, experience)")
    score: float = Field(description="Score from this retrieval method")
    score_name: str = Field(description="Name of the score (e.g., 'similarity', 'bm25_score', 'activation')")


class RetrievalMethodResults(BaseModel):
    """Results from a single retrieval method."""

    method_name: Literal["semantic", "bm25", "graph", "temporal"] = Field(description="Name of retrieval method")
    fact_type: str | None = Field(default=None, description="Fact type this retrieval was for (world, experience)")
    results: list[RetrievalResult] = Field(description="Retrieved results with ranks")
    duration_seconds: float = Field(description="Time taken for this retrieval")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Method-specific metadata")


class RRFMergeResult(BaseModel):
    """A result after RRF merging."""

    node_id: str = Field(description="Memory unit ID")
    text: str = Field(description="Memory unit text content")
    rrf_score: float = Field(description="Reciprocal Rank Fusion score")
    source_ranks: dict[str, int] = Field(description="Rank in each source that contributed (method_name -> rank)")
    final_rrf_rank: int = Field(description="Rank after RRF merge (1-based)")


class RerankedResult(BaseModel):
    """A result after reranking."""

    node_id: str = Field(description="Memory unit ID")
    text: str = Field(description="Memory unit text content")
    rerank_score: float = Field(description="Final reranking score")
    rerank_rank: int = Field(description="Rank after reranking (1-based)")
    rrf_rank: int = Field(description="Original RRF rank before reranking")
    rank_change: int = Field(description="Change in rank (positive = moved up)")
    score_components: dict[str, float] = Field(default_factory=dict, description="Score breakdown")


class SearchSummary(BaseModel):
    """Summary statistics about the search."""

    total_nodes_visited: int = Field(description="Total nodes visited")
    entry_points_found: int = Field(description="Number of entry points")
    budget_used: int = Field(description="Nodes scored against the budget")
    budget_remaining: int = Field(description="Budget left after scoring")
    total_duration_seconds: float = Field(description="Total search duration")
    results_returned: int = Field(description="Number of results returned")

    # Phase timings
    phase_metrics: list[SearchPhaseMetrics] = Field(default_factory=list, description="Metrics for each phase")


class SearchTrace(BaseModel):
    """Complete trace of a search operation."""

    query: QueryInfo = Field(description="Query information")

    # New 4-way retrieval architecture
    retrieval_results: list[RetrievalMethodResults] = Field(
        default_factory=list, description="Results from each retrieval method"
    )
    rrf_merged: list[RRFMergeResult] = Field(default_factory=list, description="Results after RRF merging")
    reranked: list[RerankedResult] = Field(default_factory=list, description="Results after reranking")

    # Legacy fields (kept for backward compatibility with graph/temporal visualizations)
    entry_points: list[EntryPoint] = Field(
        default_factory=list, description="Entry points selected for search (legacy)"
    )
    visits: list[NodeVisit] = Field(
        default_factory=list, description="All nodes visited during search (legacy, for graph viz)"
    )

    summary: SearchSummary = Field(description="Summary statistics")

    # Final results (for comparison with visits)
    final_results: list[dict[str, Any]] = Field(description="Final ranked results returned to user")

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}

    def to_json(self, **kwargs) -> str:
        """Export trace as JSON string."""
        return self.model_dump_json(indent=2, **kwargs)

    def to_dict(self) -> dict:
        """Export trace as dictionary."""
        return self.model_dump()

    def get_visit_by_node_id(self, node_id: str) -> NodeVisit | None:
        """Find a visit by node ID."""
        for visit in self.visits:
            if visit.node_id == node_id:
                return visit
        return None
