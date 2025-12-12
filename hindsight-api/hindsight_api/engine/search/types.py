"""
Type definitions for the recall pipeline.

These dataclasses replace Dict[str, Any] types throughout the recall pipeline,
providing type safety and making data flow explicit.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class RetrievalResult:
    """
    Result from a single retrieval method (semantic, BM25, graph, or temporal).

    This represents a raw result from the database query, before merging or reranking.
    """
    id: str
    text: str
    fact_type: str
    context: Optional[str] = None
    event_date: Optional[datetime] = None
    occurred_start: Optional[datetime] = None
    occurred_end: Optional[datetime] = None
    mentioned_at: Optional[datetime] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    access_count: int = 0
    embedding: Optional[List[float]] = None

    # Retrieval-specific scores (only one will be set depending on retrieval method)
    similarity: Optional[float] = None  # Semantic retrieval
    bm25_score: Optional[float] = None  # BM25 retrieval
    activation: Optional[float] = None  # Graph retrieval (spreading activation)
    temporal_score: Optional[float] = None  # Temporal retrieval
    temporal_proximity: Optional[float] = None  # Temporal retrieval

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "RetrievalResult":
        """Create from a database row (asyncpg Record converted to dict)."""
        return cls(
            id=str(row["id"]),
            text=row["text"],
            fact_type=row["fact_type"],
            context=row.get("context"),
            event_date=row.get("event_date"),
            occurred_start=row.get("occurred_start"),
            occurred_end=row.get("occurred_end"),
            mentioned_at=row.get("mentioned_at"),
            document_id=row.get("document_id"),
            chunk_id=row.get("chunk_id"),
            access_count=row.get("access_count", 0),
            embedding=row.get("embedding"),
            similarity=row.get("similarity"),
            bm25_score=row.get("bm25_score"),
            activation=row.get("activation"),
            temporal_score=row.get("temporal_score"),
            temporal_proximity=row.get("temporal_proximity"),
        )


@dataclass
class MergedCandidate:
    """
    Candidate after RRF merge of multiple retrieval results.

    Contains the original retrieval data plus RRF metadata.
    """
    # Original retrieval data
    retrieval: RetrievalResult

    # RRF metadata
    rrf_score: float
    rrf_rank: int = 0
    source_ranks: Dict[str, int] = field(default_factory=dict)  # method_name -> rank

    @property
    def id(self) -> str:
        """Convenience property to access ID."""
        return self.retrieval.id


@dataclass
class ScoredResult:
    """
    Result after reranking and scoring.

    Contains all retrieval/merge data plus reranking scores and combined score.
    """
    # Original merged candidate
    candidate: MergedCandidate

    # Reranking scores
    cross_encoder_score: float = 0.0
    cross_encoder_score_normalized: float = 0.0

    # Normalized component scores
    rrf_normalized: float = 0.0
    recency: float = 0.5
    temporal: float = 0.5

    # Final combined score
    combined_score: float = 0.0
    weight: float = 0.0  # Final weight used for ranking

    @property
    def id(self) -> str:
        """Convenience property to access ID."""
        return self.candidate.id

    @property
    def retrieval(self) -> RetrievalResult:
        """Convenience property to access retrieval data."""
        return self.candidate.retrieval

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dict for backwards compatibility.

        This is used during the transition period and for serialization.
        """
        # Start with retrieval data
        result = {
            "id": self.retrieval.id,
            "text": self.retrieval.text,
            "fact_type": self.retrieval.fact_type,
            "context": self.retrieval.context,
            "event_date": self.retrieval.event_date,
            "occurred_start": self.retrieval.occurred_start,
            "occurred_end": self.retrieval.occurred_end,
            "mentioned_at": self.retrieval.mentioned_at,
            "document_id": self.retrieval.document_id,
            "chunk_id": self.retrieval.chunk_id,
            "access_count": self.retrieval.access_count,
            "embedding": self.retrieval.embedding,
            "semantic_similarity": self.retrieval.similarity,
            "bm25_score": self.retrieval.bm25_score,
        }

        # Add temporal scores if present
        if self.retrieval.temporal_score is not None:
            result["temporal_score"] = self.retrieval.temporal_score
        if self.retrieval.temporal_proximity is not None:
            result["temporal_proximity"] = self.retrieval.temporal_proximity

        # Add RRF metadata
        result["rrf_score"] = self.candidate.rrf_score
        result["rrf_rank"] = self.candidate.rrf_rank
        result.update(self.candidate.source_ranks)

        # Add reranking scores
        result["cross_encoder_score"] = self.cross_encoder_score
        result["cross_encoder_score_normalized"] = self.cross_encoder_score_normalized
        result["rrf_normalized"] = self.rrf_normalized
        result["temporal"] = self.temporal
        result["recency"] = self.recency
        result["combined_score"] = self.combined_score
        result["weight"] = self.weight
        result["activation"] = self.weight  # Legacy field

        return result
