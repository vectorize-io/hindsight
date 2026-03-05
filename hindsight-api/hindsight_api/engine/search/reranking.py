"""
Cross-encoder neural reranking for search results.
"""

from datetime import datetime, timezone

from .types import MergedCandidate, ScoredResult

UTC = timezone.utc

# Multiplicative boost alphas for recency and temporal proximity.
# Each signal contributes at most ±(alpha/2) relative adjustment to the base CE score,
# so the max combined boost is (1 + alpha/2)^2 ≈ +21% and min is (1 - alpha/2)^2 ≈ -19%.
_RECENCY_ALPHA: float = 0.2
_TEMPORAL_ALPHA: float = 0.2


def apply_combined_scoring(
    scored_results: list[ScoredResult],
    now: datetime,
    recency_alpha: float = _RECENCY_ALPHA,
    temporal_alpha: float = _TEMPORAL_ALPHA,
) -> None:
    """Apply combined scoring to a list of ScoredResults in-place.

    Uses the cross-encoder score as the primary relevance signal, with recency
    and temporal proximity applied as multiplicative boosts. This ensures the
    influence of these secondary signals is always proportional to the base
    relevance score, regardless of the cross-encoder model's score calibration.

    Formula::

        recency_boost  = 1 + recency_alpha  * (recency  - 0.5)   # in [1-α/2, 1+α/2]
        temporal_boost = 1 + temporal_alpha * (temporal - 0.5)   # in [1-α/2, 1+α/2]
        combined_score = cross_encoder_score_normalized * recency_boost * temporal_boost

    Temporal proximity is treated as neutral (0.5) when not set by temporal retrieval,
    so temporal_boost collapses to 1.0 for non-temporal queries.

    Args:
        scored_results: Results from the cross-encoder reranker. Mutated in place.
        now: Current UTC datetime for recency calculation.
        recency_alpha: Max relative recency adjustment (default 0.2 → ±10%).
        temporal_alpha: Max relative temporal adjustment (default 0.2 → ±10%).
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    for sr in scored_results:
        # Recency: linear decay over 365 days → [0.1, 1.0]; neutral 0.5 if no date.
        sr.recency = 0.5
        if sr.retrieval.occurred_start:
            occurred = sr.retrieval.occurred_start
            if occurred.tzinfo is None:
                occurred = occurred.replace(tzinfo=UTC)
            days_ago = (now - occurred).total_seconds() / 86400
            sr.recency = max(0.1, min(1.0, 1.0 - (days_ago / 365)))

        # Temporal proximity: meaningful only for temporal queries; neutral otherwise.
        sr.temporal = sr.retrieval.temporal_proximity if sr.retrieval.temporal_proximity is not None else 0.5

        # RRF: kept at 0.0 for trace continuity but excluded from scoring.
        # RRF is batch-relative (min-max normalised) and redundant after reranking.
        sr.rrf_normalized = 0.0

        recency_boost = 1.0 + recency_alpha * (sr.recency - 0.5)
        temporal_boost = 1.0 + temporal_alpha * (sr.temporal - 0.5)
        sr.combined_score = sr.cross_encoder_score_normalized * recency_boost * temporal_boost
        sr.weight = sr.combined_score


class CrossEncoderReranker:
    """
    Neural reranking using a cross-encoder model.

    Configured via environment variables (see cross_encoder.py).
    Default local model is cross-encoder/ms-marco-MiniLM-L-6-v2.
    """

    def __init__(self, cross_encoder=None):
        """
        Initialize cross-encoder reranker.

        Args:
            cross_encoder: CrossEncoderModel instance. If None, creates one from
                          environment variables (defaults to local provider)
        """
        if cross_encoder is None:
            from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env

            cross_encoder = create_cross_encoder_from_env()
        self.cross_encoder = cross_encoder
        self._initialized = False

    async def ensure_initialized(self):
        """Ensure the cross-encoder model is initialized (for lazy initialization)."""
        if self._initialized:
            return

        import asyncio

        cross_encoder = self.cross_encoder
        # For local providers, run in thread pool to avoid blocking event loop
        if cross_encoder.provider_name == "local":
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: asyncio.run(cross_encoder.initialize()))
        else:
            await cross_encoder.initialize()
        self._initialized = True

    async def rerank(self, query: str, candidates: list[MergedCandidate]) -> list[ScoredResult]:
        """
        Rerank candidates using cross-encoder scores.

        Args:
            query: Search query
            candidates: Merged candidates from RRF

        Returns:
            List of ScoredResult objects sorted by cross-encoder score
        """
        if not candidates:
            return []

        # Prepare query-document pairs with date information
        pairs = []
        for candidate in candidates:
            retrieval = candidate.retrieval

            # Use text + context for better ranking
            doc_text = retrieval.text
            if retrieval.context:
                doc_text = f"{retrieval.context}: {doc_text}"

            # Add formatted date information for temporal awareness
            if retrieval.occurred_start:
                occurred_start = retrieval.occurred_start

                # Format in two styles for better model understanding
                # 1. ISO format: YYYY-MM-DD
                date_iso = occurred_start.strftime("%Y-%m-%d")

                # 2. Human-readable: "June 5, 2022"
                date_readable = occurred_start.strftime("%B %d, %Y")

                # Prepend date to document text
                doc_text = f"[Date: {date_readable} ({date_iso})] {doc_text}"

            pairs.append([query, doc_text])

        # Get cross-encoder scores
        scores = await self.cross_encoder.predict(pairs)

        # Normalize scores using sigmoid to [0, 1] range
        # Cross-encoder returns logits which can be negative
        import numpy as np

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        normalized_scores = [sigmoid(score) for score in scores]

        # Create ScoredResult objects with cross-encoder scores
        scored_results = []
        for candidate, raw_score, norm_score in zip(candidates, scores, normalized_scores):
            scored_result = ScoredResult(
                candidate=candidate,
                cross_encoder_score=float(raw_score),
                cross_encoder_score_normalized=float(norm_score),
                weight=float(norm_score),  # Initial weight is just cross-encoder score
            )
            scored_results.append(scored_result)

        # Sort by cross-encoder score
        scored_results.sort(key=lambda x: x.weight, reverse=True)

        return scored_results
