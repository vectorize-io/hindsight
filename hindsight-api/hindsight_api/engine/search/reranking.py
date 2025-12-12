"""
Cross-encoder neural reranking for search results.
"""

from typing import List
from .types import MergedCandidate, ScoredResult


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

    def rerank(
        self,
        query: str,
        candidates: List[MergedCandidate]
    ) -> List[ScoredResult]:
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
        scores = self.cross_encoder.predict(pairs)

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
                weight=float(norm_score)  # Initial weight is just cross-encoder score
            )
            scored_results.append(scored_result)

        # Sort by cross-encoder score
        scored_results.sort(key=lambda x: x.weight, reverse=True)

        return scored_results
