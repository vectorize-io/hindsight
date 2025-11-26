"""
Cross-encoder neural reranking for search results.
"""

from typing import List, Dict, Any


class CrossEncoderReranker:
    """
    Neural reranking using a cross-encoder model.

    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 by default:
    - Fast inference (~80ms for 100 pairs on CPU)
    - Small model (80MB)
    - Trained for passage re-ranking
    """

    def __init__(self, cross_encoder=None):
        """
        Initialize cross-encoder reranker.

        Args:
            cross_encoder: CrossEncoderReranker instance. If None, uses default
                          SentenceTransformersCrossEncoder with ms-marco-MiniLM-L-6-v2
        """
        if cross_encoder is None:
            from hindsight_api.engine.cross_encoder import SentenceTransformersCrossEncoder
            cross_encoder = SentenceTransformersCrossEncoder()
        self.cross_encoder = cross_encoder

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank using cross-encoder scores."""
        if not candidates:
            return candidates

        # Prepare query-document pairs with date information
        pairs = []
        for c in candidates:
            # Use text + context for better ranking
            doc_text = c["text"]
            if c.get("context"):
                doc_text = f"{c['context']}: {doc_text}"

            # Add formatted date information for temporal awareness
            if c.get("occurred_start"):
                occurred_start = c["occurred_start"]

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

        # Assign normalized scores to candidates
        for c, raw_score, norm_score in zip(candidates, scores, normalized_scores):
            c["weight"] = float(norm_score)
            c["cross_encoder_score"] = float(raw_score)
            c["cross_encoder_score_normalized"] = float(norm_score)

        # Sort by cross-encoder score
        candidates.sort(key=lambda x: x["weight"], reverse=True)

        return candidates
