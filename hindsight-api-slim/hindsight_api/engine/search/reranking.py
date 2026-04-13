"""
Cross-encoder neural reranking for search results.
"""

import math
from datetime import datetime, timezone

from .types import MergedCandidate, ScoredResult

UTC = timezone.utc

# Multiplicative boost alphas for recency and temporal proximity.
# Each signal contributes at most ±(alpha/2) relative adjustment to the base CE score,
# so the max combined boost is (1 + alpha/2)^2 ≈ +21% and min is (1 - alpha/2)^2 ≈ -19%.
_RECENCY_ALPHA: float = 0.2
_TEMPORAL_ALPHA: float = 0.2
_PROOF_COUNT_ALPHA: float = 0.1  # Conservative: max ±5% for evidence strength


def apply_combined_scoring(
    scored_results: list[ScoredResult],
    now: datetime,
    recency_alpha: float = _RECENCY_ALPHA,
    temporal_alpha: float = _TEMPORAL_ALPHA,
    proof_count_alpha: float = _PROOF_COUNT_ALPHA,
    is_passthrough_reranker: bool = False,
) -> None:
    """Apply combined scoring to a list of ScoredResults in-place.

    Uses the cross-encoder score as the primary relevance signal, with recency,
    temporal proximity, and proof count applied as multiplicative boosts. This
    ensures the influence of these secondary signals is always proportional to
    the base relevance score, regardless of the cross-encoder model's score
    calibration.

    Formula::

        recency_boost     = 1 + recency_alpha     * (recency     - 0.5)   # in [1-α/2, 1+α/2]
        temporal_boost    = 1 + temporal_alpha    * (temporal    - 0.5)   # in [1-α/2, 1+α/2]
        proof_count_boost = 1 + proof_count_alpha * (proof_norm  - 0.5)   # in [1-α/2, 1+α/2]
        combined_score    = CE_normalized * recency_boost * temporal_boost * proof_count_boost

    proof_norm maps proof_count using a smooth logarithmic curve centered at 0.5,
    clamped to [0, 1]:
      proof_count=1 → 0.5 + 0 = 0.5 (neutral multiplier)
      proof_count=150 → clamped to 1.0 (max +5% boost)

    Temporal proximity is treated as neutral (0.5) when not set by temporal retrieval,
    so temporal_boost collapses to 1.0 for non-temporal queries.

    Proof count is treated as neutral (0.5) when not available (non-observation facts),
    so proof_count_boost collapses to 1.0 for world/experience/opinion facts.

    Args:
        scored_results: Results from the cross-encoder reranker. Mutated in place.
        now: Current UTC datetime for recency calculation.
        recency_alpha: Max relative recency adjustment (default 0.2 → ±10%).
        temporal_alpha: Max relative temporal adjustment (default 0.2 → ±10%).
        proof_count_alpha: Max relative proof count adjustment (default 0.1 → ±5%).
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    # When the configured cross-encoder is a passthrough (e.g.
    # RRFPassthroughCrossEncoder used by slim deployments), every
    # cross_encoder_score_normalized is identical and provides no relevance
    # signal. In that case the multiplicative recency / temporal / proof_count
    # boosts below become the *only* ranking signal — making the final order a
    # pure recency sort regardless of how relevant a candidate actually is.
    #
    # Detect that case and seed cross_encoder_score_normalized from the RRF
    # rank instead, so the boosts modulate a meaningful base score rather than
    # replacing it. This is a no-op for real cross-encoders, which produce
    # diverse scores.
    # When the reranker is a passthrough (e.g. RRFPassthroughCrossEncoder used
    # by slim deployments), every cross_encoder_score_normalized is identical
    # and provides no relevance signal. The multiplicative recency / temporal /
    # proof_count boosts below would then become the *only* ranking signal,
    # making the final order a pure recency sort regardless of how relevant a
    # candidate actually is.
    #
    # Seed cross_encoder_score_normalized from the RRF rank instead, so the
    # boosts modulate a meaningful base score. Caller passes is_passthrough
    # explicitly because "all scores identical" is too fragile a heuristic —
    # a real reranker can also tie scores (especially in tests with synthetic
    # data) and we'd corrupt legitimate single-result reranks.
    if is_passthrough_reranker and scored_results:
        n = len(scored_results)
        sorted_by_rrf = sorted(
            scored_results,
            key=lambda s: getattr(getattr(s, "candidate", None), "rrf_score", 0.0),
            reverse=True,
        )
        denom = max(1, n - 1)
        for new_rank, sr in enumerate(sorted_by_rrf):
            # Map rank → [0.1, 1.0] so the recency boost can still nudge
            # ordering between adjacent candidates without overpowering RRF.
            sr.cross_encoder_score_normalized = 1.0 - (0.9 * new_rank / denom)

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

        # Proof count: log-normalized evidence strength; neutral for non-observations.
        proof_count = sr.retrieval.proof_count
        if proof_count is not None and proof_count >= 1:
            # Clamp to [0, 1] so extreme counts stay within documented ±5% range
            proof_norm = min(1.0, max(0.0, 0.5 + (math.log(proof_count) / 10.0)))
        else:
            # Neutral baseline is precisely 0.5, ensuring neutral multiplier (1.0)
            proof_norm = 0.5

        # RRF: kept at 0.0 for trace continuity but excluded from scoring.
        # RRF is batch-relative (min-max normalised) and redundant after reranking.
        sr.rrf_normalized = 0.0

        recency_boost = 1.0 + recency_alpha * (sr.recency - 0.5)
        temporal_boost = 1.0 + temporal_alpha * (sr.temporal - 0.5)
        proof_count_boost = 1.0 + proof_count_alpha * (proof_norm - 0.5)
        sr.combined_score = sr.cross_encoder_score_normalized * recency_boost * temporal_boost * proof_count_boost
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
        import math

        import numpy as np

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        normalized_scores = [sigmoid(score) for score in scores]

        # Create ScoredResult objects with cross-encoder scores
        scored_results = []
        for candidate, raw_score, norm_score in zip(candidates, scores, normalized_scores):
            # Sanitize NaN scores (cross-encoder can return NaN for certain inputs).
            # NaN propagates through all downstream scoring and Pydantic serializes
            # NaN as JSON null, which breaks clients expecting numeric values.
            raw = float(raw_score)
            norm = float(norm_score)
            if math.isnan(raw):
                raw = 0.0
            if math.isnan(norm):
                norm = 0.0
            scored_result = ScoredResult(
                candidate=candidate,
                cross_encoder_score=raw,
                cross_encoder_score_normalized=norm,
                weight=norm,  # Initial weight is just cross-encoder score
            )
            scored_results.append(scored_result)

        # Sort by cross-encoder score
        scored_results.sort(key=lambda x: x.weight, reverse=True)

        return scored_results
