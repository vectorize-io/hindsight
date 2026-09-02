"""
Cross-encoder neural reranking for search results.
"""

import calendar
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

# Recency decay: maps a memory's age (days) onto a freshness signal in [0, 1]
# where 0.5 is neutral (no boost). The signal is then folded into the
# multiplicative recency_boost via `1 + recency_alpha * (recency - 0.5)`.
#
#   "linear"      — straight line from 1.0 (today) to a floor of 0.1, reaching
#                   the floor at `linear_window_days`. The historical default.
#   "exponential" — 0.5 ** (days_ago / halflife_days). The half-life is the age
#                   at which the signal is exactly neutral (0.5): younger
#                   memories are boosted, older ones penalised, with a smooth
#                   asymptote toward 0 (no hard cutoff).
#   "none"        — always neutral (0.5), disabling the recency boost entirely.
# The validated set of names lives in config.RECENCY_DECAY_FUNCTIONS.
_RECENCY_DECAY_FUNCTION: str = "linear"
_RECENCY_DECAY_LINEAR_WINDOW_DAYS: float = 365.0
_RECENCY_DECAY_HALFLIFE_DAYS: float = 90.0


def compute_recency_decay(
    days_ago: float,
    function: str = _RECENCY_DECAY_FUNCTION,
    linear_window_days: float = _RECENCY_DECAY_LINEAR_WINDOW_DAYS,
    halflife_days: float = _RECENCY_DECAY_HALFLIFE_DAYS,
) -> float:
    """Map a memory's age in days to a freshness signal in [0, 1] (neutral 0.5).

    Future-dated memories (negative ``days_ago``) clamp to the maximum freshness
    so they are never penalised. See ``RECENCY_DECAY_FUNCTIONS`` for the shapes.
    """
    if function == "none":
        return 0.5
    if function == "exponential":
        if halflife_days <= 0:
            return 0.5
        return min(1.0, 0.5 ** (days_ago / halflife_days))
    # "linear" (default): straight decay to a 0.1 floor over the window.
    window = linear_window_days if linear_window_days > 0 else _RECENCY_DECAY_LINEAR_WINDOW_DAYS
    return max(0.1, min(1.0, 1.0 - (days_ago / window)))


# A memory dated to a *period* rather than an instant carries that period in
# (occurred_start, occurred_end). Extraction emits a full span for coarse dates —
# "in 2015" becomes 2015-01-01 → 2015-12-31, "in March 2026" becomes
# 2026-03-01 → 2026-03-31 — so the granularity is already in the data and does
# not need to be stored separately.
#
# How far SHORT of an exact calendar month/year a span may fall and still be read
# as that period. Extraction spells the last instant of a period three ways —
# Dec 31 23:59:59 (period - 1s), Dec 31 00:00:00 (period - 1 day), or the
# exclusive Jan 1 of the next year (exactly the period) — so accept that window
# rather than matching one spelling. The window is deliberately one-sided: a span
# LONGER than the calendar period is a genuine interval, not a coarse date.
_CALENDAR_PERIOD_TOLERANCE_SECONDS: float = 86400.0


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _spans_calendar_period(start: datetime, end: datetime) -> bool:
    """True when [start, end] covers exactly one calendar month or year.

    This distinguishes a *coarse date* — "sometime in 2015", where the whole
    period is one unit of uncertainty — from a *genuine interval* such as an
    employment or a multi-day conference, which really did span its whole range.
    The two are scored differently: see ``_recency_for_unit``.

    Matched on span length rather than on the boundaries lining up with midnight,
    because ``_add_temporal_offsets`` shifts every fact's timestamps by a
    sub-second amount to keep them ordered. That shift moves both ends equally,
    so it preserves the span exactly while destroying absolute alignment.
    """
    span = (end - start).total_seconds()
    if span <= 0:
        return False
    month_seconds = calendar.monthrange(start.year, start.month)[1] * 86400.0
    year_seconds = (366.0 if calendar.isleap(start.year) else 365.0) * 86400.0
    return any(
        period - _CALENDAR_PERIOD_TOLERANCE_SECONDS <= span <= period for period in (month_seconds, year_seconds)
    )


def _recency_for_unit(
    occurred_start: datetime | None,
    occurred_end: datetime | None,
    mentioned_at: datetime | None,
    now: datetime,
    function: str,
    linear_window_days: float,
    halflife_days: float,
) -> float:
    """Freshness signal in [0, 1] (neutral 0.5) for one memory unit.

    A unit carrying a *coarse* date — one the text stated only to the year or the
    month, which extraction records as that whole calendar period — is scored from
    the END of the period, the latest the event could have happened, instead of
    from ``occurred_start``. Scoring it from the start invents an age the memory
    never had: "the 2026 summit" stored as 2026-01-01 reads as eight months stale
    in August 2026 and loses to any newer but unrelated fact — issue #3893.

    For a coarse date the signal is additionally capped at neutral, because the
    end of the period is a bound and not an observation. Without the cap a period
    still in progress ends in the future, ``compute_recency_decay`` clamps it to
    1.0, and we would swap an invented staleness penalty for an equally invented
    freshness boost. Capping says "we don't know when in this period" while still
    letting genuinely old periods decay — a 2015 date reaches the floor either way.
    """
    if occurred_start is not None and occurred_end is not None:
        start, end = _as_utc(occurred_start), _as_utc(occurred_end)
        if _spans_calendar_period(start, end):
            recency = compute_recency_decay(
                (now - end).total_seconds() / 86400,
                function,
                linear_window_days,
                halflife_days,
            )
            return min(0.5, recency)

    # Not a coarse date: score the unit's effective instant, unchanged. Note this
    # deliberately leaves a genuine interval (a multi-day trip, an employment)
    # aged from occurred_start as it always has been — arguably it should age from
    # its end too, but that is a separate ranking change and not what #3893 reports.
    effective = occurred_start or mentioned_at or occurred_end
    if effective is None:
        return 0.5
    return compute_recency_decay(
        (now - _as_utc(effective)).total_seconds() / 86400,
        function,
        linear_window_days,
        halflife_days,
    )


def apply_combined_scoring(
    scored_results: list[ScoredResult],
    now: datetime,
    recency_alpha: float = _RECENCY_ALPHA,
    temporal_alpha: float = _TEMPORAL_ALPHA,
    proof_count_alpha: float = _PROOF_COUNT_ALPHA,
    is_passthrough_reranker: bool = False,
    recency_decay_function: str = _RECENCY_DECAY_FUNCTION,
    recency_decay_linear_window_days: float = _RECENCY_DECAY_LINEAR_WINDOW_DAYS,
    recency_decay_halflife_days: float = _RECENCY_DECAY_HALFLIFE_DAYS,
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
        recency_decay_function: Age→freshness curve — "linear" (default),
            "exponential", or "none". See compute_recency_decay.
        recency_decay_linear_window_days: Days over which the linear curve
            decays to its floor (default 365).
        recency_decay_halflife_days: For the exponential curve, the age at which
            the recency signal is neutral (0.5) (default 90).
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
        # Recency: configurable decay (linear default; see compute_recency_decay)
        # → [0.0, 1.0]; neutral 0.5 if no date. Period-dated units are scored from
        # the end of their period; everything else falls back to the effective
        # instant (occurred_start, then mentioned_at, then occurred_end — the same
        # COALESCE order as retrieval._coalesce_date), so a memory carrying only a
        # mentioned_at / occurred_end still gets correct recency ordering instead
        # of a flat neutral 0.5. See _recency_for_unit.
        sr.recency = _recency_for_unit(
            sr.retrieval.occurred_start,
            sr.retrieval.occurred_end,
            sr.retrieval.mentioned_at,
            now,
            recency_decay_function,
            recency_decay_linear_window_days,
            recency_decay_halflife_days,
        )

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
        # Surface the proof signal so the trace can show the proof_count_boost
        # factor (otherwise the reranked breakdown can't reconcile CE × boosts).
        sr.proof_norm = proof_norm

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

        from hindsight_api.config import ENV_MODEL_INIT_TIMEOUT, get_config

        cross_encoder = self.cross_encoder
        # For in-process models, run in thread pool to avoid blocking event loop.
        # getattr: tests inject duck-typed cross encoders that don't subclass
        # CrossEncoderModel and so don't carry the property.
        if getattr(cross_encoder, "blocking_init", False):
            loop = asyncio.get_event_loop()
            init = loop.run_in_executor(None, lambda: asyncio.run(cross_encoder.initialize()))
        else:
            init = cross_encoder.initialize()

        # Cap lazy init with the same wall-clock timeout used at startup so a
        # hung model download surfaces as a clear error on the request that
        # triggered it, rather than hanging the caller forever.
        init_timeout = get_config().model_init_timeout
        try:
            await asyncio.wait_for(init, timeout=init_timeout)
        except TimeoutError as e:
            raise RuntimeError(
                f"Cross-encoder initialization did not complete within {init_timeout:g}s. "
                f"The reranker model is likely blocked loading — e.g. an offline model "
                f"download. Increase {ENV_MODEL_INIT_TIMEOUT} if the first-time download "
                f"legitimately needs more time."
            ) from e
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

        # Normalize scores to [0, 1] range.
        # External API rerankers (Cohere, Jina, llama.cpp/Qwen, etc.) return
        # calibrated relevance_score already in [0, 1]. These are used as-is
        # so that absolute confidence is preserved — a top candidate scoring
        # 0.007 stays low rather than being inflated to 1.0 by rank normalization.
        # Local models return logits (any real number) — sigmoid is appropriate.
        import numpy as np

        def _sigmoid(x: float) -> float:
            return 1 / (1 + np.exp(-x))

        if scores and min(scores) >= 0.0 and max(scores) <= 1.0:
            # Scores already in [0, 1] — pass through to preserve absolute
            # confidence signal from calibrated rerankers.
            normalized_scores = list(scores)
        else:
            # Scores are logits (e.g. local sentence-transformers models).
            # Sigmoid maps (-inf, +inf) to (0, 1).
            normalized_scores = [_sigmoid(score) for score in scores]

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
