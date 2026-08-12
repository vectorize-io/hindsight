"""
Temporal extraction for time-aware search queries.

Handles natural language temporal expressions using transformer-based query analysis.
"""

import logging
from datetime import datetime

from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer, QueryAnalyzer

logger = logging.getLogger(__name__)

# Global default analyzer instance
# Can be overridden by passing a custom analyzer to extract_temporal_constraint
_default_analyzer: QueryAnalyzer | None = None


def get_default_analyzer() -> QueryAnalyzer:
    """
    Get or create the default query analyzer.

    Uses lazy initialization to avoid loading at import time.

    Returns:
        Default DateparserQueryAnalyzer instance
    """
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = DateparserQueryAnalyzer()
    return _default_analyzer


def extract_temporal_constraint(
    query: str,
    reference_date: datetime | None = None,
    analyzer: QueryAnalyzer | None = None,
) -> tuple[datetime, datetime] | None:
    """
    Extract temporal constraint from query.

    Returns (start_date, end_date) tuple if temporal constraint found, else None.

    Args:
        query: Search query
        reference_date: Reference date for relative terms (defaults to now)
        analyzer: Custom query analyzer (defaults to DateparserQueryAnalyzer)

    Returns:
        (start_date, end_date) tuple or None
    """
    if analyzer is None:
        analyzer = get_default_analyzer()

    # Recall must never fail because temporal analysis choked on the query text.
    # Consolidation recalls with stored fact text as the query, so a single
    # pathological phrase (e.g. "十万年前" → year -97974) would otherwise fail
    # every recall touching that bank, deterministically (issue #3217). Degrade
    # to "no temporal signal" here — the one entry point the recall path uses —
    # while analyze() itself stays strict so parser bugs still surface in tests
    # and to direct callers.
    try:
        analysis = analyzer.analyze(query, reference_date)
    except Exception as e:
        logger.warning(
            "Temporal query analysis raised %s (treating as no temporal constraint): %s",
            type(e).__name__,
            e,
        )
        return None

    if analysis.temporal_constraint:
        result = (analysis.temporal_constraint.start_date, analysis.temporal_constraint.end_date)
        return result

    return None
