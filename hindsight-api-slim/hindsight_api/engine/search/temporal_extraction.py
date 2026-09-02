"""
Temporal extraction for time-aware search queries.

Handles natural language temporal expressions using transformer-based query analysis.
"""

import asyncio
import atexit
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer, QueryAnalyzer

logger = logging.getLogger(__name__)

# Temporal extraction is pure CPU, and recall calls it from an async request
# path. Running it inline blocks the event loop for the whole duration, which
# stalls every other in-flight request in the process — not just the recall
# doing the work. Measured with 16 concurrent extractions over document-sized
# text: the loop got a single scheduler tick in 1.3 seconds.
#
# It is offloaded to a thread instead. The pool is deliberately **one worker**:
# the work is pure-Python and holds the GIL, so widening it does not add
# parallelism, it just makes threads fight over the GIL. Measured, same 16
# extractions:
#
#     inline              total= 1318ms   loop stall max=1318ms
#     max_workers=1       total= 1438ms   loop stall max=   2.8ms
#     max_workers=2       total= 2091ms   loop stall max=   4.5ms
#     max_workers=4       total= 4751ms   loop stall max=   6.3ms
#     unbounded           total=16688ms   loop stall max=  33.8ms
#
# One worker keeps throughput (+9%) while the loop stays responsive (470x), and
# preserves the serialisation the inline version already had. Anything wider
# trades throughput away for nothing.
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="temporal-extract")
                atexit.register(_shutdown_executor)
    return _executor


def _shutdown_executor() -> None:
    global _executor
    executor, _executor = _executor, None
    if executor is not None:
        executor.shutdown(wait=False)


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


async def extract_temporal_constraint_async(
    query: str,
    reference_date: datetime | None = None,
    analyzer: QueryAnalyzer | None = None,
) -> tuple[datetime, datetime] | None:
    """Async form of :func:`extract_temporal_constraint`, off the event loop.

    Same result as the sync function — this only changes *where* the CPU work
    runs. Use this from request paths; the sync form remains for callers that
    are not already async.

    Safe to run off-thread as of the detector rewrite: the analyzer owns its
    ``_ExactLanguageSearch`` rather than sharing dateparser's module-level
    singleton (which caches state on itself per call), and the character-table
    cache is lock-guarded.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _get_executor(),
        lambda: extract_temporal_constraint(query, reference_date=reference_date, analyzer=analyzer),
    )
