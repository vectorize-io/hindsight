"""Shared retry timing for Text Embeddings Inference HTTP clients."""

import errno
import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

# Ceiling for a single backoff sleep, independent of the client's request
# timeout. The reranker holds its concurrency semaphore across the sleep, so a
# large server-supplied Retry-After would stall every queued rerank behind it;
# capping well below the request timeout keeps a slow proxy from converting a
# transient overload into a minutes-long recall stall.
MAX_RETRY_DELAY_SECONDS = 5.0
TEI_KEEPALIVE_EXPIRY_SECONDS = 15.0

# Fraction of the delay used as the jitter window, matching the "equal jitter"
# policy of ``db_utils._backoff_delay``. TEI overload is self-synchronising -- a
# burst of concurrent requests exhausts the permit pool at the same instant and
# gets 429ed together -- so the window has to be wide enough to actually break
# the burst apart. A token few percent would leave the callers retrying in
# lockstep and re-colliding on the same exhausted pool.
JITTER_RATIO = 0.5

# TEI requests are usually against a pooled HTTP client. If the underlying
# socket dies while idle, the first failure often shows up as a bare OSError
# instead of a structured httpx timeout/connect error.
RETRYABLE_OS_ERRNOS = {
    errno.EBADF,
    errno.ECONNABORTED,
    errno.ECONNREFUSED,
    errno.ECONNRESET,
    errno.EHOSTDOWN,
    errno.EHOSTUNREACH,
    errno.ENETDOWN,
    errno.ENETRESET,
    errno.ENETUNREACH,
    errno.EPIPE,
    errno.ETIMEDOUT,
}


def _retry_after_seconds(value: str | None) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (IndexError, TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _delay_limit(request_timeout: float) -> float:
    """Longest sleep we are willing to take between attempts."""
    if not math.isfinite(request_timeout) or request_timeout < 0:
        return MAX_RETRY_DELAY_SECONDS
    return min(request_timeout, MAX_RETRY_DELAY_SECONDS)


def tei_retry_delay(
    response: httpx.Response,
    fallback_delay: float,
    *,
    request_timeout: float,
) -> float:
    """Seconds to sleep before retrying a transient TEI response.

    A server-supplied ``Retry-After`` wins over the caller's exponential
    backoff, and both are bounded by :func:`_delay_limit`. The result is spread
    over a jitter window so concurrent callers do not resume in lockstep.
    """
    retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
    limit = _delay_limit(request_timeout)
    fallback = fallback_delay if math.isfinite(fallback_delay) else 0.0
    requested_delay = max(fallback, retry_after or 0.0, 0.0)
    delay = min(requested_delay, limit)
    if delay <= 0:
        return 0.0
    spread = delay * JITTER_RATIO
    if requested_delay >= limit:
        # Already at the ceiling, so the only room to de-synchronise is downward.
        return delay - random.uniform(0.0, spread)
    # Retry-After is a minimum, so spread upward -- but never past the ceiling.
    return delay + random.uniform(0.0, min(spread, limit - delay))


def is_retryable_tei_transport_error(exc: BaseException) -> bool:
    """Return True for transport failures that should be retried once.

    This keeps stale pooled sockets from bubbling up as hard failures when the
    underlying descriptor died while idle.
    """

    if isinstance(exc, OSError):
        return exc.errno in RETRYABLE_OS_ERRNOS or exc.errno is None

    for attr in ("__cause__", "__context__"):
        nested = getattr(exc, attr, None)
        if nested is not None and nested is not exc and is_retryable_tei_transport_error(nested):
            return True

    return False
