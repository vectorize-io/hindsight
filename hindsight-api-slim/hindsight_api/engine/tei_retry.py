"""Shared retry timing for Text Embeddings Inference HTTP clients."""

import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx


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
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def tei_retry_delay(
    response: httpx.Response,
    fallback_delay: float,
    *,
    max_delay: float,
) -> float:
    """Honor Retry-After when present and de-synchronize fallback retries."""
    retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
    limit = max_delay if math.isfinite(max_delay) and max_delay >= 0 else 60.0
    fallback = fallback_delay if math.isfinite(fallback_delay) else 0.0
    requested_delay = max(fallback, retry_after or 0.0, 0.0)
    delay = min(requested_delay, limit)
    if delay and requested_delay >= limit:
        downward_jitter = min(delay * 0.1, 1.0)
        return delay - random.uniform(0.0, downward_jitter)
    jitter_limit = min(delay * 0.1, 1.0, limit - delay)
    return delay + random.uniform(0.0, jitter_limit) if jitter_limit else delay
