"""Time-window gating for the consolidation background job.

Consolidation is Hindsight's most LLM-heavy background operation. Operators
point it at cheaper off-peak model tiers (batch pricing, spot GPUs) by
restricting when it may run to a configurable daily window — e.g. 01:00–06:00,
or cross-midnight 22:00–06:00. Tasks that land outside the window are *deferred*
(not failed): the worker holds them as ``pending`` with ``next_retry_at`` set to
the next window-open instant, and they drain automatically once the window
opens.

This module only computes the schedule. The defers themselves are produced by
``DeferOperation`` in the engine handler; keeping the arithmetic here makes it a
pure, fully deterministic unit.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone, tzinfo

__all__ = ["next_window_open"]

UTC = timezone.utc


def next_window_open(
    now: datetime,
    start: time,
    end: time,
    tz: tzinfo,
) -> datetime | None:
    """Return the next window-open instant in UTC, or ``None`` if ``now`` is inside the window.

    The window is a repeating daily interval evaluated in ``tz``'s local time:

    - ``start < end`` — normal day window, e.g. 01:00–06:00.
    - ``start > end`` — cross-midnight window, e.g. 22:00–06:00.
    - ``start == end`` — degenerate: treated as always-open (``None``).

    The interval is half-open ``[start, end)``: the instant exactly equal to
    ``end`` is already outside.

    Args:
        now: The current instant (must be timezone-aware).
        start: Window start time-of-day (24h, minute precision).
        end: Window end time-of-day (24h, minute precision).
        tz: Timezone in which the window boundaries are defined.

    Returns:
        UTC datetime of the next window open (minute precision from the
        configured wall-clock times), or ``None`` when now is inside the window.
    """
    if start == end:
        # Degenerate configuration — treat as always-open rather than a
        # zero-length window that immediately re-opens every midnight.
        return None

    local = now.astimezone(tz)
    today_start = local.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
    today_end = local.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)

    if start < end:
        # Normal window on [today_start, today_end).
        if today_start <= local < today_end:
            return None
        if local < today_start:
            next_open = today_start
        else:
            next_open = today_start + timedelta(days=1)
    else:
        # Cross-midnight window: [today_start, tomorrow's today_end).
        # Outside range is [today_end, today_start) — i.e. late morning to
        # early evening — and the next open is today's start.
        if today_end <= local < today_start:
            next_open = today_start
        else:
            return None
    return next_open.astimezone(UTC)
