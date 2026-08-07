"""Unit tests for the consolidation window gate (pure function)."""

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from hindsight_api.engine.consolidation_window import next_window_open

SHANGHAI = ZoneInfo("Asia/Shanghai")  # UTC+8, no DST
NEW_YORK = ZoneInfo("America/New_York")  # DST-aware


def _utc(y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


# ── normal window (start < end) ──────────────────────────────────────────────


def test_inside_window_returns_none() -> None:
    # 02:00 UTC falls inside 01:00–06:00 UTC.
    assert next_window_open(_utc(2026, 8, 3, 2, 0), time(1, 0), time(6, 0), UTC) is None


def test_before_window_returns_today_start() -> None:
    nxt = next_window_open(_utc(2026, 8, 3, 0, 30), time(1, 0), time(6, 0), UTC)
    assert nxt == _utc(2026, 8, 3, 1, 0)


def test_after_window_returns_tomorrow_start() -> None:
    nxt = next_window_open(_utc(2026, 8, 3, 7, 0), time(1, 0), time(6, 0), UTC)
    assert nxt == _utc(2026, 8, 4, 1, 0)


def test_start_boundary_is_inside() -> None:
    assert next_window_open(_utc(2026, 8, 3, 1, 0), time(1, 0), time(6, 0), UTC) is None


def test_end_boundary_is_outside() -> None:
    # Half-open [start, end): the instant exactly at end is outside.
    nxt = next_window_open(_utc(2026, 8, 3, 6, 0), time(1, 0), time(6, 0), UTC)
    assert nxt == _utc(2026, 8, 4, 1, 0)


def test_one_minute_before_end_is_inside() -> None:
    assert next_window_open(_utc(2026, 8, 3, 5, 59), time(1, 0), time(6, 0), UTC) is None


# ── cross-midnight window (start > end) ──────────────────────────────────────


def test_cross_midnight_evening_inside() -> None:
    assert next_window_open(_utc(2026, 8, 3, 23, 0), time(22, 0), time(6, 0), UTC) is None


def test_cross_midnight_early_morning_inside() -> None:
    assert next_window_open(_utc(2026, 8, 4, 5, 0), time(22, 0), time(6, 0), UTC) is None


def test_cross_midnight_off_peak_returns_today_start() -> None:
    nxt = next_window_open(_utc(2026, 8, 3, 12, 0), time(22, 0), time(6, 0), UTC)
    assert nxt == _utc(2026, 8, 3, 22, 0)


def test_cross_midnight_end_boundary_is_outside() -> None:
    nxt = next_window_open(_utc(2026, 8, 3, 6, 0), time(22, 0), time(6, 0), UTC)
    assert nxt == _utc(2026, 8, 3, 22, 0)


def test_cross_midnight_exactly_at_start_is_inside() -> None:
    assert next_window_open(_utc(2026, 8, 3, 22, 0), time(22, 0), time(6, 0), UTC) is None


# ── degenerate / invalid ─────────────────────────────────────────────────────


def test_start_equals_end_is_always_open() -> None:
    assert next_window_open(_utc(2026, 8, 3, 12, 0), time(3, 0), time(3, 0), UTC) is None
    assert next_window_open(_utc(2026, 8, 3, 3, 0), time(3, 0), time(3, 0), UTC) is None


# ── timezone conversion ──────────────────────────────────────────────────────


def test_window_evaluated_in_configured_tz() -> None:
    # 20:00–22:00 Shanghai == 12:00–14:00 UTC. 13:00 UTC is inside.
    assert next_window_open(_utc(2026, 8, 3, 13, 0), time(20, 0), time(22, 0), SHANGHAI) is None


def test_tz_conversion_before_window() -> None:
    # 11:00 UTC == 19:00 Shanghai — before the 20:00–22:00 window.
    nxt = next_window_open(_utc(2026, 8, 3, 11, 0), time(20, 0), time(22, 0), SHANGHAI)
    assert nxt == _utc(2026, 8, 3, 12, 0)


def test_dst_tz_window_alignment() -> None:
    # 10:00–11:00 EDT (UTC-4) on a summer date == 14:00–15:00 UTC.
    assert next_window_open(_utc(2026, 7, 1, 14, 30), time(10, 0), time(11, 0), NEW_YORK) is None
    nxt = next_window_open(_utc(2026, 7, 1, 16, 0), time(10, 0), time(11, 0), NEW_YORK)
    assert nxt == _utc(2026, 7, 2, 14, 0)


def test_return_is_utc() -> None:
    nxt = next_window_open(_utc(2026, 8, 3, 11, 0), time(20, 0), time(22, 0), SHANGHAI)
    assert nxt is not None
    assert nxt.utcoffset() == timedelta(0)
