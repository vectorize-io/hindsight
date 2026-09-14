"""The default temporal anchor is UTC, not the server's local wall clock.

``event_date`` is stored in UTC and ``retrieve_temporal_combined_sql`` stamps a naive
window as UTC, so a window anchored on local midnight names a different day for part of
every day on a server whose ``TZ`` is not UTC.

Two zones, checked together in one test on purpose. Each one alone only disagrees with
UTC for part of the day (``Pacific/Kiritimati`` at UTC+14 from 10:00Z, ``Pacific/Midway``
at UTC-11 until 11:00Z), so the pair is what makes the check hold at any hour.
"""

import time
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer

ZONES = ("Pacific/Kiritimati", "Pacific/Midway")


def _local_utc_offset() -> timedelta:
    return datetime.now().replace(microsecond=0) - datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def test_relative_window_anchors_on_utc_not_server_local(monkeypatch):
    analyzer = DateparserQueryAnalyzer()
    for zone in ZONES:
        monkeypatch.setenv("TZ", zone)
        time.tzset()
        try:
            if _local_utc_offset() == timedelta(0):
                pytest.skip(f"tzdata for {zone} is not installed; TZ fell back to UTC")
            expected = datetime.now(UTC).date() - timedelta(days=1)
            constraint = analyzer.analyze("what happened yesterday").temporal_constraint
            assert constraint is not None
            assert constraint.start_date.date() == expected, f"{zone}: anchored on local time"
            assert constraint.end_date.date() == expected, f"{zone}: anchored on local time"
        finally:
            monkeypatch.delenv("TZ", raising=False)
            time.tzset()


def test_explicit_reference_date_is_untouched(monkeypatch):
    """A caller-supplied anchor still wins, whatever the server's zone is."""
    analyzer = DateparserQueryAnalyzer()
    reference = datetime(2025, 1, 15, 12, 0, 0)
    monkeypatch.setenv("TZ", "Pacific/Kiritimati")
    time.tzset()
    try:
        constraint = analyzer.analyze("what happened yesterday", reference).temporal_constraint
        assert constraint is not None
        assert constraint.start_date.date() == reference.date() - timedelta(days=1)
    finally:
        monkeypatch.delenv("TZ", raising=False)
        time.tzset()
