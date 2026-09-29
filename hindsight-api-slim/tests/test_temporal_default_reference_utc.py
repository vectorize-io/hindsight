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


@pytest.fixture
def set_tz(monkeypatch):
    """Switch the process zone; restore the runner's own ``TZ`` (not just unset it) afterwards."""

    def _set(zone: str) -> None:
        monkeypatch.setenv("TZ", zone)
        time.tzset()

    yield _set
    # monkeypatch restores the env var only after this teardown, so undo it here first:
    # tzset() must see the original TZ, or the rest of the worker keeps the test's zone.
    monkeypatch.undo()
    time.tzset()


def test_relative_window_anchors_on_utc_not_server_local(set_tz):
    analyzer = DateparserQueryAnalyzer()
    for zone in ZONES:
        set_tz(zone)
        if _local_utc_offset() == timedelta(0):
            pytest.skip(f"tzdata for {zone} is not installed; TZ fell back to UTC")
        expected = datetime.now(UTC).date() - timedelta(days=1)
        constraint = analyzer.analyze("what happened yesterday").temporal_constraint
        assert constraint is not None
        assert constraint.start_date.date() == expected, f"{zone}: anchored on local time"
        assert constraint.end_date.date() == expected, f"{zone}: anchored on local time"


def test_explicit_reference_date_is_untouched(set_tz):
    """A caller-supplied anchor still wins, whatever the server's zone is."""
    analyzer = DateparserQueryAnalyzer()
    reference = datetime(2025, 1, 15, 12, 0, 0)
    set_tz("Pacific/Kiritimati")
    constraint = analyzer.analyze("what happened yesterday", reference).temporal_constraint
    assert constraint is not None
    assert constraint.start_date.date() == reference.date() - timedelta(days=1)
