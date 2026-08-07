"""Unit tests for the consolidation-window env parsing in config.py.

Covers ``_parse_consolidation_window`` / ``_parse_consolidation_window_boundary``:
HH:MM validation, the START/END-must-be-paired rule, the START==END degenerate
case, and the timezone fallback. Pure env parsing — no DB required.
"""

from datetime import time

import pytest

from hindsight_api.config import (
    ENV_CONSOLIDATION_WINDOW_END,
    ENV_CONSOLIDATION_WINDOW_START,
    ENV_CONSOLIDATION_WINDOW_TZ,
    ConsolidationWindowSpec,
    _parse_consolidation_window,
    _parse_consolidation_window_boundary,
)

START = ENV_CONSOLIDATION_WINDOW_START
END = ENV_CONSOLIDATION_WINDOW_END
TZ = ENV_CONSOLIDATION_WINDOW_TZ


# ── _parse_consolidation_window_boundary ────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("01:00", time(1, 0)),
        ("23:59", time(23, 59)),
        ("00:00", time(0, 0)),
    ],
)
def test_boundary_parses_valid_hhmm(raw: str, expected: time) -> None:
    assert _parse_consolidation_window_boundary(raw, "START") == expected


@pytest.mark.parametrize("raw", ["", None])
def test_boundary_missing_returns_none(raw: str | None) -> None:
    assert _parse_consolidation_window_boundary(raw, "START") is None


@pytest.mark.parametrize("raw", ["25:00", "01:60", "0100", "abc", "1;00", " 01:00"])
def test_boundary_invalid_returns_none(raw: str) -> None:
    assert _parse_consolidation_window_boundary(raw, "START") is None


# ── _parse_consolidation_window ─────────────────────────────────────────────


def test_unset_window_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(START, raising=False)
    monkeypatch.delenv(END, raising=False)
    monkeypatch.delenv(TZ, raising=False)
    assert _parse_consolidation_window() == ConsolidationWindowSpec(None, None, "UTC")


def test_full_window_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(START, "01:00")
    monkeypatch.setenv(END, "06:00")
    monkeypatch.setenv(TZ, "Asia/Shanghai")
    assert _parse_consolidation_window() == ConsolidationWindowSpec(time(1, 0), time(6, 0), "Asia/Shanghai")


def test_window_defaults_to_utc_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(START, "01:00")
    monkeypatch.setenv(END, "06:00")
    monkeypatch.delenv(TZ, raising=False)
    assert _parse_consolidation_window() == ConsolidationWindowSpec(time(1, 0), time(6, 0), "UTC")


def test_cross_midnight_window_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(START, "22:00")
    monkeypatch.setenv(END, "06:00")
    monkeypatch.delenv(TZ, raising=False)
    assert _parse_consolidation_window() == ConsolidationWindowSpec(time(22, 0), time(6, 0), "UTC")


@pytest.mark.parametrize("start,end", [("01:00", None), (None, "06:00")])
def test_partial_window_disabled_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, start: str | None, end: str | None
) -> None:
    """START without END (or vice versa) keeps the window disabled and warns."""
    if start is None:
        monkeypatch.delenv(START, raising=False)
    else:
        monkeypatch.setenv(START, start)
    if end is None:
        monkeypatch.delenv(END, raising=False)
    else:
        monkeypatch.setenv(END, end)
    monkeypatch.delenv(TZ, raising=False)
    with caplog.at_level("WARNING", logger="hindsight_api.config"):
        spec = _parse_consolidation_window()
    assert spec == ConsolidationWindowSpec(None, None, "UTC")
    assert "must be set together" in caplog.text


def test_equal_boundaries_disabled_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(START, "01:00")
    monkeypatch.setenv(END, "01:00")
    monkeypatch.delenv(TZ, raising=False)
    with caplog.at_level("WARNING", logger="hindsight_api.config"):
        spec = _parse_consolidation_window()
    assert spec == ConsolidationWindowSpec(None, None, "UTC")
    assert "zero-length" in caplog.text


def test_invalid_boundary_disables_window(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """An invalid boundary with a valid partner still disables the window."""
    monkeypatch.setenv(START, "25:00")
    monkeypatch.setenv(END, "06:00")
    monkeypatch.delenv(TZ, raising=False)
    with caplog.at_level("WARNING", logger="hindsight_api.config"):
        spec = _parse_consolidation_window()
    assert spec == ConsolidationWindowSpec(None, None, "UTC")
    assert "must be set together" in caplog.text


def test_invalid_tz_falls_back_to_utc(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv(START, "01:00")
    monkeypatch.setenv(END, "06:00")
    monkeypatch.setenv(TZ, "Not/AZone")
    with caplog.at_level("WARNING", logger="hindsight_api.config"):
        spec = _parse_consolidation_window()
    assert spec == ConsolidationWindowSpec(time(1, 0), time(6, 0), "UTC")
    assert "falling back to UTC" in caplog.text


def test_empty_tz_falls_back_to_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly empty TZ env var behaves like an invalid one (UTC)."""
    monkeypatch.setenv(START, "01:00")
    monkeypatch.setenv(END, "06:00")
    monkeypatch.setenv(TZ, "")
    assert _parse_consolidation_window() == ConsolidationWindowSpec(time(1, 0), time(6, 0), "UTC")
