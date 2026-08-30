"""Deterministic tests for occurrence-date precision preservation."""

from datetime import datetime, timezone

import pytest

from hindsight_api.engine.temporal_precision import (
    OCCURRENCE_PRECISION_METADATA_KEY,
    calendar_bounds,
    coarse_occurrence_start,
    infer_occurrence_precision,
    parse_coarse_occurrence,
    recover_legacy_occurrence_precision,
    resolve_stored_occurrence_precision,
    with_occurrence_precision,
)

UTC = timezone.utc


@pytest.mark.parametrize(
    ("value", "expected_precision", "expected_year", "expected_month"),
    [
        ("2026", "year", 2026, None),
        ("in 2026", "year", 2026, None),
        ("2026年", "year", 2026, None),
        ("2026-08", "month", 2026, 8),
        ("2026/8", "month", 2026, 8),
        ("August 2026", "month", 2026, 8),
        ("2026年8月", "month", 2026, 8),
    ],
)
def test_parse_coarse_occurrence(value, expected_precision, expected_year, expected_month):
    parsed = parse_coarse_occurrence(value)

    assert parsed is not None
    assert parsed.precision == expected_precision
    assert parsed.year == expected_year
    assert parsed.month == expected_month


@pytest.mark.parametrize(
    "value",
    [
        "January 1, 2026",
        "2026-01-01",
        "The summit happened in 2026",
        "2026年8月30日",
        "N/A",
        "0000",
        None,
    ],
)
def test_parse_coarse_occurrence_rejects_exact_dates_and_arbitrary_prose(value):
    assert parse_coarse_occurrence(value) is None


def test_coarse_occurrence_start_supports_chinese_values():
    assert coarse_occurrence_start("2026年8月") == datetime(2026, 8, 1)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"explicit": "year", "occurred_start": "2026-01-01"}, "year"),
        ({"occurred_start": "2026"}, "year"),
        ({"occurred_start": "2026-08"}, "month"),
        ({"occurred_start": "2026-01-01", "when": "2026"}, "year"),
        ({"occurred_start": "2026-08-01", "when": "August 2026"}, "month"),
        ({"occurred_start": "2026-08-30"}, "day"),
        ({"occurred_start": "2026-08-30T12:15:00Z"}, "instant"),
        ({"occurred_start": "2026-08-01", "occurred_end": "2026-08-03"}, "range"),
        ({"explicit": "invalid", "occurred_start": "2026"}, "year"),
    ],
)
def test_infer_occurrence_precision(kwargs, expected):
    assert infer_occurrence_precision(**kwargs) == expected


@pytest.mark.parametrize("explicit", ["unknown", "day", "instant"])
def test_deterministic_coarse_evidence_overrides_inconsistent_model_precision(explicit):
    assert infer_occurrence_precision(explicit=explicit, occurred_start="2026-01-01", when="2026") == "year"


def test_calendar_bounds_preserve_timezone_and_cover_full_period():
    reference = datetime(2024, 2, 1, 0, 0, 0, 50_000, tzinfo=UTC)

    month_bounds = calendar_bounds(reference, "month")
    year_bounds = calendar_bounds(reference, "year")

    assert month_bounds is not None
    assert month_bounds.earliest == datetime(2024, 2, 1, tzinfo=UTC)
    assert month_bounds.latest == datetime(2024, 2, 29, 23, 59, 59, 999999, tzinfo=UTC)
    assert year_bounds is not None
    assert year_bounds.earliest == datetime(2024, 1, 1, tzinfo=UTC)
    assert year_bounds.latest == datetime(2024, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)


@pytest.mark.parametrize(
    ("reference", "expected_last_day"),
    [
        (datetime(2023, 2, 15), 28),
        (datetime(2024, 2, 15), 29),
        (datetime(2026, 12, 31, tzinfo=UTC), 31),
        (datetime(2027, 1, 1, tzinfo=UTC), 31),
    ],
)
def test_month_bounds_cover_non_leap_leap_and_year_boundaries(reference, expected_last_day):
    bounds = calendar_bounds(reference, "month")

    assert bounds is not None
    assert bounds.earliest == reference.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    assert bounds.latest == reference.replace(
        day=expected_last_day,
        hour=23,
        minute=59,
        second=59,
        microsecond=999999,
    )


def test_metadata_is_copied_reserved_value_wins_and_unknown_is_omitted():
    caller_metadata = {"source": "test", OCCURRENCE_PRECISION_METADATA_KEY: "day"}

    year_metadata = with_occurrence_precision(caller_metadata, "year")
    unknown_metadata = with_occurrence_precision(caller_metadata, "unknown")

    assert caller_metadata == {"source": "test", OCCURRENCE_PRECISION_METADATA_KEY: "day"}
    assert year_metadata == {"source": "test", OCCURRENCE_PRECISION_METADATA_KEY: "year"}
    assert unknown_metadata == {"source": "test"}


@pytest.mark.parametrize(
    ("text", "occurred_start", "expected"),
    [
        ("Summit talk | When: 2026 | Involving: user", datetime(2026, 1, 1, 0, 0, 0, 20_000), "year"),
        ("Trip | When: August 2026", datetime(2026, 8, 1), "month"),
        ("Trip | When: 2026-08", datetime(2026, 8, 1), "month"),
        ("旅行 | When: 2026年8月 | Involving: 用户", datetime(2026, 8, 1), "month"),
        ("峰会分享 | When: 2026年 | Involving: 用户", datetime(2026, 1, 1), "year"),
        ("Summit talk | When: January 1, 2026", datetime(2026, 1, 1), None),
        ("Trip | When: 2026-08-01", datetime(2026, 8, 1), None),
        ("The summit happened in 2026", datetime(2026, 1, 1), None),
        ("Metadata | NotWhen: 2026", datetime(2026, 1, 1), None),
        ("When: 2026", datetime(2026, 1, 1), None),
        ("Summit talk | When: 2026", datetime(2026, 1, 2), None),
        ("Trip | When: August 2026", datetime(2026, 9, 1), None),
    ],
)
def test_legacy_precision_recovery_is_narrow(text, occurred_start, expected):
    assert recover_legacy_occurrence_precision(text, occurred_start, occurred_start) == expected


def test_legacy_precision_accepts_equivalent_timezone_points_only():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    equivalent_end = datetime.fromisoformat("2026-01-01T08:00:00+08:00")

    assert recover_legacy_occurrence_precision("Summit | When: 2026", start, equivalent_end) == "year"


def test_legacy_precision_does_not_reclassify_a_real_range():
    assert (
        recover_legacy_occurrence_precision(
            "A project ran | When: 2026",
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 12, 31, tzinfo=UTC),
        )
        is None
    )


def test_legacy_precision_requires_a_point_shaped_stored_occurrence():
    assert (
        recover_legacy_occurrence_precision(
            "Summit talk | When: 2026",
            datetime(2026, 1, 1, tzinfo=UTC),
            None,
        )
        is None
    )


def test_stored_metadata_prevents_genuine_january_first_from_legacy_reclassification():
    precision = resolve_stored_occurrence_precision(
        metadata={OCCURRENCE_PRECISION_METADATA_KEY: "day"},
        fact_text="New year event | When: 2026",
        occurred_start=datetime(2026, 1, 1, tzinfo=UTC),
        occurred_end=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert precision == "day"


def test_invalid_stored_metadata_is_safe_and_allows_legacy_recovery():
    precision = resolve_stored_occurrence_precision(
        metadata={OCCURRENCE_PRECISION_METADATA_KEY: "not-valid"},
        fact_text="Summit talk | When: 2026 | Involving: user",
        occurred_start=datetime(2026, 1, 1, tzinfo=UTC),
        occurred_end=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert precision == "year"


def test_non_mapping_stored_metadata_is_safe_and_allows_legacy_recovery():
    precision = resolve_stored_occurrence_precision(
        metadata="not-a-metadata-map",  # type: ignore[arg-type]
        fact_text="Summit talk | When: 2026 | Involving: user",
        occurred_start=datetime(2026, 1, 1, tzinfo=UTC),
        occurred_end=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert precision == "year"


def test_legacy_precision_recovery_can_be_disabled_for_derived_observations():
    precision = resolve_stored_occurrence_precision(
        metadata=None,
        fact_text="Observation | When: 2026",
        occurred_start=datetime(2026, 1, 1, tzinfo=UTC),
        occurred_end=datetime(2026, 1, 1, tzinfo=UTC),
        allow_legacy_recovery=False,
    )

    assert precision == "unknown"
