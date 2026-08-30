"""Occurrence-date precision helpers shared by retain and recall.

The timestamp columns describe *when an event occurred*.  A value such as
``2026`` still needs a concrete timestamp for storage, but that timestamp must
not make recall believe January 1 was supplied by the user.  This module keeps
that uncertainty separate from ingestion and mention timestamps.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, cast

OccurrencePrecision = Literal["instant", "day", "month", "year", "range", "unknown"]
CoarseOccurrencePrecision = Literal["month", "year"]

OCCURRENCE_PRECISION_METADATA_KEY = "hindsight:occurred_precision"

_VALID_OCCURRENCE_PRECISIONS = frozenset({"instant", "day", "month", "year", "range", "unknown"})

_MONTH_NAMES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_NAME_PATTERN = "|".join(sorted(_MONTH_NAMES, key=lambda month_name: len(month_name), reverse=True))

_NUMERIC_YEAR_RE = re.compile(r"^(?:(?:in|during|the\s+year)\s+)?(?P<year>\d{4})$", re.IGNORECASE)
_NUMERIC_MONTH_RE = re.compile(
    r"^(?:(?:in|during)\s+)?(?P<year>\d{4})[-/](?P<month>0?[1-9]|1[0-2])$",
    re.IGNORECASE,
)
_CHINESE_YEAR_RE = re.compile(r"^(?:在)?\s*(?P<year>\d{4})\s*年$")
_CHINESE_MONTH_RE = re.compile(r"^(?:在)?\s*(?P<year>\d{4})\s*年\s*(?P<month>0?[1-9]|1[0-2])\s*月$")
_ENGLISH_MONTH_RE = re.compile(
    rf"^(?:(?:in|during)\s+)?(?P<month_name>{_MONTH_NAME_PATTERN})\.?\s+(?P<year>\d{{4}})$",
    re.IGNORECASE,
)
_ISO_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_INSTANT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[Tt ]")
_CANONICAL_WHEN_RE = re.compile(
    r"\s+\|\s+When:\s*(?P<value>[^|]+?)(?=\s+\|\s+|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CoarseOccurrence:
    """A calendar period whose exact day is unknown."""

    precision: CoarseOccurrencePrecision
    year: int
    month: int | None = None


@dataclass(frozen=True)
class CalendarBounds:
    """Earliest and latest possible instants for a coarse occurrence."""

    earliest: datetime
    latest: datetime


def coerce_occurrence_precision(value: object) -> OccurrencePrecision | None:
    """Return a normalized precision value, or ``None`` for invalid input."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized not in _VALID_OCCURRENCE_PRECISIONS:
        return None
    return cast(OccurrencePrecision, normalized)


def parse_coarse_occurrence(value: object) -> CoarseOccurrence | None:
    """Parse an entire year/month expression without matching arbitrary prose."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().rstrip(".。")
    if not normalized:
        return None

    match = _NUMERIC_MONTH_RE.fullmatch(normalized) or _CHINESE_MONTH_RE.fullmatch(normalized)
    if match:
        year = int(match.group("year"))
        if year == 0:
            return None
        return CoarseOccurrence(
            precision="month",
            year=year,
            month=int(match.group("month")),
        )

    match = _ENGLISH_MONTH_RE.fullmatch(normalized)
    if match:
        year = int(match.group("year"))
        if year == 0:
            return None
        return CoarseOccurrence(
            precision="month",
            year=year,
            month=_MONTH_NAMES[match.group("month_name").lower()],
        )

    match = _NUMERIC_YEAR_RE.fullmatch(normalized) or _CHINESE_YEAR_RE.fullmatch(normalized)
    if match:
        year = int(match.group("year"))
        return CoarseOccurrence(precision="year", year=year) if year != 0 else None

    return None


def coarse_occurrence_start(value: object) -> datetime | None:
    """Materialize the storage timestamp used for a recognized coarse value."""
    coarse = parse_coarse_occurrence(value)
    if coarse is None:
        return None
    return datetime(coarse.year, coarse.month or 1, 1)


def _normalized_datetime(value: object) -> datetime | None:
    """Best-effort normalization used only to distinguish points from ranges."""
    coarse_start = coarse_occurrence_start(value)
    if coarse_start is not None:
        return coarse_start
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _is_distinct_range(occurred_start: object, occurred_end: object) -> bool:
    if not isinstance(occurred_start, str) or not occurred_start.strip():
        return False
    if not isinstance(occurred_end, str) or not occurred_end.strip():
        return False
    start = _normalized_datetime(occurred_start)
    end = _normalized_datetime(occurred_end)
    if start is not None and end is not None:
        return start != end
    return occurred_start.strip() != occurred_end.strip()


def infer_occurrence_precision(
    *,
    explicit: object = None,
    occurred_start: object = None,
    occurred_end: object = None,
    when: object = None,
) -> OccurrencePrecision:
    """Resolve occurrence precision from a structured extraction response.

    Exact-looking timestamps may have been imputed by an older model, so a
    coarse structured ``when`` value is considered before classifying such a
    timestamp as an exact day.  A genuinely distinct start/end range remains a
    range unless the model supplied an explicit valid precision.
    """
    explicit_precision = coerce_occurrence_precision(explicit)
    if explicit_precision in ("year", "month", "range"):
        return explicit_precision

    if _is_distinct_range(occurred_start, occurred_end):
        return "range"

    coarse_start = parse_coarse_occurrence(occurred_start)
    if coarse_start is not None:
        return coarse_start.precision

    coarse_when = parse_coarse_occurrence(when)
    if coarse_when is not None:
        return coarse_when.precision

    if explicit_precision in ("day", "instant"):
        return explicit_precision

    if isinstance(occurred_start, str):
        normalized_start = occurred_start.strip()
        if _ISO_DAY_RE.fullmatch(normalized_start):
            return "day"
        if _ISO_INSTANT_RE.match(normalized_start):
            return "instant"

    return "unknown"


def with_occurrence_precision(
    metadata: Mapping[str, Any] | None,
    precision: object,
) -> dict[str, Any]:
    """Copy metadata and install a concrete engine-owned precision value.

    ``unknown`` is represented by absence so undated/non-event memories do not
    gain an internal key, while any caller-supplied reserved value is removed.
    """
    result = dict(metadata or {})
    result.pop(OCCURRENCE_PRECISION_METADATA_KEY, None)
    normalized_precision = coerce_occurrence_precision(precision)
    if normalized_precision not in (None, "unknown"):
        result[OCCURRENCE_PRECISION_METADATA_KEY] = normalized_precision
    return result


def calendar_bounds(reference: datetime, precision: object) -> CalendarBounds | None:
    """Return the earliest/latest possible instants for a coarse occurrence."""
    normalized_precision = coerce_occurrence_precision(precision)
    if normalized_precision == "year":
        start = reference.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = reference.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
        return CalendarBounds(earliest=start, latest=end)
    if normalized_precision == "month":
        last_day = calendar.monthrange(reference.year, reference.month)[1]
        start = reference.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = reference.replace(day=last_day, hour=23, minute=59, second=59, microsecond=999999)
        return CalendarBounds(earliest=start, latest=end)
    return None


def _same_point(start: datetime, end: datetime) -> bool:
    start_normalized = start
    end_normalized = end
    if start_normalized.tzinfo is not None:
        start_normalized = start_normalized.astimezone(timezone.utc).replace(tzinfo=None)
    if end_normalized.tzinfo is not None:
        end_normalized = end_normalized.astimezone(timezone.utc).replace(tzinfo=None)
    return start_normalized == end_normalized


def resolve_edited_occurrence_precision(
    *,
    stored_precision: object,
    occurred_start_supplied: bool,
    occurred_start_value: object,
    occurred_end_supplied: bool,
    final_start: datetime | None,
    final_end: datetime | None,
) -> OccurrencePrecision:
    """Resolve the precision of a curation edit from its final window.

    Curation receives lexical evidence only for fields supplied by the caller.
    An untouched point therefore preserves its stored precision, while a newly
    supplied full date/datetime can safely become ``day``/``instant``.  Clearing
    the occurrence never falls back to mention time, and collapsing an old range
    without new start evidence becomes ``unknown`` rather than inventing a
    granularity for the retained timestamp.
    """
    current = coerce_occurrence_precision(stored_precision) or "unknown"
    if not occurred_start_supplied and not occurred_end_supplied:
        return current
    if final_start is None:
        return "unknown"
    if final_end is not None and not _same_point(final_start, final_end):
        return "range"

    if occurred_start_supplied:
        coarse = parse_coarse_occurrence(occurred_start_value)
        if coarse is not None:
            return coarse.precision
        if isinstance(occurred_start_value, str):
            normalized = occurred_start_value.strip()
            if _ISO_DAY_RE.fullmatch(normalized):
                return "day"
            if _ISO_INSTANT_RE.match(normalized):
                return "instant"
        return "unknown"

    return current if current != "range" else "unknown"


def recover_legacy_occurrence_precision(
    fact_text: str,
    occurred_start: datetime | None,
    occurred_end: datetime | None = None,
) -> CoarseOccurrencePrecision | None:
    """Recover only the point-shaped coarse ``When:`` form emitted by retain.

    The text must contain a canonical pipe-delimited ``When:`` segment, the
    segment must consist entirely of a recognized year/month value, and the
    stored occurrence must still sit on that period's imputed first calendar
    day.  This deliberately avoids treating arbitrary prose or genuine January
    1 events as coarse.
    """
    if not fact_text or occurred_start is None or occurred_end is None:
        return None
    if not _same_point(occurred_start, occurred_end):
        return None

    match = _CANONICAL_WHEN_RE.search(fact_text)
    if match is None:
        return None
    coarse = parse_coarse_occurrence(match.group("value"))
    if coarse is None or occurred_start.year != coarse.year or occurred_start.day != 1:
        return None
    if coarse.precision == "year":
        return "year" if occurred_start.month == 1 else None
    return "month" if occurred_start.month == coarse.month else None


def resolve_stored_occurrence_precision(
    *,
    metadata: Mapping[str, object] | None,
    fact_text: str,
    occurred_start: datetime | None,
    occurred_end: datetime | None = None,
    allow_legacy_recovery: bool = True,
) -> OccurrencePrecision:
    """Resolve precision for recall, with a narrow pre-metadata fallback."""
    if isinstance(metadata, Mapping) and OCCURRENCE_PRECISION_METADATA_KEY in metadata:
        stored = coerce_occurrence_precision(metadata.get(OCCURRENCE_PRECISION_METADATA_KEY))
        if stored is not None:
            return stored

    if not allow_legacy_recovery:
        return "unknown"
    legacy = recover_legacy_occurrence_precision(fact_text, occurred_start, occurred_end)
    return legacy or "unknown"


def format_occurrence_date(reference: datetime, precision: object) -> str:
    """Format occurrence context without inventing a missing month or day."""
    normalized_precision = coerce_occurrence_precision(precision)
    if normalized_precision == "year":
        return f"{reference.year:04d}"
    if normalized_precision == "month":
        return f"{reference.strftime('%B %Y')} ({reference.strftime('%Y-%m')})"
    return f"{reference.strftime('%B %d, %Y')} ({reference.strftime('%Y-%m-%d')})"


def format_embedding_occurrence_date(
    reference: datetime,
    precision: object,
    format_exact_date: Callable[[datetime], str],
) -> str:
    """Format retain embedding context without fabricating a calendar part.

    Exact dates retain the historical caller-provided representation so this
    changes only coarse occurrences.  The embedding path intentionally differs
    from :func:`format_occurrence_date`, whose richer ISO suffix is for reranker
    context rather than stored-vector compatibility.
    """
    normalized_precision = coerce_occurrence_precision(precision)
    if normalized_precision == "year":
        return f"{reference.year:04d}"
    if normalized_precision == "month":
        return reference.strftime("%B %Y")
    return format_exact_date(reference)


def format_text_signal_occurrence_date(reference: datetime, precision: object) -> str:
    """Format one BM25 date token while preserving occurrence granularity."""
    normalized_precision = coerce_occurrence_precision(precision)
    if normalized_precision == "year":
        return f"{reference.year:04d}"
    if normalized_precision == "month":
        return reference.strftime("%B %Y")
    return reference.strftime("%B %d %Y").replace(" 0", " ")
