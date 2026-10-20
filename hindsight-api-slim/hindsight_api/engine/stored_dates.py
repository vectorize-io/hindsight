"""What counts as a real date in a memory's time fields (#5820).

Year 1 and year 9999 are not dates anyone means: they are how a model or an export writes
"since forever" / "still ongoing". Kept, they do damage — datetime.min/max are stored by
Postgres as ``-infinity``/``infinity`` and read back by asyncpg *without* a timezone while
every finite value comes back aware, so the row breaks any arithmetic or comparison across its
dates (temporal recall, consolidation's date merging). The per-fact ordering offset also pushes a
late-9999 value past datetime.max. Every place a date enters a memory treats them as unknown.
"""

from datetime import MAXYEAR, MINYEAR, UTC, datetime


def is_placeholder_date(value: datetime) -> bool:
    """True for a year-1 / year-9999 value — an open bound, not a date."""
    return value.year in (MINYEAR, MAXYEAR)


def finite_utc(value: datetime | None) -> datetime | None:
    """The value as an aware UTC datetime, or None when unset or a placeholder.

    A naive value is read as UTC, as everywhere else in the repo (the Oracle driver returns
    every timestamp naive).
    """
    if value is None or is_placeholder_date(value):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
