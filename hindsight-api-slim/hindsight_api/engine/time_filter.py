"""Server-side time windows for the two list surfaces (memories, documents).

One helper rather than two copies because ``time_field`` is INTERPOLATED into
SQL: it must be whitelisted, and a whitelist that exists twice is a whitelist
that drifts. See #4349.

The rule the whole feature rests on, stated once here:

* ``time_field`` names a single time axis, and that axis drives BOTH the
  ``start_date``/``end_date`` filter and the ordering. There is no way to filter
  on one column and sort by another — that combination has no meaning anyone has
  asked for, and it doubles the surface.
* Rows with no value on that axis are EXCLUDED, not sorted to the end. "Order by
  when it was mentioned" cannot answer for a memory that was never dated, so it
  says nothing rather than guessing ``created_at`` (which is what the
  ``memories-timeseries`` bucketing does, deliberately, for a different question:
  a chart wants every row somewhere, a filtered list wants only the rows that
  genuinely fall in the window).
* Consequences worth knowing: the ``IS NOT NULL`` makes the partial indexes from
  ``b3c4d5e6f7g8`` usable, and it removes any need for ``NULLS LAST`` — which the
  Oracle SQL rewriter does not translate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, get_args

# The axes each list surface can be filtered and ordered by. `event_date` is
# deliberately absent from the memory set: it is kept only for backward
# compatibility, and exposing it would invite callers to filter on the wrong one.
#
# Declared as Literals so the HTTP layer can annotate the query parameter with
# them — FastAPI then rejects an unknown value with a 422 that NAMES the valid
# ones, and the generated clients get a real enum. The tuples below are derived
# from the Literals rather than written twice, because the runtime whitelist and
# the advertised enum disagreeing is the failure mode worth designing out.
MemoryTimeField = Literal["created_at", "updated_at", "mentioned_at", "occurred_start", "occurred_end"]
DocumentTimeField = Literal["created_at", "updated_at"]

MEMORY_TIME_FIELDS: tuple[str, ...] = get_args(MemoryTimeField)
DOCUMENT_TIME_FIELDS: tuple[str, ...] = get_args(DocumentTimeField)


@dataclass(frozen=True)
class TimeClause:
    """The SQL a time window contributes, in the same shape as ``TagClause``.

    ``conditions`` are ANDed into the caller's WHERE list; ``order_by`` replaces
    the caller's default ordering, or is ``None`` when no time parameter was
    given and today's ordering must stand.
    """

    conditions: list[str]
    params: list[datetime]
    next_param_offset: int
    order_by: str | None


def validate_time_window(
    *,
    time_field: str | None,
    start_date: datetime | None,
    end_date: datetime | None,
    allowed: tuple[str, ...],
) -> None:
    """Reject a window the store must not be asked to answer. Raises ``ValueError``.

    Separate from :func:`build_time_clause` because a store that owns its own
    documents never reaches the SQL builder — ``list_documents`` returns to it
    before any clause is built — and an inverted window has to be an error there
    too, not a silently empty page on one backend and a 400 on the other.
    """
    if time_field is not None and time_field not in allowed:
        raise ValueError(f"Invalid time_field '{time_field}': expected one of {', '.join(allowed)}.")
    if start_date is not None and end_date is not None and end_date <= start_date:
        raise ValueError("Invalid time window: end_date must be after start_date.")


def build_time_clause(
    *,
    time_field: str | None,
    start_date: datetime | None,
    end_date: datetime | None,
    allowed: tuple[str, ...],
    default_field: str,
    param_offset: int = 1,
) -> TimeClause:
    """Build the WHERE conditions and ORDER BY for one time window.

    Args:
        time_field: The axis to filter and order by. ``None`` falls back to
            ``default_field``. A value outside ``allowed`` raises — unlike
            ``memories-timeseries``, which silently falls back, because a list
            that quietly answers about a different column than the one asked for
            is a wrong answer, not a lenient one.
        start_date: Inclusive lower bound, or ``None``.
        end_date: Exclusive upper bound, or ``None`` — half-open ``[start, end)``,
            matching ``audit-logs`` so adjacent windows neither overlap nor gap.
        allowed: The whitelist for this surface (``MEMORY_TIME_FIELDS`` /
            ``DOCUMENT_TIME_FIELDS``).
        default_field: The axis used when only ``start_date``/``end_date`` are given.
        param_offset: First free ``$N`` placeholder.

    Returns:
        A ``TimeClause``. When no time parameter was given at all, it is empty
        and ``order_by`` is ``None``, so the caller's existing query is unchanged.
    """
    validate_time_window(time_field=time_field, start_date=start_date, end_date=end_date, allowed=allowed)

    if time_field is None and start_date is None and end_date is None:
        return TimeClause(conditions=[], params=[], next_param_offset=param_offset, order_by=None)

    # `field` is one of `allowed` here, never caller text, so interpolating it is safe.
    column = time_field or default_field

    # Excluding NULLs is what makes the ordering total, so it belongs to the
    # window itself, not to the bounds — a bare `time_field` with no dates still
    # has to drop the rows it cannot place.
    conditions = [f"{column} IS NOT NULL"]
    params: list[datetime] = []
    offset = param_offset
    if start_date is not None:
        conditions.append(f"{column} >= ${offset}")
        params.append(start_date)
        offset += 1
    if end_date is not None:
        conditions.append(f"{column} < ${offset}")
        params.append(end_date)
        offset += 1

    # `id` breaks ties so paging with OFFSET cannot repeat or skip a row when many
    # share a timestamp — the bulk-import case, where every row has the same one.
    return TimeClause(
        conditions=conditions,
        params=params,
        next_param_offset=offset,
        order_by=f"{column} DESC, id",
    )
