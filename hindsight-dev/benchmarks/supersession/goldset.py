"""Supersession gold set — structured fixture for the interval-algebra contract.

The supersession module's correctness surface is small (one pure function
``resolve_supersession`` and one validator ``validate_verdict_indices``) but
its branches are subtle: equal-start ties, mentioned_at floor, disjoint
intervals, late-arrival reversal. Unit-testing each branch with hand-rolled
data is fine, but it doesn't scale as the contract grows — and it loses the
property "every case I care about is documented somewhere grep-able".

This file is that documentation. Each ``GoldSupersessionCase`` is a triple of
inputs (new fact, candidate fact) plus the expected verdict shape. The
companion test (``test_supersession_goldset.py``) loads them and asserts the
engine produces the documented outcome. A reviewer can scan this file and
know exactly what the contract is; a regression is a single row's failure
that names the row, not a stack trace.

The cases are also loadable as JSON for downstream tooling (e.g. an e2e
runner that ingests the case and reads ``valid_until`` back from the DB).
The JSON shape is documented on ``gold_case_to_dict`` below; keep them in
sync if you add fields.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from hindsight_api.engine.retain.supersession import FactTimeline

GoldKind = Literal["SUPERSEDE", "COEXIST", "UNDECIDABLE", "RAISE"]


@dataclass(frozen=True)
class GoldOutcome:
    """Expected outcome of one ``resolve_supersession`` invocation.

    Discriminated by ``kind``:

    * ``SUPERSEDE`` — one fact wins, the other is superseded. ``loser_id``,
      ``winner_id`` and ``valid_until`` are the contract's full answer.
    * ``COEXIST`` — disjoint intervals, both facts stay. ``loser_id`` is None.
    * ``UNDECIDABLE`` — the inputs are outside the function's decidable
      range (missing ``occurred_start``, equal-start with tied
      ``mentioned_at``). Function returns None; the field is for the
      "why" so the test log reads cleanly.
    * ``RAISE`` — the validator (``validate_verdict_indices``) is supposed
      to raise on this input. ``reason`` says what kind of input it is.

    Field-level ``reason`` is the comment that survives into the test
    failure message — when a row fails, the test log says "out_of_range_
    negative_index_raises: Pydantic validation rejects -1" rather than
    "test_supersession_goldset failed: assert None == 0".
    """

    kind: GoldKind
    loser_id: str | None = None
    winner_id: str | None = None
    valid_until: datetime | None = None
    reason: str | None = None


@dataclass(frozen=True)
class GoldSupersessionCase:
    """One row of the supersession gold set."""

    name: str
    new: FactTimeline
    candidate: FactTimeline
    expected: GoldOutcome
    # When the case is for ``validate_verdict_indices`` rather than
    # ``resolve_supersession``, the rows below populate and ``new`` /
    # ``candidate`` are ignored. Keeping both shapes in one file makes
    # the gold set a single grep target for the whole module.
    verdict_indices: tuple[int, ...] | None = None
    verdict_candidate_count: int | None = None


# ---------------------------------------------------------------------------
# Helpers for compact row construction
# ---------------------------------------------------------------------------

_DT = datetime
_T0 = datetime(2024, 1, 1, tzinfo=UTC)
_T1 = datetime(2024, 2, 1, tzinfo=UTC)
_T2 = datetime(2024, 3, 1, tzinfo=UTC)
_T3 = datetime(2024, 4, 1, tzinfo=UTC)


def make_fact_timeline(
    id: str,
    *,
    occurred_start: datetime | None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
) -> FactTimeline:
    """Compact ``FactTimeline`` constructor — only the used fields are filled.

    Default ``mentioned_at`` is the ``occurred_start`` + 1h so equal-start
    cases that need a non-None mention have a stable default. Tests that
    care about the mention explicitly pass it.

    Public so the gold set's test module can construct fixtures without
    reaching into a module-private helper.
    """
    return FactTimeline(
        id=id,
        occurred_start=occurred_start,
        occurred_end=occurred_end,
        mentioned_at=mentioned_at
        if mentioned_at is not None
        else (occurred_start + timedelta(hours=1) if occurred_start is not None else None),
    )


# ---------------------------------------------------------------------------
# Cases — one per branch of resolve_supersession / validate_verdict_indices
# ---------------------------------------------------------------------------


CASES: tuple[GoldSupersessionCase, ...] = (
    # --- resolve_supersession: disjoint intervals (both stay) -------------
    GoldSupersessionCase(
        name="disjoint_intervals_coexist",
        new=make_fact_timeline("new", occurred_start=_T1, occurred_end=_T2),
        candidate=make_fact_timeline("cand", occurred_start=_T3, occurred_end=_T3 + timedelta(days=30)),
        expected=GoldOutcome(kind="COEXIST", reason="non-overlapping windows, both stay"),
    ),
    GoldSupersessionCase(
        name="disjoint_point_intervals_supersede",
        # Two point facts at different times. ``_intervals_disjoint``
        # returns False when either side has no ``occurred_end`` — point
        # facts are not closed intervals, so the disjoint guard does not
        # apply. Falls through to the "earlier loses" rule.
        new=make_fact_timeline("new", occurred_start=_T1),
        candidate=make_fact_timeline("cand", occurred_start=_T2),
        expected=GoldOutcome(
            kind="SUPERSEDE",
            loser_id="new",
            winner_id="cand",
            valid_until=_T2,
            reason="point facts have no occurred_end; disjoint guard doesn't fire; earlier loses",
        ),
    ),
    # --- resolve_supersession: new fact is earlier, candidate is later ----
    GoldSupersessionCase(
        name="new_earlier_loses_to_candidate",
        new=make_fact_timeline("new", occurred_start=_T1),
        candidate=make_fact_timeline("cand", occurred_start=_T2),
        expected=GoldOutcome(
            kind="SUPERSEDE",
            loser_id="new",
            winner_id="cand",
            valid_until=_T2,
            reason="earlier-starting fact loses; later-starting candidate wins",
        ),
    ),
    # --- resolve_supersession: candidate is earlier (late-arrival reversal)
    GoldSupersessionCase(
        name="candidate_earlier_loses_to_new",
        # This is the Graphiti late-arrival case: a system learned a fact
        # earlier but only just wrote it down. The candidate is the
        # earlier one, so the candidate loses and the new fact wins.
        new=make_fact_timeline("new", occurred_start=_T2),
        candidate=make_fact_timeline("cand", occurred_start=_T1),
        expected=GoldOutcome(
            kind="SUPERSEDE",
            loser_id="cand",
            winner_id="new",
            valid_until=_T2,
            reason="candidate is earlier so it loses; the newer occurrence wins",
        ),
    ),
    # --- resolve_supersession: equal starts, mentions differ ---------------
    GoldSupersessionCase(
        name="equal_start_newer_mention_wins",
        new=make_fact_timeline("new", occurred_start=_T1, mentioned_at=_T1 + timedelta(hours=2)),
        candidate=make_fact_timeline("cand", occurred_start=_T1, mentioned_at=_T1 + timedelta(hours=1)),
        expected=GoldOutcome(
            kind="SUPERSEDE",
            loser_id="cand",
            winner_id="new",
            # boundary = max(new.mentioned_at, new.occurred_start + 1s) so
            # the CHECK constraint valid_until > occurred_start is satisfied
            # even when mention == start.
            valid_until=_T1 + timedelta(hours=2),
            reason="equal starts → mention wins; newer mention loses 1s floor",
        ),
    ),
    GoldSupersessionCase(
        name="equal_start_candidate_mention_newer",
        new=make_fact_timeline("new", occurred_start=_T1, mentioned_at=_T1 + timedelta(hours=1)),
        candidate=make_fact_timeline("cand", occurred_start=_T1, mentioned_at=_T1 + timedelta(hours=2)),
        expected=GoldOutcome(
            kind="SUPERSEDE",
            loser_id="new",
            winner_id="cand",
            valid_until=_T1 + timedelta(hours=2),
            reason="equal starts → mention wins; older mention is the loser",
        ),
    ),
    # --- resolve_supersession: equal starts, tied mentions = undecidable --
    GoldSupersessionCase(
        name="equal_start_tied_mention_undecidable",
        new=make_fact_timeline("new", occurred_start=_T1, mentioned_at=_T1 + timedelta(hours=1)),
        candidate=make_fact_timeline("cand", occurred_start=_T1, mentioned_at=_T1 + timedelta(hours=1)),
        expected=GoldOutcome(
            kind="UNDECIDABLE",
            reason="tied mentions on equal start → observation-layer reconciles",
        ),
    ),
    # --- resolve_supersession: equal starts, missing mention = undecidable
    GoldSupersessionCase(
        name="equal_start_missing_mention_undecidable",
        new=make_fact_timeline("new", occurred_start=_T1, mentioned_at=None),
        candidate=make_fact_timeline("cand", occurred_start=_T1, mentioned_at=None),
        expected=GoldOutcome(
            kind="UNDECIDABLE",
            reason="any side missing mention → cannot break tie",
        ),
    ),
    # --- resolve_supersession: missing occurred_start = undecidable -------
    GoldSupersessionCase(
        name="missing_new_occurred_start_returns_none",
        new=make_fact_timeline("new", occurred_start=None),
        candidate=make_fact_timeline("cand", occurred_start=_T1),
        expected=GoldOutcome(
            kind="UNDECIDABLE",
            reason="new fact lacks occurred_start — defensive log + return None",
        ),
    ),
    GoldSupersessionCase(
        name="missing_candidate_occurred_start_returns_none",
        new=make_fact_timeline("new", occurred_start=_T1),
        candidate=make_fact_timeline("cand", occurred_start=None),
        expected=GoldOutcome(
            kind="UNDECIDABLE",
            reason="candidate lacks occurred_start — defensive log + return None",
        ),
    ),
    # --- validate_verdict_indices: in-range, out-of-range, negative --------
    GoldSupersessionCase(
        name="verdict_indices_in_range_passes",
        # The interval-algebra inputs are ignored for validator-only cases
        # but kept for symmetry — every case in the file is a row, no
        # half-rows.
        new=make_fact_timeline("new", occurred_start=_T0),
        candidate=make_fact_timeline("cand", occurred_start=_T0),
        expected=GoldOutcome(kind="COEXIST", reason="validator case; algebra is a no-op"),
        verdict_indices=(0, 1, 2),
        verdict_candidate_count=5,
    ),
    GoldSupersessionCase(
        name="verdict_indices_out_of_range_dropped",
        new=make_fact_timeline("new", occurred_start=_T0),
        candidate=make_fact_timeline("cand", occurred_start=_T0),
        expected=GoldOutcome(kind="COEXIST", reason="indices >= count are dropped"),
        verdict_indices=(0, 1, 99),
        verdict_candidate_count=3,
    ),
    GoldSupersessionCase(
        name="verdict_indices_negative_dropped",
        new=make_fact_timeline("new", occurred_start=_T0),
        candidate=make_fact_timeline("cand", occurred_start=_T0),
        expected=GoldOutcome(
            kind="COEXIST",
            reason=(
                "negative index is filtered out by validate_verdict_indices' 0 <= i < count "
                "guard (deep-dive 5 §1.2 defensive parse — mirrors relation_extraction); "
                "no raise, just a quiet drop"
            ),
        ),
        verdict_indices=(-1,),
        verdict_candidate_count=5,
    ),
    GoldSupersessionCase(
        name="verdict_indices_empty_passes",
        new=make_fact_timeline("new", occurred_start=_T0),
        candidate=make_fact_timeline("cand", occurred_start=_T0),
        expected=GoldOutcome(kind="COEXIST", reason="no contradictions cited is valid"),
        verdict_indices=(),
        verdict_candidate_count=3,
    ),
)


def gold_case_to_dict(case: GoldSupersessionCase) -> dict[str, Any]:
    """JSON-friendly projection of one gold row.

    Datetimes round-trip as ISO 8601 with explicit ``+00:00`` so the file
    is portable across Python versions and parsers. None fields are
    preserved (the loader is happy with the field absent or null).
    """

    def _iso(dt: datetime | None) -> str | None:
        return dt.isoformat() if dt is not None else None

    def _ft_dict(ft: FactTimeline) -> dict[str, Any]:
        return {
            "id": ft.id,
            "occurred_start": _iso(ft.occurred_start),
            "occurred_end": _iso(ft.occurred_end),
            "mentioned_at": _iso(ft.mentioned_at),
        }

    out: dict[str, Any] = {
        "name": case.name,
        "new": _ft_dict(case.new),
        "candidate": _ft_dict(case.candidate),
        "expected": {
            "kind": case.expected.kind,
            "loser_id": case.expected.loser_id,
            "winner_id": case.expected.winner_id,
            "valid_until": _iso(case.expected.valid_until),
            "reason": case.expected.reason,
        },
    }
    if case.verdict_indices is not None:
        out["verdict_indices"] = list(case.verdict_indices)
    if case.verdict_candidate_count is not None:
        out["verdict_candidate_count"] = case.verdict_candidate_count
    return out


def gold_case_from_dict(data: dict[str, Any]) -> GoldSupersessionCase:
    """Inverse of :func:`gold_case_to_dict`."""

    def _dt(s: str | None) -> datetime | None:
        return datetime.fromisoformat(s) if s is not None else None

    def make_fact_timeline(data: dict[str, Any]) -> FactTimeline:
        return FactTimeline(
            id=data["id"],
            occurred_start=_dt(data.get("occurred_start")),
            occurred_end=_dt(data.get("occurred_end")),
            mentioned_at=_dt(data.get("mentioned_at")),
        )

    exp = data["expected"]
    return GoldSupersessionCase(
        name=data["name"],
        new=make_fact_timeline(data["new"]),
        candidate=make_fact_timeline(data["candidate"]),
        expected=GoldOutcome(
            kind=exp["kind"],
            loser_id=exp.get("loser_id"),
            winner_id=exp.get("winner_id"),
            valid_until=_dt(exp.get("valid_until")),
            reason=exp.get("reason"),
        ),
        verdict_indices=tuple(data["verdict_indices"]) if "verdict_indices" in data else None,
        verdict_candidate_count=data.get("verdict_candidate_count"),
    )


def load_gold_set(path: str | Path) -> list[GoldSupersessionCase]:
    """Load a gold-set JSON file produced by :func:`dump_gold_set`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [gold_case_from_dict(entry) for entry in raw]


def dump_gold_set(path: str | Path, cases: list[GoldSupersessionCase] | None = None) -> None:
    """Write the gold set to a JSON file (default: :data:`CASES`)."""
    rows = [gold_case_to_dict(c) for c in (cases if cases is not None else CASES)]
    Path(path).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
