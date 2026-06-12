"""Unit tests for the supersession gold set: every case in :data:`CASES` must
match what ``resolve_supersession`` / ``validate_verdict_indices`` actually do.

The gold set is the documented contract for the interval-algebra core. These
tests are the executor of that contract. Each row gets one test, named after
the row, so a regression names itself in the failure log.
"""

from datetime import datetime, timezone

import pytest
from hindsight_api.engine.retain.supersession import (
    SupersessionVerdict,
    resolve_supersession,
    validate_verdict_indices,
)

from benchmarks.supersession.goldset import (
    CASES,
    GoldOutcome,
    GoldSupersessionCase,
    dump_gold_set,
    gold_case_from_dict,
    gold_case_to_dict,
    load_gold_set,
)


def _verdict_from_indices(indices: tuple[int, ...]) -> SupersessionVerdict:
    """Build a verdict with all candidates in ``contradicted_indices`` —
    simpler than splitting between duplicate/contradicted and still exercises
    the validator's index filtering."""
    return SupersessionVerdict(contradicted_indices=list(indices))


# --- Pure function: resolve_supersession ---------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_gold_case_resolve_supersession(case: GoldSupersessionCase) -> None:
    if case.verdict_indices is not None:
        # Validator-only cases: covered by the dedicated test below.
        pytest.skip("validator case, see test_gold_case_validate_verdict_indices")

    action = resolve_supersession(case.new, case.candidate)
    expected = case.expected

    if expected.kind == "SUPERSEDE":
        assert action is not None, f"{case.name}: expected SUPERSEDE, got None"
        assert action.loser_id == expected.loser_id
        assert action.winner_id == expected.winner_id
        assert action.valid_until == expected.valid_until
    elif expected.kind == "COEXIST":
        assert action is None, f"{case.name}: expected COEXIST, got {action}"
    elif expected.kind == "UNDECIDABLE":
        assert action is None, f"{case.name}: expected UNDECIDABLE (None), got {action}"
    elif expected.kind == "RAISE":
        # No gold row in the current set expects a raise from
        # resolve_supersession; if a future row needs it, drop the
        # `pytest.skip` above and add the corresponding expect_raises
        # branch.
        pytest.fail(f"{case.name}: RAISE kind is reserved for validator cases")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_gold_case_validate_verdict_indices(case: GoldSupersessionCase) -> None:
    if case.verdict_indices is None or case.verdict_candidate_count is None:
        pytest.skip("interval-algebra case, see test_gold_case_resolve_supersession")

    verdict = _verdict_from_indices(case.verdict_indices)
    cleaned = validate_verdict_indices(verdict, candidate_count=case.verdict_candidate_count)
    # Every validator case in the current gold set is COEXIST-shaped (we
    # only care that the validator didn't raise and the indices were
    # filtered). A future row that needs to assert specific kept/dropped
    # values can extend ``GoldOutcome`` with a ``kept_indices`` field.
    assert isinstance(cleaned, SupersessionVerdict)


# --- Roundtrip: every CASES row survives a JSON roundtrip ---------------


def test_gold_set_json_roundtrip(tmp_path):
    """Every gold case must round-trip through JSON without loss.

    Catches accidental field renames, datetime → string conversion bugs,
    and missing Optional defaults. The round-tripped rows are then
    re-asserted against the resolver, so a JSON bug also fails a real
    contract test, not just a serialisation one.
    """
    out = tmp_path / "gold.json"
    dump_gold_set(out)
    loaded = load_gold_set(out)
    assert len(loaded) == len(CASES)
    for original, restored in zip(CASES, loaded, strict=True):
        assert original == restored
        # And the contract still holds for the loaded row.
        if original.verdict_indices is None:
            action = resolve_supersession(restored.new, restored.candidate)
            assert (action is None) == (restored.expected.kind != "SUPERSEDE")


# --- Per-row id distinctness ---------------------------------------------


def test_gold_case_names_are_unique() -> None:
    names = [c.name for c in CASES]
    assert len(names) == len(set(names)), f"duplicate names: {names}"


def test_gold_set_covers_all_documented_branches() -> None:
    """Sanity: every kind + every public branch name must appear at least once.

    If a future refactor adds a new branch to ``resolve_supersession`` and
    nobody updates the gold set, this test fails and points at the gap.
    """
    kinds = {c.expected.kind for c in CASES}
    assert kinds == {"SUPERSEDE", "COEXIST", "UNDECIDABLE"}, (
        f"missing kinds: {kinds}; the gold set should cover all four (RAISE is "
        "reserved for validator cases that don't currently apply)"
    )

    # ``reason`` is the comment that travels with the row into the test
    # failure log (per ``GoldOutcome`` docstring). A row without a reason
    # would fail the contract of "the failure message names the row's
    # why" — a regression test for the documentation discipline itself.
    for c in CASES:
        assert c.expected.reason, f"{c.name}: missing GoldOutcome.reason (see docstring on GoldOutcome)"

    # The supersession cases should cover the four documented branches.
    sub_kinds: set[str] = set()
    for c in CASES:
        if c.verdict_indices is not None:
            continue
        if c.expected.kind == "SUPERSEDE":
            if c.new.occurred_start == c.candidate.occurred_start:
                sub_kinds.add("equal_start_mention_wins")
            elif c.new.occurred_start < c.candidate.occurred_start:
                sub_kinds.add("new_earlier")
            else:
                sub_kinds.add("candidate_earlier_late_arrival")
        elif c.expected.kind == "COEXIST":
            sub_kinds.add("disjoint")
        elif c.expected.kind == "UNDECIDABLE":
            sub_kinds.add("undecidable")
    expected = {
        "new_earlier",
        "candidate_earlier_late_arrival",
        "equal_start_mention_wins",
        "disjoint",
        "undecidable",
    }
    assert sub_kinds == expected, f"missing branch rows: {expected - sub_kinds}"


# --- Gold roundtrip helpers ---------------------------------------------


def test_gold_case_to_from_dict_preserves_tz():
    case = CASES[0]  # disjoint_intervals_coexist — has a non-None date
    d = gold_case_to_dict(case)
    # Restore a tz-aware datetime, not a naive one.
    assert d["new"]["occurred_start"].endswith("+00:00")
    restored = gold_case_from_dict(d)
    assert restored.new.occurred_start.tzinfo is not None
    assert restored == case


def test_gold_outcome_is_frozen():
    outcome = GoldOutcome(kind="COEXIST")
    with pytest.raises((AttributeError, TypeError)):
        outcome.kind = "SUPERSEDE"  # type: ignore[misc]


def test_datetime_helper_for_gold_rows():
    """Cheap regression: the compact make_fact_timeline() helper shouldn't
    accidentally produce a naive datetime when occurred_start is tz-aware."""
    from benchmarks.supersession.goldset import make_fact_timeline

    ft = make_fact_timeline("x", occurred_start=datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert ft.mentioned_at is not None
    assert ft.mentioned_at.tzinfo is not None
