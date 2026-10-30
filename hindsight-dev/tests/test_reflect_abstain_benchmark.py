"""The reflect abstain benchmark's dataset and scoring, without a decision model.

A dataset row the runner cannot load, or a routing that drifts from reflect's own cut-offs,
would quietly score something other than what ships.
"""

from __future__ import annotations

from benchmarks.reflect_abstain.reflect_abstain_benchmark import (
    Outcome,
    load,
    route_pick,
    route_verdict,
    separation,
)


def test_dataset_loads_with_both_splits_and_both_labels() -> None:
    moments = load("all")
    assert len(moments) == 118
    assert {m.split for m in moments} == {"tuned", "heldout"}
    for m in moments:
        assert m.label in ("speak", "silent")
        assert m.memories and m.query and m.context
        assert m.state().startswith(m.context) and "\n\nMEMORY:\n[1] " in m.state()
    assert len({m.id for m in moments}) == len(moments)


def test_pick_routing_matches_reflects_cut_offs() -> None:
    assert route_pick(0.95) == "silent"
    assert route_pick(0.9) == "silent"
    assert route_pick(0.5) == "check"
    assert route_pick(0.05) == "writer"


def test_verdict_routing_bands() -> None:
    assert route_verdict(0.7, write_at=0.66, check_at=0.5) == "writer"
    assert route_verdict(0.6, write_at=0.66, check_at=0.5) == "check"
    assert route_verdict(0.2, write_at=0.66, check_at=0.5) == "silent"


def test_only_a_silent_speak_or_a_written_silent_is_wrong() -> None:
    def o(label: str, route: str) -> Outcome:
        return Outcome("x", "tuned", label, route, 0.5, 0.1)  # type: ignore[arg-type]

    assert o("speak", "silent").wrong == "missed note"
    assert o("silent", "writer").wrong == "false note"
    assert o("speak", "check").wrong is None and o("silent", "check").wrong is None


def test_separation() -> None:
    perfect = [Outcome("a", "t", "speak", "writer", 0.9, 0), Outcome("b", "t", "silent", "silent", 0.1, 0)]
    assert separation(perfect) == 1.0
    assert separation(perfect[:1]) is None
