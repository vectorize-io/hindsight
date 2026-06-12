"""Unit tests for the as_of synthesizer.

The synthesizer is a pure function over the LoCoMo conversation dict; these
tests pin down its contract: which pairs get produced, in what order, and
what their scenarios are. The downstream e2e harness is the thing that
actually scores a recall call against the synthesized pairs; here we just
verify the synthesizer doesn't drop or duplicate rows.
"""

import pytest

from benchmarks.supersession.as_of_synthesizer import (
    AsOfPair,
    AsOfScenario,
    _evidence_sessions,
    _session_dates,
    build_as_of_pairs,
    cap_per_scenario,
)


def _item(*, sessions: list[tuple[int, str, list[str] | None]]) -> dict:
    """Build a minimal LoCoMo item.

    ``sessions`` is a list of (n, date_str, dia_ids). dia_ids == None
    means the session is malformed (its content is not a list) and should
    produce a None event_date.
    """
    conv: dict = {"speaker_a": "A", "speaker_b": "B"}
    for n, date_str, dia_ids in sessions:
        if dia_ids is None:
            conv[f"session_{n}"] = "not-a-list"
        else:
            conv[f"session_{n}"] = [{"speaker": "A", "dia_id": d, "text": "hi"} for d in dia_ids]
        conv[f"session_{n}_date_time"] = date_str
    return {"sample_id": "conv-X", "conversation": conv}


def _qa(question: str, answer: str, evidence: list[str], category: int = 1) -> dict:
    return {"question": question, "answer": answer, "evidence": evidence, "category": category}


# --- Public surface ------------------------------------------------------


def test_synthesizer_produces_before_after_for_single_session_evidence():
    item = _item(
        sessions=[
            (1, "1:00 pm on 8 May, 2023", ["D1:1"]),
            (2, "1:00 pm on 9 May, 2023", ["D2:1"]),
        ],
    )
    item["qa"] = [_qa("q1", "a1", ["D1:1"])]
    pairs = build_as_of_pairs(item)
    scenarios = [p.scenario for p in pairs]
    assert scenarios == ["before", "after"]
    assert pairs[0].as_of.isoformat() == "2023-05-08T12:59:59+00:00"
    assert pairs[1].as_of.isoformat() == "2023-05-08T13:00:00+00:00"


def test_synthesizer_emits_spanning_for_multi_session_evidence():
    item = _item(
        sessions=[
            (1, "1:00 pm on 8 May, 2023", ["D1:1"]),
            (2, "1:00 pm on 9 May, 2023", ["D2:1"]),
            (3, "1:00 pm on 10 May, 2023", ["D3:1"]),
        ],
    )
    item["qa"] = [_qa("q1", "a1", ["D1:1", "D2:1", "D3:1"])]
    pairs = build_as_of_pairs(item)
    scenarios = [p.scenario for p in pairs]
    # 1 before + 1 after + 1 spanning (intermediate session 2)
    assert scenarios == ["before", "after", "spanning"]
    spanning = [p for p in pairs if p.scenario == "spanning"][0]
    assert spanning.as_of.isoformat() == "2023-05-09T13:00:00+00:00"


def test_synthesizer_no_spanning_for_two_session_evidence():
    """Two-session evidence has no intermediate session in the evidence
    set, so no spanning pair is generated. The as_of at the non-evidence
    session date would be arbitrary and meaningless."""
    item = _item(
        sessions=[
            (1, "1:00 pm on 8 May, 2023", ["D1:1"]),
            (2, "1:00 pm on 9 May, 2023", ["D2:1"]),
            (3, "1:00 pm on 10 May, 2023", ["D3:1"]),
        ],
    )
    item["qa"] = [_qa("q1", "a1", ["D1:1", "D3:1"])]
    pairs = build_as_of_pairs(item)
    scenarios = [p.scenario for p in pairs]
    assert scenarios == ["before", "after"]


def test_synthesizer_skips_adversarial_category_5():
    """LoCoMo category 5 is adversarial/unanswerable — same skip rule as
    the dual-memory taskset. The synthesizer must not produce pairs that
    would force the recall to return 'I don't know' on a category-5
    question (which is the correct answer and not a system failure)."""
    item = _item(sessions=[(1, "1:00 pm on 8 May, 2023", ["D1:1"])])
    item["qa"] = [_qa("q1", "a1", ["D1:1"], category=5)]
    assert build_as_of_pairs(item) == []


def test_synthesizer_skips_qa_without_evidence_or_answer():
    item = _item(sessions=[(1, "1:00 pm on 8 May, 2023", ["D1:1"])])
    item["qa"] = [
        _qa("q-no-evidence", "a1", []),
        _qa("q-no-answer", None, ["D1:1"]),  # type: ignore[arg-type]
    ]
    assert build_as_of_pairs(item) == []


def test_synthesizer_skips_malformed_session_dates():
    """If a session's date is unparseable the QA pair that depends on it is
    dropped — better to skip a pair than to produce one with a None
    as_of that crashes the downstream recall."""
    item = _item(sessions=[(1, "not a date", ["D1:1"])])
    item["qa"] = [_qa("q1", "a1", ["D1:1"])]
    assert build_as_of_pairs(item) == []


def test_synthesizer_skips_qa_with_session_evidence_outside_item():
    """LoCoMo evidence sometimes references sessions that don't exist
    in the conversation (corpus quirks). Synthesizer must skip rather
    than index-error."""
    item = _item(sessions=[(1, "1:00 pm on 8 May, 2023", ["D1:1"])])
    item["qa"] = [_qa("q1", "a1", ["D99:1"])]
    assert build_as_of_pairs(item) == []


def test_synthesizer_pairs_have_unique_as_of_per_question():
    item = _item(
        sessions=[
            (1, "1:00 pm on 8 May, 2023", ["D1:1"]),
            (2, "1:00 pm on 9 May, 2023", ["D2:1"]),
            (3, "1:00 pm on 10 May, 2023", ["D3:1"]),
            (4, "1:00 pm on 11 May, 2023", ["D4:1"]),
        ],
    )
    item["qa"] = [_qa("q1", "a1", ["D1:1", "D2:1", "D3:1", "D4:1"])]
    pairs = build_as_of_pairs(item)
    by_q: dict[str, list[AsOfPair]] = {}
    for p in pairs:
        by_q.setdefault(p.question, []).append(p)
    for q, plist in by_q.items():
        as_ofs = [p.as_of for p in plist]
        assert len(set(as_ofs)) == len(as_ofs), f"{q}: duplicate as_of in {as_ofs}"


def test_synthesizer_scenario_count_invariant():
    """``before`` + ``after`` always appear once per QA pair that
    survives the filters; ``spanning`` appears once per intermediate
    evidence session."""
    item = _item(
        sessions=[
            (1, "1:00 pm on 8 May, 2023", ["D1:1"]),
            (2, "1:00 pm on 9 May, 2023", ["D2:1"]),
            (3, "1:00 pm on 10 May, 2023", ["D3:1"]),
            (4, "1:00 pm on 11 May, 2023", ["D4:1"]),
        ],
    )
    item["qa"] = [_qa("q1", "a1", ["D1:1", "D2:1", "D3:1", "D4:1"])]
    pairs = build_as_of_pairs(item)
    counts: dict[AsOfScenario, int] = {"before": 0, "after": 0, "spanning": 0}
    for p in pairs:
        counts[p.scenario] += 1
    assert counts["before"] == 1
    assert counts["after"] == 1
    assert counts["spanning"] == 2  # intermediates: session 2 and session 3


def test_cap_per_scenario_balances_counts():
    item = _item(
        sessions=[
            (1, "1:00 pm on 8 May, 2023", ["D1:1"]),
            (2, "1:00 pm on 9 May, 2023", ["D2:1"]),
            (3, "1:00 pm on 10 May, 2023", ["D3:1"]),
        ],
    )
    item["qa"] = [_qa(f"q{i}", f"a{i}", ["D1:1", "D2:1", "D3:1"]) for i in range(5)]
    pairs = build_as_of_pairs(item)
    # 5 qa pairs × 3 scenarios each = 15
    assert len(pairs) == 15
    capped = cap_per_scenario(pairs, per_scenario=2)
    counts: dict[AsOfScenario, int] = {"before": 0, "after": 0, "spanning": 0}
    for p in capped:
        counts[p.scenario] += 1
    assert counts == {"before": 2, "after": 2, "spanning": 2}


# --- Internal helpers ----------------------------------------------------


def test_session_dates_orders_by_number():
    item = _item(
        sessions=[
            (2, "1:00 pm on 9 May, 2023", ["D2:1"]),
            (1, "1:00 pm on 8 May, 2023", ["D1:1"]),
        ],
    )
    dates = _session_dates(item)
    assert [d.number for d in dates] == [1, 2]


def test_evidence_sessions_parses_string_repr():
    # LoCoMo sometimes stores evidence as a string of a list — the dual-memory
    # taskset handles both forms; the synthesizer must too.
    assert _evidence_sessions("['D1:3', 'D2:5']") == (1, 2)
    assert _evidence_sessions(["D3:1"]) == (3,)
    assert _evidence_sessions("garbage") == ()
    assert _evidence_sessions(None) == ()


def test_as_of_pair_is_frozen():
    p = AsOfPair(
        question="q",
        gold_answer="a",
        as_of=__import__("datetime").datetime(2024, 1, 1, tzinfo=__import__("datetime").timezone.utc),
        scenario="before",
        evidence_sessions=(1,),
        conv_id="c",
        locomo_category=1,
        explanation="e",
    )
    with pytest.raises((AttributeError, TypeError)):
        p.scenario = "after"  # type: ignore[misc]
