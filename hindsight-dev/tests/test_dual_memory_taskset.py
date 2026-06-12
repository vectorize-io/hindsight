"""Unit tests for the dual-memory task-set construction (pure functions)."""

from benchmarks.dual_memory.taskset import build_tasks, cap_per_category, split_sessions


def _item(qa: list[dict], sessions: int = 4) -> dict:
    conv: dict = {"speaker_a": "Caroline", "speaker_b": "Melanie"}
    for n in range(1, sessions + 1):
        conv[f"session_{n}"] = [{"speaker": "Caroline", "dia_id": f"D{n}:1", "text": "hi"}]
        conv[f"session_{n}_date_time"] = "1:00 pm on 8 May, 2023"
    return {"sample_id": "conv-1", "conversation": conv, "qa": qa}


def test_split_alternates_sessions():
    split = split_sessions(_item([], sessions=5))
    assert split.a_sessions == frozenset({1, 3, 5})
    assert split.b_sessions == frozenset({2, 4})
    assert split.owner(3) == "a" and split.owner(4) == "b"


def test_single_owner_evidence_yields_private_and_blind_pair():
    item = _item([{"question": "q?", "answer": "x", "evidence": ["D1:3"], "category": 2}])
    tasks = build_tasks(item, split_sessions(item))
    assert {(t.category, t.asker) for t in tasks} == {("private", "a"), ("blind", "b")}
    # Difficulty-matched: both tasks share question and gold answer.
    assert len({t.question for t in tasks}) == 1


def test_spanning_evidence_yields_one_spanning_task():
    item = _item([{"question": "q?", "answer": "x", "evidence": ["D1:1", "D2:5"], "category": 1}])
    tasks = build_tasks(item, split_sessions(item))
    assert [t.category for t in tasks] == ["spanning"]
    assert tasks[0].evidence_sessions == (1, 2)


def test_adversarial_and_malformed_excluded():
    item = _item(
        [
            {"question": "q1?", "answer": "x", "evidence": ["D1:1"], "category": 5},  # adversarial
            {"question": "q2?", "answer": "x", "evidence": [], "category": 1},  # no evidence
            {"question": "q3?", "answer": "x", "evidence": ["D99:1"], "category": 1},  # unknown session
            {"question": "q4?", "answer": None, "evidence": ["D1:1"], "category": 1},  # no answer
        ]
    )
    assert build_tasks(item, split_sessions(item)) == []


def test_string_repr_evidence_parsed():
    # locomo10.json stores evidence as a list, but some dumps carry str(list).
    item = _item([{"question": "q?", "answer": "x", "evidence": "['D2:3']", "category": 4}])
    tasks = build_tasks(item, split_sessions(item))
    assert tasks and tasks[0].evidence_sessions == (2,)
    assert {t.asker for t in tasks if t.category == "private"} == {"b"}


def test_cap_per_category_is_deterministic():
    item = _item([{"question": f"q{i}?", "answer": "x", "evidence": ["D1:1"], "category": 4} for i in range(5)])
    tasks = build_tasks(item, split_sessions(item))  # 5 private + 5 blind
    capped = cap_per_category(tasks, 2)
    assert sum(1 for t in capped if t.category == "private") == 2
    assert sum(1 for t in capped if t.category == "blind") == 2
    assert capped[0].question == "q0?"
