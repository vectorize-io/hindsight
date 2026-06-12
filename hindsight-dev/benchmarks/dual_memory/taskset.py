"""Multi-agent task set derived from LoCoMo (pure functions, no I/O).

Each LoCoMo conversation has two speakers and ~19 dated sessions. We turn it
into a two-agent scenario by splitting the sessions: agent A holds the
odd-numbered sessions, agent B the even-numbered ones — as if two assistants
had each logged half of the encounters. LoCoMo's per-question ``evidence``
(dialog turn ids like ``D7:3`` → session 7) then classifies every question
mechanically:

* ``private``  — all evidence sessions belong to the asker. Answerable from
  the asker's own bank; the dual arm must not degrade here (decision rule:
  regression > 5pp → tool routing needs rework).
* ``blind``    — all evidence sessions belong to the OTHER agent. The asker's
  private bank cannot answer; only a shared world graph can (the C-track
  value case). Generated from the same QA pairs as ``private`` by flipping
  the asker, so the two categories are perfectly matched in difficulty.
* ``spanning`` — evidence spans both agents' sessions. Needs the asker's own
  memory AND shared knowledge (the mixed case).

LoCoMo category 5 (adversarial/unanswerable) is excluded: arm differences on
refusals measure prompt style, not memory quality.
"""

import ast
import re
from dataclasses import dataclass
from typing import Any, Literal

_DIA_ID_RE = re.compile(r"D(\d+):\d+")

TaskCategory = Literal["private", "blind", "spanning"]


@dataclass(frozen=True)
class SessionSplit:
    """Which speaker-agent owns which LoCoMo session numbers."""

    conv_id: str
    agent_a: str  # speaker name
    agent_b: str
    a_sessions: frozenset[int]
    b_sessions: frozenset[int]

    def owner(self, session: int) -> str:
        return "a" if session in self.a_sessions else "b"


@dataclass(frozen=True)
class Task:
    """One question posed by one agent under one capability category."""

    conv_id: str
    question: str
    gold_answer: str
    category: TaskCategory
    asker: str  # "a" | "b"
    evidence_sessions: tuple[int, ...]
    locomo_category: int


def split_sessions(item: dict[str, Any]) -> SessionSplit:
    """Assign odd-numbered sessions to agent A, even-numbered to agent B."""
    conv = item["conversation"]
    numbers = sorted(
        int(k.split("_")[1])
        for k in conv
        if k.startswith("session_") and not k.endswith("_date_time") and isinstance(conv[k], list)
    )
    return SessionSplit(
        conv_id=str(item["sample_id"]),
        agent_a=conv["speaker_a"],
        agent_b=conv["speaker_b"],
        a_sessions=frozenset(n for n in numbers if n % 2 == 1),
        b_sessions=frozenset(n for n in numbers if n % 2 == 0),
    )


def _evidence_sessions(evidence: Any) -> tuple[int, ...]:
    """Parse LoCoMo evidence (list or its string repr) into session numbers."""
    if isinstance(evidence, str):
        try:
            evidence = ast.literal_eval(evidence)
        except (ValueError, SyntaxError):
            return ()
    if not isinstance(evidence, list):
        return ()
    sessions: set[int] = set()
    for dia_id in evidence:
        match = _DIA_ID_RE.search(str(dia_id))
        if match:
            sessions.add(int(match.group(1)))
    return tuple(sorted(sessions))


def build_tasks(item: dict[str, Any], split: SessionSplit) -> list[Task]:
    """Classify every usable QA pair of one conversation into tasks."""
    tasks: list[Task] = []
    for qa in item.get("qa", []):
        locomo_category = int(qa.get("category", 0))
        if locomo_category == 5:  # adversarial/unanswerable
            continue
        answer = qa.get("answer")
        sessions = _evidence_sessions(qa.get("evidence"))
        if answer is None or not sessions:
            continue
        # Sessions outside the split (malformed evidence) disqualify the pair.
        known = split.a_sessions | split.b_sessions
        if any(s not in known for s in sessions):
            continue

        owners = {split.owner(s) for s in sessions}
        common = {
            "conv_id": split.conv_id,
            "question": str(qa["question"]),
            "gold_answer": str(answer),
            "evidence_sessions": sessions,
            "locomo_category": locomo_category,
        }
        if owners == {"a"} or owners == {"b"}:
            owner = owners.pop()
            other = "b" if owner == "a" else "a"
            # Same QA pair, two askers: difficulty-matched private/blind tasks.
            tasks.append(Task(category="private", asker=owner, **common))
            tasks.append(Task(category="blind", asker=other, **common))
        else:
            tasks.append(Task(category="spanning", asker="a", **common))
    return tasks


def cap_per_category(tasks: list[Task], per_category: int) -> list[Task]:
    """Deterministically cap each category (keeps dataset order)."""
    kept: list[Task] = []
    counts: dict[TaskCategory, int] = {"private": 0, "blind": 0, "spanning": 0}
    for task in tasks:
        if counts[task.category] < per_category:
            counts[task.category] += 1
            kept.append(task)
    return kept
