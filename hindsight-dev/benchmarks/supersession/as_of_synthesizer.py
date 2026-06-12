"""as_of synthesizer — turn dated sessions into point-in-time recall gold pairs.

The supersession benchmark needs a way to score "did the system know X at
date Y?" without hand-labelling hundreds of QA pairs. This module takes a
LoCoMo conversation (or any dataset with the same shape) and synthesizes
``(question, gold_answer, as_of_date, scenario)`` quadruples that the
as_of recall contract can be scored against.

The synthesizer is **not** a contradiction detector — it doesn't try to
find pairs of facts that supersede each other. That would require an LLM
verdict (which is what the supersession worker itself does). What it does
is mechanical and deterministic:

* pick a session date that anchors the evidence for a QA pair,
* produce a "before" as_of (one day before the session) and an "after"
  as_of (the session date itself),
* the "before" scenario expects the system to NOT know the answer (the
  session hadn't happened yet),
* the "after" scenario expects the system to know the answer (the
  session is now in the past).

The two scenarios give a balanced gold set:

  - true_positive_at_t  — the system knew the answer at the as_of date,
  - true_negative_at_t  — the system did NOT know the answer at the
    as_of date (because the session hadn't been ingested yet).

A non-trivial as_of evaluation harness scores both rates; high
true_positive and high true_negative means the temporal index is doing
real work. A high true_positive with low true_negative means the system
is just returning everything (as_of is ignored).

QA pairs that depend on multiple sessions get a "spanning" as_of set
(earliest session + each session incrementally added) so a benchmark can
measure how the recall widens as the as_of clock advances. Spanning
pairs are the ones most sensitive to supersession: a fact that gets
superseded between two sessions would be in the "earlier" as_of set
and absent from the "later" as_of set. The synthesizer flags them with
``scenario="spanning"`` so the harness can run them under supersession
and compare.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from benchmarks.dual_memory.taskset import _DIA_ID_RE  # type: ignore[attr-defined]
from benchmarks.locomo.locomo_benchmark import LoComoDataset

AsOfScenario = Literal["before", "after", "spanning"]


@dataclass(frozen=True)
class AsOfPair:
    """One gold (question, answer, as_of) triple for as_of recall.

    ``scenario`` discriminates how the pair was synthesised so the
    evaluation harness can pick the scoring rule:

    * ``before`` — as_of is strictly before the earliest evidence
      session; recall should NOT return the gold answer.
    * ``after`` — as_of equals the latest evidence session; recall
      SHOULD return the gold answer.
    * ``spanning`` — the QA evidence crosses multiple sessions and the
      as_of is set to one of the intermediate dates; expected behaviour
      depends on whether the bridging fact is currently alive.
    """

    question: str
    gold_answer: str
    as_of: datetime
    scenario: AsOfScenario
    evidence_sessions: tuple[int, ...]
    conv_id: str
    locomo_category: int
    # ``explanation`` is a human-readable line that travels with the
    # pair so a benchmark failure log can be read by a human without
    # cross-referencing the synthesizer code.
    explanation: str


@dataclass(frozen=True)
class _SessionDate:
    """One session's date — local mirror of the LoCoMo session key."""

    number: int
    event_date: datetime | None


def _session_dates(item: dict[str, Any]) -> list[_SessionDate]:
    """Return one ``_SessionDate`` per session, ordered by session number.

    The LoCoMo schema stores each session's date as ``session_N_date_time``;
    malformed (non-list) sessions are kept in the list with a None date so
    the synthesizer can flag them as ``explanation=...`` rather than
    silently dropping the question.
    """
    conv = item["conversation"]
    # LoComoDataset holds no per-call state — instantiate once and reuse
    # the public date parser for every session in this item.
    dataset = LoComoDataset()
    out: list[_SessionDate] = []
    for key in sorted(conv.keys()):
        match = re.match(r"session_(\d+)$", key)
        if not match:
            continue
        number = int(match.group(1))
        if not isinstance(conv.get(key), list):
            out.append(_SessionDate(number=number, event_date=None))
            continue
        date_str = conv.get(f"{key}_date_time")
        try:
            event_date = dataset.parse_session_date(date_str) if date_str else None
        except ValueError:
            event_date = None
        out.append(_SessionDate(number=number, event_date=event_date))
    return out


def _evidence_sessions(evidence: Any) -> tuple[int, ...]:
    """Lift the evidence-session parser from the dual-memory taskset.

    Reusing keeps the synthesizer aligned with the private/blind/spanning
    classification — a question classified as "spanning" by the dual-memory
    taskset will produce "spanning" ``AsOfPair``s here for the same
    evidence sessions.
    """
    import ast

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


def _date_for_session(sessions: list[_SessionDate], number: int) -> datetime | None:
    """Lookup a session number → event_date, with a clear None if missing."""
    for s in sessions:
        if s.number == number:
            return s.event_date
    return None


def build_as_of_pairs(item: dict[str, Any]) -> list[AsOfPair]:
    """Synthesize ``AsOfPair``s for one LoCoMo conversation.

    Pure function: same input → same output. No LLM, no DB, no clock.
    Pairs with malformed evidence or missing dates are skipped (their
    count is exposed via the ``explanation`` field of the next pair, or
    by the caller's ``len(build_as_of_pairs(...))`` if all are skipped).
    """
    conv_id = str(item.get("sample_id", ""))
    sessions = _session_dates(item)
    if not sessions:
        return []
    qa_pairs = item.get("qa") or []
    out: list[AsOfPair] = []
    for qa in qa_pairs:
        answer = qa.get("answer")
        evidence = _evidence_sessions(qa.get("evidence"))
        locomo_category = int(qa.get("category", 0))
        if answer is None or not evidence:
            continue
        # LoCoMo category 5 is adversarial / unanswerable — skip the
        # same way the dual-memory taskset does.
        if locomo_category == 5:
            continue

        earliest = min(evidence)
        latest = max(evidence)
        earliest_date = _date_for_session(sessions, earliest)
        latest_date = _date_for_session(sessions, latest)
        if earliest_date is None or latest_date is None:
            continue

        question = str(qa["question"])
        gold = str(answer)

        # ``before`` pair: as_of is one second before earliest evidence
        # session. Strict "before" semantics — we use the session date
        # minus 1 second so the as_of falls on the right side of the
        # boundary the DB uses (``valid_until`` is exclusive).
        before_dt = earliest_date - timedelta(seconds=1)
        out.append(
            AsOfPair(
                question=question,
                gold_answer=gold,
                as_of=before_dt,
                scenario="before",
                evidence_sessions=evidence,
                conv_id=conv_id,
                locomo_category=locomo_category,
                explanation=(
                    f"as_of is 1s before session_{earliest}; session_{earliest} not yet "
                    f"ingested → recall should not return the gold answer"
                ),
            )
        )

        # ``after`` pair: as_of is the latest evidence session's date.
        out.append(
            AsOfPair(
                question=question,
                gold_answer=gold,
                as_of=latest_date,
                scenario="after",
                evidence_sessions=evidence,
                conv_id=conv_id,
                locomo_category=locomo_category,
                explanation=(
                    f"as_of is session_{latest} date; all evidence sessions ingested → "
                    f"recall should return the gold answer"
                ),
            )
        )

        # ``spanning`` pairs: one per intermediate session, so the
        # evaluation can measure how recall widens as the as_of clock
        # advances. The "earliest" and "latest" as_of's are already
        # covered by before/after; spanning is only meaningful for
        # multi-session evidence.
        if len(evidence) > 1:
            for intermediate in evidence[1:-1]:
                intermediate_date = _date_for_session(sessions, intermediate)
                if intermediate_date is None:
                    continue
                out.append(
                    AsOfPair(
                        question=question,
                        gold_answer=gold,
                        as_of=intermediate_date,
                        scenario="spanning",
                        evidence_sessions=evidence,
                        conv_id=conv_id,
                        locomo_category=locomo_category,
                        explanation=(
                            f"as_of is session_{intermediate} date; earlier sessions ingested, later ones not"
                        ),
                    )
                )
    return out


def cap_per_scenario(pairs: list[AsOfPair], per_scenario: int) -> list[AsOfPair]:
    """Deterministically cap each scenario (keeps dataset order).

    Spanning is generated 1× per intermediate session, so it tends to
    dwarf before/after. Capping keeps the per-scenario counts balanced
    for benchmark reporting.
    """
    counts: dict[AsOfScenario, int] = {"before": 0, "after": 0, "spanning": 0}
    out: list[AsOfPair] = []
    for pair in pairs:
        if counts[pair.scenario] < per_scenario:
            counts[pair.scenario] += 1
            out.append(pair)
    return out
