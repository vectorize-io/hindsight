"""Case ids, the per-category report, the table that compares runs, and the two-case floor."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from hindsight_system_evals.report import EvalRecord, by_category, case_params
from hindsight_system_evals.table import table


@dataclass(frozen=True)
class _Case:
    id: str
    category: str


def test_case_params_id_by_category_and_mark_the_rest_full() -> None:
    params = case_params([_Case("a", "temporal"), _Case("b", "language")], minimum={"a"})

    assert [p.id for p in params] == ["temporal/a", "language/b"]
    assert [[m.name for m in p.marks] for p in params] == [[], ["full"]]


def _record(kind: str, category: str, correct: bool) -> EvalRecord:
    return EvalRecord(
        kind=kind, question_id="q", category=category, correct=correct, hit_trap=False, bank_id="b", reason=""
    )


def test_by_category_groups_kinds_under_their_operation() -> None:
    records = [
        _record("knowledge_page", "update", True),
        _record("delta_update", "update", False),
        _record("source_priority", "update", True),
    ]

    grouped = by_category(records)

    assert grouped["refresh_pages"]["update"].total == 2
    assert grouped["refresh_pages"]["update"].correct == 1
    assert grouped["reflect"]["update"].correct == 1


def _run(path: Path, model: str, records: list[EvalRecord]) -> Path:
    report = {
        "model_config": {"hindsight": {"model": model}, "judge": {"model": "judge"}},
        "items": [r.model_dump() for r in records],
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_table_puts_each_run_in_its_own_column(tmp_path: Path) -> None:
    a = _run(tmp_path / "a.json", "gemma", [_record("retain", "temporal", True), _record("retain", "temporal", False)])
    b = _run(tmp_path / "b.json", "sol", [_record("retain", "temporal", True), _record("retain", "mode", True)])

    lines = table([a, b]).splitlines()

    assert lines[0] == "| operation | category | gemma | sol |"
    assert "| retain | mode | – | 1/1 |" in lines
    assert "| retain | temporal | 1/2 | 1/1 |" in lines


def test_every_category_has_at_least_two_cases() -> None:
    """Read from the collected evals, so a new case lands in a group someone can compare."""
    root = Path(__file__).resolve().parents[1]
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-qq", "--full", "-p", "no:cacheprovider", "evals"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # evals/<operation>/test_<operation>.py::test_x[<category>/<case>]; -qq, as the config adds -v
    groups = Counter(
        f"{match.group(1)}:{match.group(2)}"
        for match in re.finditer(r"^evals/([^/]+)/\S+\[([^/\]]+)/", collected, re.M)
    )
    assert groups, collected
    thin = {group: n for group, n in groups.items() if n < 2}
    assert not thin, f"categories with fewer than two cases: {thin}"
