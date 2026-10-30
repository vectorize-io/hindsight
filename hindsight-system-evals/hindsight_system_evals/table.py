"""Per-category results of several runs side by side, one column per run.

    uv run pytest evals --full --output gemma.json      # once per model
    uv run eval-table gemma.json sol.json luna.json

Prints a markdown table: operation, category, and each run's correct/total, so a
model that is strong on one category and weak on another shows it. Traps count
as failures here (a trap is never ``correct``); the run report keeps them apart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hindsight_system_evals.report import EvalRecord, ModelConfig, by_category


def table(paths: list[Path]) -> str:
    columns = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        label = ModelConfig.model_validate(data["model_config"]).hindsight.model
        columns.append((label, by_category([EvalRecord.model_validate(item) for item in data["items"]])))

    rows = sorted({(op, cat) for _, ops in columns for op, cats in ops.items() for cat in cats})
    lines = [
        "| operation | category | " + " | ".join(label for label, _ in columns) + " |",
        "|---|---|" + "---|" * len(columns),
    ]
    for op, cat in rows:
        cells = []
        for _, ops in columns:
            summary = ops.get(op, {}).get(cat)
            cells.append(f"{summary.correct}/{summary.total}" if summary else "–")
        lines.append(f"| {op} | {cat} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: eval-table RUN.json [RUN.json ...]  (each from `pytest evals --output RUN.json`)")
    print(table([Path(p) for p in sys.argv[1:]]))
