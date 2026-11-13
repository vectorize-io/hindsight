"""Per-category results of several runs side by side, one column per run.

    uv run pytest evals --full --output gemma.json      # once per model
    uv run eval-table gemma.json sol.json luna.json

Prints a markdown table: operation, category, and each run's correct/total, so a
model that is strong on one category and weak on another shows it. Traps count
as failures here (a trap is never ``correct``); the run report keeps them apart.

A run with knowledge-bank suites (``evals/knowledge``) gets a second table: each
suite's scores, ingest rate and search (or agent answer) latency, one column per run — how one
knowledge embedder is compared with another. Such a run is labelled by its
knowledge embedder rather than its LLM, which those suites never call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hindsight_system_evals.report import EvalRecord, KnowledgeResult, ModelConfig, by_category


def _knowledge_rows(results: list[KnowledgeResult]) -> dict[tuple[str, str], str]:
    cells: dict[tuple[str, str], str] = {}
    for r in results:
        name = f"{r.category}/{r.case}"
        for metric, value in r.metrics.items():
            cells[(name, metric)] = f"{value:.1f}"
        if r.ingest:
            cells[(name, "ingest docs/s")] = f"{r.ingest.documents_per_s:.2f}"
        if r.latency:
            cells[(name, "search p50 / p95 ms")] = f"{r.latency.client_p50_ms:.0f} / {r.latency.client_p95_ms:.0f}"
        if r.agent_latency:
            a = r.agent_latency
            cells[(name, "answer p50 / p95 ms")] = f"{a.answer_p50_ms:.0f} / {a.answer_p95_ms:.0f}"
    return cells


def table(paths: list[Path]) -> str:
    columns = []
    knowledge = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        label = ModelConfig.model_validate(data["model_config"]).hindsight.model
        results = [KnowledgeResult.model_validate(k) for k in data.get("knowledge", [])]
        if results and not data["items"]:
            label = results[0].embedder
        columns.append((label, by_category([EvalRecord.model_validate(item) for item in data["items"]])))
        knowledge.append((label, _knowledge_rows(results)))

    rows = sorted({(op, cat) for _, ops in columns for op, cats in ops.items() for cat in cats})
    lines = []
    if rows:
        lines += [
            "| operation | category | " + " | ".join(label for label, _ in columns) + " |",
            "|---|---|" + "---|" * len(columns),
        ]
    for op, cat in rows:
        cells = []
        for _, ops in columns:
            summary = ops.get(op, {}).get(cat)
            cells.append(f"{summary.correct}/{summary.total}" if summary else "–")
        lines.append(f"| {op} | {cat} | " + " | ".join(cells) + " |")

    keys = list(dict.fromkeys(key for _, cells in knowledge for key in cells))
    if keys:
        if lines:
            lines.append("")
        lines += [
            "| knowledge suite | metric | " + " | ".join(label for label, _ in knowledge) + " |",
            "|---|---|" + "---|" * len(knowledge),
        ]
        for name, metric in keys:
            lines.append(
                f"| {name} | {metric} | " + " | ".join(c.get((name, metric), "–") for _, c in knowledge) + " |"
            )
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: eval-table RUN.json [RUN.json ...]  (each from `pytest evals --output RUN.json`)")
    print(table([Path(p) for p in sys.argv[1:]]))
