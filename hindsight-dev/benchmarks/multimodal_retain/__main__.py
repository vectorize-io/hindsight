"""CLI: run the multimodal-retain benchmark and report both arms.

    uv run python -m benchmarks.multimodal_retain run --api-url http://localhost:8888

The server must be configured with a vision-capable retain LLM; without one the
multimodal arm is refused with 422 and the report says so rather than quietly
reporting zeros.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .corpus import ARTICLES, article_by_name
from .runner import ARMS, ArticleRun, BenchmarkArtifact, run_benchmark

console = Console()
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "multimodal_retain"


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def _arm_totals(runs: list[ArticleRun], arm: str) -> dict[str, int]:
    totals = {"facts": 0, "facts_total": 0, "correct": 0, "abstained": 0, "wrong": 0, "asked": 0, "failed": 0}
    for run in runs:
        for candidate in run.arms:
            if candidate.arm != arm:
                continue
            if candidate.retain_error:
                totals["failed"] += 1
                continue
            totals["facts"] += sum(1 for verdict in candidate.image_facts if verdict.present)
            totals["facts_total"] += len(candidate.image_facts)
            for entry in candidate.questions:
                totals["asked"] += 1
                if entry.verdict is None:
                    continue
                if entry.verdict.correct:
                    totals["correct"] += 1
                elif entry.verdict.abstained:
                    totals["abstained"] += 1
                else:
                    totals["wrong"] += 1
    return totals


def _report(artifact: BenchmarkArtifact) -> None:
    console.print()
    console.print(f"[bold]Multimodal retain[/bold] — build {artifact.build}, api {artifact.api_version or '?'}")

    refused = [arm for run in artifact.runs for arm in run.arms if arm.retain_error]
    if refused:
        # Only a 422 means "this LLM cannot read images". Anything else is an
        # ordinary server error and saying otherwise sends the reader to the wrong
        # place — a 500 from a schema mismatch once printed as a vision problem.
        vision = [arm for arm in refused if arm.retain_error.startswith("422")]
        console.print(f"\n[yellow]{len(refused)} retain(s) failed on the server.[/yellow]")
        if vision:
            console.print(
                "  Refused with 422: the configured retain LLM is not vision-capable — "
                "set a vision model, or HINDSIGHT_API_LLM_VISION=true for a gateway."
            )
        for arm in refused[:2]:
            console.print(f"  [dim]{arm.retain_error}[/dim]")

    table = Table(title="Answers to questions only the images can answer", show_lines=False)
    table.add_column("Arm")
    table.add_column("Image facts recalled", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Wrong", justify="right")
    table.add_column("Abstained", justify="right")

    for arm in ARMS:
        totals = _arm_totals(artifact.runs, arm)
        table.add_row(
            arm,
            f"{totals['facts']}/{totals['facts_total']} ({_ratio(totals['facts'], totals['facts_total']):.0%})",
            f"{totals['correct']}/{totals['asked']} ({_ratio(totals['correct'], totals['asked']):.0%})",
            str(totals["wrong"]),
            str(totals["abstained"]),
        )
    console.print()
    console.print(table)

    console.print(
        "\n[dim]Every question is unanswerable from the article's prose alone, so the text-only arm is the "
        "pre-feature baseline. 'Wrong' counts confidently incorrect answers — the failure inline images exist "
        "to remove — and is tracked apart from 'Abstained', which is a far better way to be unhelpful.[/dim]"
    )

    for run in artifact.runs:
        console.print(f"\n[bold]{run.article}[/bold]")
        for arm in run.arms:
            if arm.retain_error:
                console.print(f"  {arm.arm}: [red]retain refused[/red] — {arm.retain_error}")
                continue
            for entry in arm.questions:
                verdict = entry.verdict
                mark = "[green]✓[/green]" if verdict and verdict.correct else "[red]✗[/red]"
                if verdict and not verdict.correct and verdict.abstained:
                    mark = "[yellow]~[/yellow]"
                console.print(f"  {mark} {arm.arm}: {entry.answer.strip()[:160] or '(no answer)'}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="benchmarks.multimodal_retain")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run both arms against a live server.")
    run.add_argument("--api-url", default="http://localhost:8888")
    run.add_argument("--build", default="local", help="Label for this build, recorded in the artifact.")
    run.add_argument("--article", action="append", help="Run only these articles (repeatable).")
    run.add_argument("--out", type=Path, help="Where to write the artifact JSON.")
    run.add_argument("--keep-banks", action="store_true", help="Leave the benchmark banks behind for inspection.")

    report = sub.add_parser("report", help="Re-render the report from a saved artifact.")
    report.add_argument("artifact", type=Path)

    args = parser.parse_args()

    if args.command == "report":
        _report(BenchmarkArtifact.model_validate_json(args.artifact.read_text()))
        return

    articles = [article_by_name(name) for name in args.article] if args.article else ARTICLES
    run_id = uuid.uuid4().hex[:8]

    with console.status("Running...") as status:
        artifact = asyncio.run(
            run_benchmark(
                args.api_url,
                articles,
                build=args.build,
                run_id=run_id,
                keep_banks=args.keep_banks,
                progress=lambda label: status.update(f"Running {label}..."),
            )
        )

    out = args.out or (DEFAULT_RESULTS_DIR / f"{args.build}-{run_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2))

    _report(artifact)
    console.print(f"\n[dim]Artifact: {out}[/dim]")


if __name__ == "__main__":
    main()
