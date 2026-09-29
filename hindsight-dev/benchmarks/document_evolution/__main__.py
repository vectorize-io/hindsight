"""CLI: run one build, then compare two runs.

# once per build, against a server running that build
uv run python -m benchmarks.document_evolution run --api-url http://localhost:8888 --build main --out main.json
uv run python -m benchmarks.document_evolution run --api-url http://localhost:9999 --build branch --out branch.json

# then, from either build
uv run python -m benchmarks.document_evolution compare main.json branch.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from .corpus import CASES, case_by_name
from .judge import judge_coverage, judge_preference
from .metrics import ShapeDelta, damage_in_untouched_sections
from .runner import BenchmarkArtifact, CaseRun, run_build

console = Console()
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "document_evolution"


class StructuralSummary(BaseModel):
    """The no-LLM half of a build's result, in one typed row.

    Every field counts something lost, so zero is the good value and the report
    can render the whole model without knowing which metric is which.
    """

    model_config = ConfigDict(extra="forbid")
    rounds: int = 0
    collapsed_tables: int = 0
    table_rows_lost: int = 0
    nesting_lost: int = 0
    hard_breaks_lost: int = 0
    fences_lost: int = 0
    quotes_lost: int = 0
    headings_lost: int = 0
    damaged_rounds: int = 0
    drifted_sections: int = 0
    delta_applied_rounds: int = 0
    failed_rounds: int = 0
    ops_skipped: int = 0
    median_ms: float = 0.0


class _MeanCoverage:
    """Coverage averaged over repetitions, in the shape the report reads."""

    def __init__(self, recall: float, staleness: float) -> None:
        self.recall = recall
        self.staleness = staleness


def _mean_coverage(results: list) -> _MeanCoverage:
    if not results:
        return _MeanCoverage(recall=0.0, staleness=0.0)
    return _MeanCoverage(
        recall=statistics.mean(r.recall for r in results),
        staleness=statistics.mean(r.staleness for r in results),
    )


def _structural_summary(runs: list[CaseRun]) -> StructuralSummary:
    """Damage counts across every round of every run — the no-LLM half.

    Recomputed here from the stored documents rather than read off the runner's
    per-round fields, so a sharper metric can be applied to results that already
    exist instead of costing another few hundred LLM calls to re-measure.

    Damage is counted only in sections no operation named. A refresh that
    rewrites a section it deliberately targeted may legitimately restructure it;
    scoring that as corruption would punish the model for doing its job.
    """
    damage: list[ShapeDelta] = []
    rounds = []
    for run in runs:
        for previous, current in zip(run.rounds, run.rounds[1:], strict=False):
            rounds.append(current)
            damage.append(
                damage_in_untouched_sections(previous.document, current.document, set(current.touched_sections))
            )
    if not rounds:
        return StructuralSummary()
    return StructuralSummary(
        rounds=len(rounds),
        collapsed_tables=sum(d.collapsed_tables_introduced for d in damage),
        table_rows_lost=sum(d.table_rows_lost for d in damage),
        nesting_lost=sum(d.indented_list_lines_lost for d in damage),
        hard_breaks_lost=sum(d.hard_break_lines_lost for d in damage),
        fences_lost=sum(d.fenced_blocks_lost for d in damage),
        quotes_lost=sum(d.blockquote_lines_lost for d in damage),
        headings_lost=sum(len(d.headings_lost) for d in damage),
        damaged_rounds=sum(d.damaged for d in damage),
        drifted_sections=sum(len(r.drifted_sections) for r in rounds),
        delta_applied_rounds=sum(r.delta_applied for r in rounds),
        failed_rounds=sum(bool(r.error or r.refresh_skipped) for r in rounds),
        ops_skipped=sum(r.ops_skipped for r in rounds),
        median_ms=statistics.median(r.duration_ms for r in rounds),
    )


async def _run(args: argparse.Namespace) -> None:
    cases = [case_by_name(n) for n in args.case] if args.case else CASES
    artifact = await run_build(
        args.api_url,
        cases,
        build=args.build,
        seeding=args.seeding,
        db_url=args.db_url,
        repetitions=args.repetitions,
        settle_seconds=args.settle,
        keep_banks=args.keep_banks,
    )
    out = Path(args.out) if args.out else DEFAULT_RESULTS_DIR / f"{args.build}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(artifact.model_dump_json(indent=2))
    console.print(f"[green]wrote[/green] {out}")
    _print_structural(artifact)


def _print_structural(artifact: BenchmarkArtifact) -> None:
    table = Table(title=f"structural damage — {artifact.build}")
    table.add_column("case")
    table.add_column("rounds", justify="right")
    table.add_column("damaged", justify="right")
    table.add_column("collapsed tables", justify="right")
    table.add_column("rows lost", justify="right")
    table.add_column("drifted sections", justify="right")
    for case in sorted({r.case for r in artifact.runs}):
        summary = _structural_summary([r for r in artifact.runs if r.case == case])
        table.add_row(
            case,
            str(summary.rounds),
            str(summary.damaged_rounds),
            str(summary.collapsed_tables),
            str(summary.table_rows_lost),
            str(summary.drifted_sections),
        )
    console.print(table)


async def _compare(args: argparse.Namespace) -> None:
    left = BenchmarkArtifact.model_validate_json(Path(args.left).read_text())
    right = BenchmarkArtifact.model_validate_json(Path(args.right).read_text())

    _print_structural(left)
    _print_structural(right)

    comparison = Table(title="structural damage, side by side (lower is better)")
    comparison.add_column("metric")
    comparison.add_column(left.build, justify="right")
    comparison.add_column(right.build, justify="right")
    left_summary = _structural_summary(left.runs)
    right_summary = _structural_summary(right.runs)
    left_row = left_summary.model_dump()
    right_row = right_summary.model_dump()
    for key in sorted(left_row):
        comparison.add_row(key, str(left_row[key]), str(right_row[key]))
    console.print(comparison)

    if args.no_judge:
        return

    verdicts = Table(title="content quality (judged)")
    verdicts.add_column("case")
    verdicts.add_column(f"{left.build} recall", justify="right")
    verdicts.add_column(f"{right.build} recall", justify="right")
    verdicts.add_column(f"{left.build} stale", justify="right")
    verdicts.add_column(f"{right.build} stale", justify="right")
    # Length sits next to preference on purpose: LLM judges favour longer
    # answers, so a preference that tracks the length column is a confound to
    # read sceptically rather than a quality signal.
    verdicts.add_column("mean bytes", justify="right")
    verdicts.add_column("preferred")

    report: dict[str, dict[str, float | str]] = {}
    for case_name in sorted({r.case for r in left.runs} & {r.case for r in right.runs}):
        case = case_by_name(case_name)
        stated = [r.asserts for r in case.rounds]
        superseded = [r.supersedes for r in case.rounds if r.supersedes]

        left_runs = [r for r in left.runs if r.case == case_name]
        right_runs = [r for r in right.runs if r.case == case_name]

        # Every repetition is judged, not just the first: one LLM run is an
        # anecdote, and coverage is exactly the metric that varies between them.
        left_covs, right_covs = await asyncio.gather(
            asyncio.gather(*(judge_coverage(r.final_document, stated, superseded) for r in left_runs)),
            asyncio.gather(*(judge_coverage(r.final_document, stated, superseded) for r in right_runs)),
        )
        left_cov = _mean_coverage(left_covs)
        right_cov = _mean_coverage(right_covs)
        preferences = await asyncio.gather(
            *(
                judge_preference(case.topic, left.build, lhs.final_document, right.build, rhs.final_document)
                for lhs, rhs in zip(left_runs, right_runs, strict=False)
            )
        )
        wins = {left.build: 0, right.build: 0, "tie": 0}
        for preference in preferences:
            wins[preference.winner] = wins.get(preference.winner, 0) + 1
        preferred = ", ".join(f"{label}={count}" for label, count in wins.items())

        left_bytes = statistics.mean(len(r.final_document) for r in left_runs)
        right_bytes = statistics.mean(len(r.final_document) for r in right_runs)
        verdicts.add_row(
            case_name,
            f"{left_cov.recall:.0%}",
            f"{right_cov.recall:.0%}",
            f"{left_cov.staleness:.0%}",
            f"{right_cov.staleness:.0%}",
            f"{left_bytes:.0f} / {right_bytes:.0f}",
            preferred,
        )
        report[case_name] = {
            f"{left.build}_recall": left_cov.recall,
            f"{right.build}_recall": right_cov.recall,
            f"{left.build}_staleness": left_cov.staleness,
            f"{right.build}_staleness": right_cov.staleness,
            "preference": preferred,
            f"{left.build}_mean_bytes": left_bytes,
            f"{right.build}_mean_bytes": right_bytes,
        }

    console.print(verdicts)
    console.print(
        "[dim]recall: facts that reached the document. stale: superseded claims still stated. "
        f"mean bytes: {left.build} / {right.build} — judges favour longer documents, so read a "
        "preference that tracks this column sceptically.[/dim]"
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "left": left.build,
                    "right": right.build,
                    "structural": {
                        left.build: left_summary.model_dump(),
                        right.build: right_summary.model_dump(),
                    },
                    "content": report,
                },
                indent=2,
            )
        )
        console.print(f"[green]wrote[/green] {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="document_evolution", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="drive one build through the corpus")
    run.add_argument("--api-url", required=True)
    run.add_argument("--build", required=True, help="label for this build, e.g. a git ref")
    run.add_argument("--case", action="append", help="run only this case (repeatable)")
    run.add_argument("--seeding", choices=["authored", "generated"], default="generated")
    run.add_argument("--db-url", help="postgres URL; required for --seeding authored")
    run.add_argument("--repetitions", type=int, default=1)
    run.add_argument("--settle", type=float, default=8.0, help="seconds to wait for consolidation")
    run.add_argument("--keep-banks", action="store_true", help="leave the benchmark banks behind for inspection")
    run.add_argument("--out")

    compare = sub.add_parser("compare", help="score two runs against each other")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument("--no-judge", action="store_true", help="structural metrics only, no LLM calls")
    compare.add_argument("--out")

    args = parser.parse_args()
    asyncio.run(_run(args) if args.command == "run" else _compare(args))


if __name__ == "__main__":
    main()
