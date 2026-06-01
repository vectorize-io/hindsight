"""CLI entry point for mission-sandbox."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def cmd_init(args: argparse.Namespace) -> None:
    from .init import run_init

    run_init(
        project_path=Path(args.project),
        documents_path=Path(args.documents),
        bank_id=args.bank_id,
        api_url=args.api_url,
    )


def cmd_agent_label(args: argparse.Namespace) -> None:
    from .agent_labeler import run_agent_label

    run_agent_label(
        project_path=Path(args.project),
        instructions=args.instructions,
        model=args.model,
    )


def cmd_optimize(args: argparse.Namespace) -> None:
    from .optimizer import run_optimize

    run_optimize(
        project_path=Path(args.project),
        model=args.model,
    )


def cmd_run(args: argparse.Namespace) -> None:
    from .run import run_run

    run_run(project_path=Path(args.project))


def cmd_promote(args: argparse.Namespace) -> None:
    from .promote import run_promote

    run_promote(
        project_path=Path(args.project),
        target_bank=args.target_bank,
        backfill=args.backfill,
    )


def cmd_ui(args: argparse.Namespace) -> None:
    import subprocess

    ui_script = Path(__file__).parent / "ui.py"
    subprocess.run(
        ["streamlit", "run", str(ui_script), "--", str(Path(args.project).resolve())],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mission-sandbox",
        description="Iterate on Hindsight observation missions with a fast feedback loop.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Create project, ingest documents, run baseline consolidation")
    p_init.add_argument("project", help="Project directory to create")
    p_init.add_argument("--documents", required=True, help="Path to documents dir or file")
    p_init.add_argument("--bank-id", required=True, help="Sandbox bank ID")
    p_init.add_argument("--api-url", default="http://localhost:8888", help="Hindsight API URL")
    p_init.set_defaults(func=cmd_init)

    # agent-label
    p_label = sub.add_parser("agent-label", help="LLM-based observation labeling")
    p_label.add_argument("project", help="Project directory")
    p_label.add_argument("--instructions", required=True, help="What observations you want")
    p_label.add_argument("--model", default="anthropic/claude-sonnet-4-20250514", help="LiteLLM model")
    p_label.set_defaults(func=cmd_agent_label)

    # optimize
    p_opt = sub.add_parser("optimize", help="Generate improved mission from labeled observations")
    p_opt.add_argument("project", help="Project directory")
    p_opt.add_argument("--model", default="anthropic/claude-sonnet-4-20250514", help="LiteLLM model")
    p_opt.set_defaults(func=cmd_optimize)

    # run
    p_run = sub.add_parser("run", help="Apply mission, re-consolidate, update observations for next label round")
    p_run.add_argument("project", help="Project directory")
    p_run.set_defaults(func=cmd_run)

    # promote
    p_prom = sub.add_parser("promote", help="Push optimized mission to a target bank")
    p_prom.add_argument("project", help="Project directory")
    p_prom.add_argument("--target-bank", required=True, help="Production bank ID")
    p_prom.add_argument("--backfill", action="store_true", help="Trigger consolidation backfill")
    p_prom.set_defaults(func=cmd_promote)

    # ui
    p_ui = sub.add_parser("ui", help="Open the project browser UI")
    p_ui.add_argument("project", help="Project directory")
    p_ui.set_defaults(func=cmd_ui)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
