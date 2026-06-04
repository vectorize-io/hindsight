"""Command-line entry point for observation deduplication.

Example:
    uv run find-duplicate-observations --bank-id hermes --threshold 0.92
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from hindsight_api.config import DEFAULT_EMBEDDINGS_LOCAL_MODEL
from rich.console import Console

from .client import ObservationClient
from .dedup import find_duplicate_clusters
from .report import DedupReport, render_report

DEFAULT_API_URL = os.environ.get("HINDSIGHT_API_URL", "http://localhost:8888")

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="find-duplicate-observations",
        description="Find near-duplicate observations in a Hindsight bank via cosine similarity.",
    )
    parser.add_argument("--bank-id", required=True, help="Bank to scan (e.g. 'hermes').")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Base URL of the Hindsight API (default: {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("HINDSIGHT_API_KEY"),
        help="Optional API key (sent as a Bearer token).",
    )
    parser.add_argument("--tenant", default="default", help="Tenant segment in the API path (default: 'default').")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.92,
        help="Minimum cosine similarity for two observations to be linked (default: 0.92).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum cluster size to report (default: 2).",
    )
    parser.add_argument("--page-size", type=int, default=200, help="API pagination page size (default: 200).")
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDINGS_LOCAL_MODEL,
        help=f"Local sentence-transformers model used to re-embed text (default: {DEFAULT_EMBEDDINGS_LOCAL_MODEL}).",
    )
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU embedding even if a GPU is available.")
    parser.add_argument("--max-text", type=int, default=160, help="Truncate displayed text to this length.")
    parser.add_argument("--json-out", type=Path, default=None, help="Write the full report to this JSON file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not 0.0 < args.threshold <= 1.0:
        console.print("[red]--threshold must be in (0, 1].[/red]")
        return 2

    try:
        with ObservationClient(args.api_url, api_key=args.api_key, tenant=args.tenant) as client:
            try:
                client.check_health()
            except httpx.HTTPError as exc:
                console.print(f"[red]Cannot reach Hindsight API at {args.api_url}: {exc}[/red]")
                return 1

            console.print(f"Fetching observations from bank [cyan]{args.bank_id}[/cyan]…")
            with console.status("Listing observations…") as status:

                def _progress(fetched: int, total: int) -> None:
                    status.update(f"Listing observations… {fetched}/{total}")

                observations = client.fetch_observations(args.bank_id, page_size=args.page_size, progress=_progress)
    except httpx.HTTPStatusError as exc:
        console.print(f"[red]API returned {exc.response.status_code}: {exc.response.text}[/red]")
        return 1
    except httpx.HTTPError as exc:
        console.print(f"[red]Request failed: {exc}[/red]")
        return 1

    console.print(f"Fetched [cyan]{len(observations)}[/cyan] observations.")
    if len(observations) < 2:
        console.print("[green]Nothing to compare — fewer than 2 observations.[/green]")
        return 0

    with console.status(f"Embedding with {args.embedding_model} and scanning for duplicates…"):
        clusters = find_duplicate_clusters(
            observations,
            threshold=args.threshold,
            min_cluster_size=args.min_cluster_size,
            model_name=args.embedding_model,
            force_cpu=args.force_cpu,
        )

    report = DedupReport(
        bank_id=args.bank_id,
        total_observations=len(observations),
        threshold=args.threshold,
        clusters=clusters,
    )
    render_report(report, console, max_text=args.max_text)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"\nWrote JSON report to [cyan]{args.json_out}[/cyan].")

    return 0


if __name__ == "__main__":
    sys.exit(main())
