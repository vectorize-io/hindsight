"""Report model and rendering for observation-dedup results."""

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from .models import DuplicateCluster


@dataclass(frozen=True)
class DedupReport:
    """Outcome of a dedup run over a single bank."""

    bank_id: str
    total_observations: int
    threshold: float
    clusters: list[DuplicateCluster]

    @property
    def duplicate_clusters(self) -> int:
        return len(self.clusters)

    @property
    def redundant_observations(self) -> int:
        """Total observations that are redundant copies within some cluster."""
        return sum(c.redundant_count for c in self.clusters)

    def to_dict(self) -> dict:
        """Serialisable representation for ``--json-out``."""
        return {
            "bank_id": self.bank_id,
            "total_observations": self.total_observations,
            "threshold": self.threshold,
            "duplicate_clusters": self.duplicate_clusters,
            "redundant_observations": self.redundant_observations,
            "clusters": [
                {
                    "size": c.size,
                    "max_similarity": round(c.max_similarity, 4),
                    "min_similarity": round(c.min_similarity, 4),
                    "observations": [
                        {
                            "id": obs.id,
                            "text": obs.text,
                            "entities": obs.entities,
                            "tags": list(obs.tags),
                            "mentioned_at": obs.mentioned_at,
                        }
                        for obs in c.observations
                    ],
                }
                for c in self.clusters
            ],
        }


def _truncate(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def render_report(report: DedupReport, console: Console, *, max_text: int = 160) -> None:
    """Print a human-readable summary of the dedup run."""
    console.print()
    console.rule(f"[bold]Observation duplicates — bank '{report.bank_id}'[/bold]")
    summary = Table.grid(padding=(0, 2))
    summary.add_row("Total observations:", f"[cyan]{report.total_observations}[/cyan]")
    summary.add_row("Similarity threshold:", f"[cyan]{report.threshold:.2f}[/cyan]")
    summary.add_row("Duplicate clusters:", f"[yellow]{report.duplicate_clusters}[/yellow]")
    summary.add_row("Redundant observations:", f"[red]{report.redundant_observations}[/red]")
    console.print(summary)

    if not report.clusters:
        console.print("\n[green]No near-duplicate observations found.[/green]")
        return

    for rank, cluster in enumerate(report.clusters, start=1):
        sim_range = (
            f"{cluster.min_similarity:.3f}"
            if cluster.min_similarity == cluster.max_similarity
            else f"{cluster.min_similarity:.3f}–{cluster.max_similarity:.3f}"
        )
        console.print(
            f"\n[bold]Cluster {rank}[/bold] "
            f"([cyan]{cluster.size}[/cyan] observations, similarity [magenta]{sim_range}[/magenta])"
        )
        table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
        table.add_column("id", style="dim", no_wrap=True)
        table.add_column("text")
        for obs in cluster.observations:
            table.add_row(obs.id, _truncate(obs.text, max_text))
        console.print(table)
