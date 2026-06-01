"""promote command: push optimized mission to a target bank."""

from __future__ import annotations

from pathlib import Path

from .helpers import _run, console
from .project import Project, PromoteStep


def run_promote(
    project_path: Path,
    target_bank: str,
    backfill: bool = False,
) -> None:
    from hindsight_client import Hindsight

    proj = Project.load(project_path)

    if not proj.mission:
        console.print("[red]No mission in project. Run optimize first.[/red]")
        return

    client = Hindsight(base_url=proj.api_url)

    console.print(f"[bold]Promoting mission to bank [cyan]{target_bank}[/cyan]...[/bold]")
    client.update_bank_config(bank_id=target_bank, observations_mission=proj.mission)
    console.print("[green]Mission updated.[/green]")

    if backfill:
        console.print("[bold]Triggering backfill consolidation...[/bold]")
        _run(client.banks.clear_observations(target_bank))
        _run(client.banks.trigger_consolidation(target_bank))
        console.print("[green]Backfill consolidation triggered.[/green]")

    # Record history
    step = PromoteStep(
        target_bank=target_bank,
        mission=proj.mission,
        backfill=backfill,
    )
    step_path = proj.add_step(step)
    console.print(f"  History: {step_path.name}")
