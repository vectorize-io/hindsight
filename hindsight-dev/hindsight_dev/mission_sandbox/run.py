"""run command: apply mission, re-consolidate, update project with new observations."""

from __future__ import annotations

from pathlib import Path

from .helpers import _run, console, fetch_observations, wait_for_consolidation
from .project import Project, RunStep


def run_run(project_path: Path) -> None:
    from hindsight_client import Hindsight

    proj = Project.load(project_path)

    if not proj.mission:
        console.print("[red]No mission in project. Run optimize first.[/red]")
        return

    client = Hindsight(base_url=proj.api_url)

    # 1. Update observation mission on the sandbox bank
    console.print(f"[bold]Updating mission on bank [cyan]{proj.bank_id}[/cyan]...[/bold]")
    client.update_bank_config(bank_id=proj.bank_id, observations_mission=proj.mission)

    # 2. Clear existing observations and re-consolidate
    console.print("[bold]Clearing existing observations...[/bold]")
    _run(client.banks.clear_observations(proj.bank_id))

    console.print("[bold]Re-running consolidation with new mission...[/bold]")
    _run(client.banks.trigger_consolidation(proj.bank_id))
    wait_for_consolidation(client, proj.bank_id)
    console.print("[green]Consolidation complete.[/green]")

    # 3. Fetch new observations
    new_observations = fetch_observations(client, proj.bank_id)

    # Show summary
    old_count = len(proj.observations)
    old_labeled = sum(1 for o in proj.observations if o.label is not None)
    console.print(f"\n[bold]Previous:[/bold] {old_count} observations ({old_labeled} labeled)")
    console.print(f"[bold]New:[/bold] {len(new_observations)} observations (all unlabeled, ready for agent-label)")

    # 4. Record history and update project
    step = RunStep(
        mission_applied=proj.mission,
        observations=new_observations,
    )
    proj.observations = new_observations
    step_path = proj.add_step(step)

    console.print(f"\n[green bold]Done![/green bold] Dataset updated with {len(new_observations)} observations.")
    console.print(f"  History: {step_path.name}")
    console.print("[dim]Run agent-label next to score these observations.[/dim]")
