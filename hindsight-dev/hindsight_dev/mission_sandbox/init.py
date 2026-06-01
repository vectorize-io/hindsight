"""init command: create project, ingest documents, consolidate, export observations."""

from __future__ import annotations

from pathlib import Path

from .helpers import _run, console, fetch_observations, wait_for_consolidation
from .project import InitStep, Project


def _collect_documents(path: Path) -> list[Path]:
    """Collect .txt and .md files from a path (file or directory)."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(f for f in path.rglob("*") if f.is_file() and f.suffix in (".txt", ".md"))
    raise FileNotFoundError(f"Path not found: {path}")


def run_init(
    project_path: Path,
    documents_path: Path,
    bank_id: str,
    api_url: str,
) -> None:
    from hindsight_client import Hindsight

    # 1. Create project
    proj = Project.create(project_path, bank_id=bank_id, api_url=api_url)
    console.print(f"[bold]Created project at [cyan]{project_path}[/cyan][/bold]")

    client = Hindsight(base_url=api_url)

    # 2. Create bank (no mission for baseline)
    console.print(f"[bold]Creating bank [cyan]{bank_id}[/cyan]...[/bold]")
    client.create_bank(bank_id=bank_id, enable_observations=True)

    # 3. Ingest documents
    docs = _collect_documents(documents_path)
    console.print(f"[bold]Ingesting {len(docs)} document(s)...[/bold]")
    doc_names = []
    for doc in docs:
        content = doc.read_text()
        console.print(f"  Retaining: {doc.name} ({len(content)} chars)")
        client.retain(bank_id=bank_id, content=content, document_id=doc.stem)
        doc_names.append(doc.name)

    # 4. Trigger consolidation and wait
    console.print("[bold]Triggering consolidation...[/bold]")
    _run(client.banks.trigger_consolidation(bank_id))
    wait_for_consolidation(client, bank_id)
    console.print("[green]Consolidation complete.[/green]")

    # 5. Export observations
    console.print("[bold]Exporting observations...[/bold]")
    observations = fetch_observations(client, bank_id)

    # 6. Save to project
    proj.observations = observations
    step = InitStep(documents=doc_names, observations=observations)
    step_path = proj.add_step(step)

    console.print(f"[green bold]Done![/green bold] {len(observations)} observations")
    console.print(f"  Project: {project_path}")
    console.print(f"  History: {step_path.name}")
