"""optimize command: generate improved mission from labeled observations."""

from __future__ import annotations

from pathlib import Path

from rich.panel import Panel

from .helpers import console
from .project import OptimizeStep, Project

_OPTIMIZE_SYSTEM_PROMPT = """\
You are an expert at writing observation missions for a memory system.

An "observation mission" is a prompt that controls how raw facts get consolidated
into observations. Observations are synthesized summaries derived from multiple raw facts.

The user has labeled a set of observations as "good" (valuable) or "bad" (noise/unwanted).
Each observation has a reason explaining why it was labeled that way.

Your job: write an improved observation mission prompt that would produce more of the
good observations and fewer of the bad ones.

Rules:
- The mission should be concise and actionable (a few sentences to a short paragraph)
- Focus on what TO track and what NOT to track
- Be specific about the domain based on the patterns you see
- If there's an existing mission, improve it; otherwise write one from scratch

Respond with ONLY the mission text, no explanation or wrapper.
"""


def run_optimize(
    project_path: Path,
    model: str,
) -> None:
    import litellm

    proj = Project.load(project_path)

    labeled = [o for o in proj.observations if o.label is not None]
    if not labeled:
        console.print("[red]No labeled observations found. Run agent-label first.[/red]")
        return

    good = [o for o in labeled if o.label == "good"]
    bad = [o for o in labeled if o.label == "bad"]
    console.print(f"[bold]Optimizing mission from {len(good)} good + {len(bad)} bad examples...[/bold]")

    # Build the examples section
    examples_parts: list[str] = []
    for o in good:
        examples_parts.append(f"GOOD: {o.text}\n  Reason: {o.reason}")
    for o in bad:
        examples_parts.append(f"BAD: {o.text}\n  Reason: {o.reason}")
    examples_str = "\n\n".join(examples_parts)

    current_mission = proj.mission or "(no mission set — using system defaults)"

    user_msg = f"""\
## Current mission
{current_mission}

## Labeled observations
{examples_str}

Write an improved observation mission.
"""

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": _OPTIMIZE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
    )
    new_mission = response.choices[0].message.content.strip()

    console.print("\n[bold]Proposed mission:[/bold]")
    console.print(Panel(new_mission, title="New Mission", border_style="green"))

    if proj.mission:
        console.print("\n[bold]Previous mission:[/bold]")
        console.print(Panel(proj.mission, title="Old Mission", border_style="dim"))

    # Record history
    step = OptimizeStep(
        model=model,
        previous_mission=proj.mission,
        proposed_mission=new_mission,
        good_count=len(good),
        bad_count=len(bad),
    )
    proj.mission = new_mission
    step_path = proj.add_step(step)

    console.print("\n[green bold]Mission saved.[/green bold]")
    console.print(f"  History: {step_path.name}")
