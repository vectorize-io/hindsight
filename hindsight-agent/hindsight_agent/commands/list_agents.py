"""hindsight-agent list — show all configured agents."""

from __future__ import annotations

import json

import click

from ..config import load_config


@click.command("list")
def list_agents() -> None:
    """List all configured agents and their settings."""
    agents = load_config()
    if not agents:
        click.echo("No agents configured. Run 'hindsight-agent setup' to add one.")
        return
    click.echo(json.dumps(
        {aid: cfg.to_dict() for aid, cfg in agents.items()},
        indent=2,
    ))
