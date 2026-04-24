"""hindsight-agent CLI — agent scaffolding and runtime for Hindsight memory."""

from __future__ import annotations

import click

from .commands.documents import documents
from .commands.list_agents import list_agents
from .commands.pages import pages
from .commands.recall import recall
from .commands.retain import retain
from .commands.setup import setup


@click.group()
def main() -> None:
    """Agent scaffolding and runtime CLI for Hindsight memory."""


main.add_command(setup)
main.add_command(list_agents)
main.add_command(retain)
main.add_command(pages)
main.add_command(recall)
main.add_command(documents)


if __name__ == "__main__":
    main()
