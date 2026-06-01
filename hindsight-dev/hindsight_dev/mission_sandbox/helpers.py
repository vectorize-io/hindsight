"""Shared helpers for mission-sandbox commands."""

from __future__ import annotations

import asyncio
import time

from rich.console import Console

from .project import ObservationSample

console = Console()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def wait_for_consolidation(client, bank_id: str, timeout: int = 600) -> None:
    """Poll bank stats until no pending consolidation remains."""
    start = time.time()
    while time.time() - start < timeout:
        stats = _run(client.banks.get_agent_stats(bank_id))
        pending = getattr(stats, "pending_consolidation", 0)
        if pending == 0:
            return
        console.print(f"  Waiting for consolidation... ({pending} pending)")
        time.sleep(3)
    raise TimeoutError(f"Consolidation did not complete within {timeout}s")


def fetch_observations(client, bank_id: str) -> list[ObservationSample]:
    """Fetch all observations from a bank."""
    observations: list[ObservationSample] = []
    offset = 0
    while True:
        page = client.list_memories(bank_id=bank_id, type="observation", limit=100, offset=offset)
        items = page.items if hasattr(page, "items") else page.memories if hasattr(page, "memories") else []
        if not items:
            break
        for mem in items:
            mem_id = mem["id"] if isinstance(mem, dict) else mem.id
            mem_text = mem["text"] if isinstance(mem, dict) else mem.text
            observations.append(
                ObservationSample(
                    id=str(mem_id),
                    text=mem_text,
                    source_facts=[],
                )
            )
        offset += len(items)
    return observations
