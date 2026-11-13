#!/usr/bin/env python3
"""
Observation tree API examples for Hindsight — the bank's observations as markdown files.
Run: python examples/api/observation-tree.py
"""
import asyncio
import os
import tempfile

from hindsight_client import Hindsight

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
BANK_ID = "observation-tree-demo-bank"


async def wait_for_idle(client, bank_id, *, attempts=120, interval=0.5):
    """Block until the bank has no pending/processing operations (consolidation)."""
    await asyncio.sleep(interval)
    for _ in range(attempts):
        busy = False
        for state in ("pending", "processing"):
            res = await client.operations.list_operations(bank_id=bank_id, status=state, limit=1)
            if res.operations:
                busy = True
                break
        if not busy:
            return
        await asyncio.sleep(interval)


async def main():
    # =========================================================================
    # Setup (not shown in docs)
    # =========================================================================
    client = Hindsight(base_url=HINDSIGHT_URL)
    await client.acreate_bank(bank_id=BANK_ID, name="Observation Tree Demo")
    await client.aretain(bank_id=BANK_ID, content="The API runs on Kubernetes behind an nginx ingress.", tags=["team:platform"])
    await client.aretain(bank_id=BANK_ID, content="Redis keys expire after an hour.", tags=["team:platform"])
    await client.aretain(bank_id=BANK_ID, content="The GPU nodes serve Gemma with vLLM.", tags=["team:ml"])
    await wait_for_idle(client, BANK_ID)
    out_dir = tempfile.mkdtemp()

    # =========================================================================
    # Doc Examples
    # =========================================================================

    # [docs:get-tree]
    # The whole bank: folders, topic files and each folder's index.md.
    tree = await client.memory.get_observation_tree(bank_id=BANK_ID)
    print(f"{tree.observation_count} observations")

    def walk(folder):
        for f in folder.files:
            print(f"/{f.path} ({f.observation_count})")
        for sub in folder.folders:
            walk(sub)

    walk(tree)

    # Only one scope: the same tag filters as recall.
    platform = await client.memory.get_observation_tree(bank_id=BANK_ID, tags=["team:platform"], tags_match="any_strict")
    walk(platform)
    # [/docs:get-tree]

    # [docs:write-bundle]
    # Write the bundle to disk: each file's markdown at its path.
    def write(folder):
        for f in folder.files:
            path = os.path.join(out_dir, f.path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f.content)
        for sub in folder.folders:
            write(sub)

    write(tree)
    # [/docs:write-bundle]
    assert os.path.exists(os.path.join(out_dir, "index.md"))

    # =========================================================================
    # Cleanup (not shown in docs)
    # =========================================================================
    await client.adelete_bank(bank_id=BANK_ID)
    print("observation-tree.py: All examples passed")


asyncio.run(main())
