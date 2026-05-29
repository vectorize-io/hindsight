#!/usr/bin/env python3
"""
Operations API examples for Hindsight (async tracking).
Run: python examples/api/operations.py
"""
import asyncio
import os

from hindsight_client import Hindsight

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
client = Hindsight(base_url=HINDSIGHT_URL)


async def main() -> None:
    # Seed a real pending operation so the per-id endpoints below have something
    # to act on. Staying on the async path (aretain_batch + retain_async=True)
    # keeps everything on one event loop — mixing the sync convenience methods
    # with the async operations API breaks the underlying HTTP client.
    seeded = await client.aretain_batch(
        bank_id="my-bank",
        items=[{"content": "Alice joined Google in 2023"}],
        retain_async=True,
    )
    seeded_id = seeded.operation_id

    # [docs:operations-list]
    # List recent operations for a bank (default: 20 most recent).
    result = await client.operations.list_operations("my-bank")
    for op in result.operations:
        print(op.id, op.task_type, op.status)

    # Filter by status and type.
    pending_recompute = await client.operations.list_operations(
        "my-bank", status="pending", type="graph_maintenance"
    )

    # Hide retain_batch parent rows (show only individual child retain jobs).
    flat = await client.operations.list_operations("my-bank", exclude_parents=True)
    # [/docs:operations-list]

    # [docs:operations-get]
    status = await client.operations.get_operation_status("my-bank", seeded_id)
    print(status.status, status.error_message)

    # Include the submission payload (can be large for retain batches).
    detailed = await client.operations.get_operation_status(
        "my-bank", seeded_id, include_payload=True
    )
    # [/docs:operations-get]

    # [docs:operations-cancel]
    # Cancel a pending operation before a worker claims it.
    # Returns 409 if the operation is already processing/completed/failed.
    try:
        await client.operations.cancel_operation("my-bank", seeded_id)
    except Exception:
        # Already in a non-pending state — fine for this example.
        pass
    # [/docs:operations-cancel]

    # [docs:operations-retry]
    # Re-queue a failed (or cancelled) operation.
    # Returns 409 if the operation isn't in failed/cancelled state.
    try:
        await client.operations.retry_operation("my-bank", seeded_id)
    except Exception:
        # Operation already in a terminal state we can't retry — fine here.
        pass
    # [/docs:operations-retry]

    # [docs:operations-async-retain]
    # Submit a batch asynchronously — the call returns immediately with an
    # operation_id you can poll.
    submission = await client.aretain_batch(
        bank_id="my-bank",
        items=[
            {"content": "Alice joined Google in 2023"},
            {"content": "Bob prefers Python over JavaScript"},
        ],
        retain_async=True,
    )
    op_id = submission.operation_id

    while True:
        s = await client.operations.get_operation_status("my-bank", op_id)
        if s.status in ("completed", "failed", "cancelled"):
            print(f"finished: {s.status}")
            break
        await asyncio.sleep(2)
    # [/docs:operations-async-retain]


asyncio.run(main())
