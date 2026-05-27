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

# Create a real operation to use in the examples below. Sync retain is fine
# for setup; the docs sections only show the operations API itself.
client.retain(bank_id="my-bank", content="Alice joined Google in 2023")


async def main() -> None:
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

    # Pick a real operation for the per-id examples.
    operation_id = result.operations[0].id if result.operations else None

    if operation_id:
        # [docs:operations-get]
        status = await client.operations.get_operation_status("my-bank", operation_id)
        print(status.status, status.error_message)

        # Include the submission payload (can be large for retain batches).
        detailed = await client.operations.get_operation_status(
            "my-bank", operation_id, include_payload=True
        )
        # [/docs:operations-get]

    # [docs:operations-cancel]
    # Cancel a pending operation before a worker claims it.
    # Returns 409 if the operation is already processing/completed/failed.
    try:
        await client.operations.cancel_operation("my-bank", operation_id)
    except Exception:
        # Already in a non-pending state — fine for this example.
        pass
    # [/docs:operations-cancel]

    # [docs:operations-retry]
    # Re-queue a failed (or cancelled) operation.
    # Returns 409 if the operation isn't in failed/cancelled state.
    try:
        await client.operations.retry_operation("my-bank", operation_id)
    except Exception:
        # Operation already in a terminal state we can't retry — fine here.
        pass
    # [/docs:operations-retry]

    # [docs:operations-async-retain]
    # Submit a large batch asynchronously — the call returns immediately with
    # an operation_id you can poll.
    submission = client.retain_batch(
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
