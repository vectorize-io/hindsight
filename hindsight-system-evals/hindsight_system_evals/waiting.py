"""Wait for the asynchronous half of the system to finish.

A retain returns as soon as the facts are stored, but the server is not done: the
worker still has observation extraction and possibly consolidation to run. Those
are not noise to be suppressed — they are where a large share of this project's
composition bugs live — so tests let them run and wait for them here.

The wait is expressed through the public operations API, like everything else in
this suite. It settles when the bank has nothing pending or processing, and it
fails loudly on a failed operation rather than letting a test assert against a
half-built bank and call it a pass.
"""

from __future__ import annotations

import asyncio
import os
import time

from hindsight_client import Hindsight

# Far longer than the system tests use, and for a concrete reason: there a stub
# answers instantly, so 90s is generous. Here a settle waits on real fact
# extraction AND consolidation over the whole corpus, each a real model call. The
# first run of this suite failed at 90s with `consolidation(processing)` still in
# flight — the eval was fine, the inherited timeout was not.
SETTLE_TIMEOUT_SECONDS = float(os.getenv("HINDSIGHT_EVAL_SETTLE_TIMEOUT", "900"))
# Polling five times a second against a real server for fifteen minutes is a lot
# of pointless requests; a settle here takes minutes, not milliseconds.
POLL_INTERVAL_SECONDS = 1.0

# The bank must look idle for two consecutive polls before we believe it. One
# quiet poll proves nothing: a worker that has just claimed the next task, or an
# operation enqueued a moment after the previous one completed, both read as
# "nothing in flight" for an instant.
_CONSECUTIVE_QUIET_POLLS = 2

_BUSY_STATUSES = ("pending", "processing")


async def wait_until_settled(
    client: Hindsight,
    bank_id: str,
    *,
    timeout: float = SETTLE_TIMEOUT_SECONDS,
) -> None:
    """Block until the bank's background work has finished.

    Raises on a failed operation, and on timeout reports what was still in flight
    — a bare "timed out" would send the reader to the server log for something the
    API could have told them.
    """
    deadline = time.monotonic() + timeout
    quiet_polls = 0
    in_flight: list[str] = []

    while time.monotonic() < deadline:
        failed = await client.operations.list_operations(bank_id, status="failed", limit=100)
        if failed.operations:
            summary = ", ".join(
                f"{op.task_type}: {op.error_message or 'no error recorded'}" for op in failed.operations
            )
            raise AssertionError(f"bank {bank_id} has failed background operations — {summary}")

        busy = [
            op
            for status in _BUSY_STATUSES
            for op in (await client.operations.list_operations(bank_id, status=status, limit=100)).operations
        ]

        # An operation carrying an error is already lost, even while its status
        # still reads `pending`: that is the worker holding it for a retry. With a
        # deterministic stub the retry gets the same answer, so waiting out the
        # backoff only delays the same failure — report it now, with the message.
        errored = [op for op in busy if op.error_message]
        if errored:
            summary = ", ".join(f"{op.task_type}: {op.error_message}" for op in errored)
            raise AssertionError(f"bank {bank_id} has a background operation that failed — {summary}")

        in_flight = [f"{op.task_type}({op.status})" for op in busy]

        if in_flight:
            quiet_polls = 0
        else:
            quiet_polls += 1
            if quiet_polls >= _CONSECUTIVE_QUIET_POLLS:
                return

        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    raise AssertionError(f"bank {bank_id} did not settle within {timeout}s; still in flight: {in_flight or 'nothing'}")
