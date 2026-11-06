"""A rollout hands an in-flight operation to the pod that replaces it.

A rollout stops the old pod with SIGTERM and kills it outright once the grace period
is over. The API's embedded worker must hand its in-flight operations back before
that, or they stay claimed by a worker id that never comes back and nothing ever
finishes them (#6021). It used to fail on timing alone: the web server drained its
open requests first, and only after that did the worker start its drain, a fixed
budget counted from that late start — so the hand-back landed past the grace period.

Composition, which is why it lives here: the worker's drain and the HTTP server's
drain each behave on their own. What broke was the order they ran in inside one
process, and only a real process stopped by a real signal shows it.

Scaled down: a 4s drain inside an 8s grace stands in for 20s inside Kubernetes' 30s.
An open slow request keeps the HTTP drain busy for its full 5s, as in the report.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from pathlib import Path

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests import start_hindsight_server
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

#: Its own schema, so no other story's worker claims this one's operation.
SCHEMA = "systest_rollout"
DRAIN_SECONDS = 4
GRACE_SECONDS = 8.0
COMPLETION_TIMEOUT_SECONDS = 60.0


def _pod_env(worker_id: str) -> dict[str, str]:
    return {
        "HINDSIGHT_API_DATABASE_SCHEMA": SCHEMA,
        # A hostname-derived id in a real pod: the new pod never has the old one's.
        "HINDSIGHT_API_WORKER_ID": worker_id,
        "HINDSIGHT_API_WORKER_SHUTDOWN_DRAIN_SECONDS": str(DRAIN_SECONDS),
        # The interrupted run counts against the retry budget; the suite's 0 would
        # fail the handed-back operation instead of letting the new pod run it.
        "HINDSIGHT_API_WORKER_MAX_RETRIES": "1",
    }


async def test_a_rollout_hands_in_flight_work_to_the_new_pod(stub_server, llm, tmp_path: Path):
    llm.on_step("extract_facts").returns(extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice"])))
    llm.on_step("consolidate").returns(consolidation())
    bank_id = f"systest-{uuid.uuid4().hex[:12]}"

    old_pod = start_hindsight_server(
        stub_url=stub_server.url, log_path=tmp_path / "old-pod.log", extra_env=_pod_env("old-pod")
    )
    old_client = Hindsight(base_url=old_pod.url)
    slow_client = Hindsight(base_url=old_pod.url, timeout=120)
    try:
        async with llm.hold("extract_facts") as held:
            receipt = await old_client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", retain_async=True)
            await held.reached()  # the worker is mid-operation
            slow_request = asyncio.ensure_future(slow_client.aretain(bank_id=bank_id, content="Bob plays chess."))
            await asyncio.sleep(1.0)  # let the slow request reach the server

            started = time.monotonic()
            exited_in_time = await asyncio.to_thread(old_pod.terminate_like_a_platform, GRACE_SECONDS)
            stopped_after = time.monotonic() - started
        with contextlib.suppress(Exception):  # it dies with the old pod; only the operation matters
            await slow_request
    finally:
        await old_client.aclose()
        await slow_client.aclose()
        old_pod.stop()

    new_pod = start_hindsight_server(
        stub_url=stub_server.url, log_path=tmp_path / "new-pod.log", extra_env=_pod_env("new-pod")
    )
    client = Hindsight(base_url=new_pod.url)
    try:
        deadline = time.monotonic() + COMPLETION_TIMEOUT_SECONDS
        status = await client.operations.get_operation_status(bank_id, receipt.operation_id)
        while status.status in ("pending", "processing") and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            status = await client.operations.get_operation_status(bank_id, receipt.operation_id)

        assert status.status == "completed", (
            f"the old pod's operation is {status.status!r} on the new pod (old pod "
            f"{'exited' if exited_in_time else 'was killed'} after {stopped_after:.1f}s):\n{old_pod.logs()[-3000:]}"
        )
        assert exited_in_time, f"the old pod outlived its {GRACE_SECONDS}s grace period ({stopped_after:.1f}s)"
    finally:
        with contextlib.suppress(Exception):
            await client.banks.delete_bank(bank_id)
        await client.aclose()
        new_pod.stop()
