"""Asking for the same refresh twice runs it once.

A mental-model refresh is expensive — it drives the whole reflect loop — and the
requests arrive from several directions at once: a scheduled sweep, a
consolidation that finished, an impatient caller. Without deduplication a busy
bank queues one refresh per trigger and the worker spends its time recomputing
the same answer, which is how a slow-draining bank accumulated ~45 copies of one
model (#3487).

Two guarantees, and they are separate. **At most one refresh waits** per model: a
submit that finds one queued returns that operation instead of adding a copy.
**At most one refresh runs** per model: the operation queued behind a running one
starts when that one finishes, so two refreshes never write the model side by
side and let whichever finished last decide the content.

A refresh that is already running is *not* reused, though — it may have read the
bank before the caller's latest change, so an explicit request always gets an
operation of its own, queued behind it. See
`docs/developer/api/mental-models.mdx`.

Each story below holds the running refresh open (`llm.hold`, see the README) and
asserts what the queue does while the work is genuinely in flight. Submitting and
then reading the queue proves nothing on its own: the worker claims within
milliseconds of the insert, so what a request finds waiting for it would be a
race between HTTP latency and claim latency.
"""

from __future__ import annotations

import asyncio

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.server import WORKER_POLL_INTERVAL_MS

pytestmark = pytest.mark.asyncio

ANSWER = "Alice lives in Berlin."

#: Long enough for the worker to have tried, and failed, to claim what is queued.
#:
#: Three poll intervals (the server pins ``HINDSIGHT_API_WORKER_POLL_INTERVAL_MS``),
#: and the worker re-polls immediately after a successful claim, so anything still
#: pending after this waited because it *cannot* be claimed — not because nobody has
#: looked yet. Asserting without the wait would pass on either behaviour: a burst
#: finishes inside a single poll interval, so the queue has not been touched yet
#: whatever the rules are.
_CLAIM_GRACE_SECONDS = 3 * WORKER_POLL_INTERVAL_MS / 1000


@pytest.fixture
async def model(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=ANSWER)

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    created = await client.mental_models.create_mental_model(
        bank_id, {"name": "Housing", "source_query": "Where does Alice live?"}
    )
    await settled(bank_id)
    return created.mental_model_id


async def _refreshes(client, bank_id: str, *, status: str | None = None) -> int:
    """How many refresh operations this bank has, optionally in one status."""
    listed = await client.operations.list_operations(bank_id, status=status, type="refresh_mental_model", limit=100)
    return listed.total


async def _refresh(client, bank_id: str, model: str) -> str:
    return (await client.mental_models.refresh_mental_model(bank_id, model)).operation_id


async def test_a_burst_arriving_mid_run_queues_exactly_one_more(client, llm, bank_id, model, settled):
    """The #3487 shape. Callers keep arriving while a refresh runs, and the queue
    grows by one operation, not one per request — everybody after the first is
    handed the one already waiting.

    The tail arrives all at once (``gather``), which is the harder case: the
    submits then race each other as well as the worker, and what makes them fold
    is the bank row they serialise on inside the submit transaction.
    """
    before = await _refreshes(client, bank_id)

    async with llm.hold("reflect") as held:
        running = await _refresh(client, bank_id, model)
        await held.reached()

        queued = await _refresh(client, bank_id, model)
        assert queued != running, "an explicit refresh must not join a run that may predate its intent"
        await asyncio.sleep(_CLAIM_GRACE_SECONDS)

        later = set(await asyncio.gather(*(_refresh(client, bank_id, model) for _ in range(4))))
        assert later == {queued}, f"4 simultaneous requests queued {len(later)} operations instead of folding into one"

    await settled(bank_id)
    assert await _refreshes(client, bank_id) - before == 2, "one run, one queued behind it"


async def test_the_queued_refresh_waits_for_the_running_one(client, llm, bank_id, model, settled):
    """It is queued *behind*, not beside. Both refreshes write the same model, so
    running them together would leave the content whichever one finished last —
    and in delta mode they would move each other's watermark."""
    async with llm.hold("reflect") as held:
        await _refresh(client, bank_id, model)
        await held.reached()

        queued = await _refresh(client, bank_id, model)
        await asyncio.sleep(_CLAIM_GRACE_SECONDS)

        assert await _refreshes(client, bank_id, status="processing") == 1
        assert await _refreshes(client, bank_id, status="pending") == 1

    await settled(bank_id)

    status = await client.operations.get_operation_status(bank_id, queued)
    assert status.status == "completed", "the queued refresh must still run once the way is clear"


async def test_the_model_still_ends_up_refreshed(client, llm, bank_id, model, settled):
    """Deduplication must not swallow the work — collapsing to zero refreshes
    would look identical from the queue's point of view."""
    async with llm.hold("reflect") as held:
        await _refresh(client, bank_id, model)
        await held.reached()
        for _ in range(3):
            await _refresh(client, bank_id, model)

    await settled(bank_id)

    current = await client.mental_models.get_mental_model(bank_id, model, detail="full")
    assert current.content.strip() == ANSWER
    assert current.is_stale is False
