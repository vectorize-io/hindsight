"""Submit-time dedup for graph maintenance must also treat a *running* job as
covering the bank.

`run_graph_maintenance_job` states that it relies on submit-time dedup to keep
at most one job per bank running, and skips SKIP LOCKED on that basis. That was
not true: dedup matched only `status = 'pending'`, and a pending row is claimed
within milliseconds under load. Each subsequent trigger therefore found no
pending row and inserted another job, so a bank taking sustained writes
accumulated one maintenance job per write — hundreds of concurrent jobs
against a single bank, all doing the same bank-wide sweeps.

Consolidation must keep the old behaviour: its job fixes a watermark when it
starts, so content added afterwards genuinely needs a fresh run. Graph
maintenance drains its own queue to empty instead, which is what makes matching
`processing` safe there and not there.

These run against the real Postgres test DB — the dedup is a SQL predicate over
`async_operations`, so a mock would prove nothing about it.
"""

from __future__ import annotations

import uuid

import pytest

# These seed `graph_maintenance_queue` / `entity_maintenance_queue` with raw SQL and assert the
# dedup behaviour of the rows they put there. A store that owns its memories has no such rows --
# the entity postings travel inside the memory, so `submit_async_graph_maintenance` short-circuits
# with `no_work` before any queue is consulted (see the store_owned_for branch in the engine). The
# dedup they pin is Postgres-queue behaviour, not engine behaviour.

from hindsight_api.engine.memories.base import RelinkPassResult
from hindsight_api.engine.memory_engine import MemoryEngine

pytestmark = [
    pytest.mark.memory_backend_incompatible,
    pytest.mark.xdist_group("graph_maintenance_submit_dedup_tests"),
]


async def _progress_relink():
    """Pretend Pass 1 drained rows (queue reported exhausted) while a row is left
    in the queue, standing in for work enqueued after the final claim — the gap
    the hand-off exists to close."""
    return RelinkPassResult(units_processed=3, links_added=0, queue_exhausted=True)


@pytest.fixture
def no_inline_execution(memory):
    """Stop SyncTaskBackend running submitted ops inline, so a submitted job
    stays in whatever status the test puts it in."""

    async def _noop(_payload):
        return None

    original = memory._task_backend.submit_task
    memory._task_backend.submit_task = _noop
    yield
    memory._task_backend.submit_task = original


async def _make_bank(memory: MemoryEngine, request_context) -> str:
    """async_operations has an FK to banks, so the row has to exist first."""
    bank_id = f"gm-dedup-{uuid.uuid4().hex[:8]}"
    await memory._ensure_bank_exists(bank_id, request_context)
    return bank_id


async def _queue_one(memory: MemoryEngine, bank_id: str) -> None:
    """Put a row in graph_maintenance_queue so the submit short-circuit
    ('no work → don't enqueue') doesn't fire."""
    backend = await memory._get_backend()
    from hindsight_api.engine.memory_engine import acquire_with_retry

    async with acquire_with_retry(backend) as conn:
        await backend.ops.enqueue_graph_maintenance(conn, "graph_maintenance_queue", bank_id, [uuid.uuid4()])


async def _set_status(memory: MemoryEngine, operation_id: str, status: str) -> None:
    backend = await memory._get_backend()
    from hindsight_api.engine.memory_engine import acquire_with_retry

    async with acquire_with_retry(backend) as conn:
        await conn.execute(
            "UPDATE async_operations SET status = $2 WHERE operation_id = $1::uuid",
            operation_id,
            status,
        )


async def _count_ops(memory: MemoryEngine, bank_id: str, operation_type: str) -> int:
    backend = await memory._get_backend()
    from hindsight_api.engine.memory_engine import acquire_with_retry

    async with acquire_with_retry(backend) as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM async_operations WHERE bank_id = $1 AND operation_type = $2",
            bank_id,
            operation_type,
        )


@pytest.mark.asyncio
async def test_pending_job_dedupes_a_second_submit(memory, request_context, no_inline_execution):
    """Pre-existing behaviour, pinned so the widened predicate doesn't lose it."""
    bank_id = await _make_bank(memory, request_context)
    await _queue_one(memory, bank_id)

    first = await memory.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)
    second = await memory.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)

    assert second.get("deduplicated") is True
    assert second["operation_id"] == first["operation_id"]
    assert await _count_ops(memory, bank_id, "graph_maintenance") == 1


@pytest.mark.asyncio
async def test_processing_job_dedupes_a_new_submit(memory, request_context, no_inline_execution):
    """The fix. A claimed (processing) job must suppress further submits.

    Without this the queue grows by one job per triggering write, because the
    pending row the previous submit created is already gone from 'pending'.
    """
    bank_id = await _make_bank(memory, request_context)
    await _queue_one(memory, bank_id)

    first = await memory.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)
    # A worker claims it.
    await _set_status(memory, first["operation_id"], "processing")

    second = await memory.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)

    assert second.get("deduplicated") is True
    assert second["operation_id"] == first["operation_id"]
    assert await _count_ops(memory, bank_id, "graph_maintenance") == 1


@pytest.mark.asyncio
async def test_sustained_triggers_do_not_stack_jobs(memory, request_context, no_inline_execution):
    """Reproduces the production shape: repeated triggers against a bank whose
    job is claimed immediately. One job, not one per trigger."""
    bank_id = await _make_bank(memory, request_context)
    await _queue_one(memory, bank_id)

    first = await memory.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)
    await _set_status(memory, first["operation_id"], "processing")

    for _ in range(25):
        await _queue_one(memory, bank_id)
        await memory.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)

    assert await _count_ops(memory, bank_id, "graph_maintenance") == 1


@pytest.mark.asyncio
async def test_completed_job_does_not_dedupe(memory, request_context, no_inline_execution):
    """Once the job finishes, the next trigger must enqueue again — otherwise
    work queued after it would never be drained."""
    bank_id = await _make_bank(memory, request_context)
    await _queue_one(memory, bank_id)

    first = await memory.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)
    await _set_status(memory, first["operation_id"], "completed")

    second = await memory.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)

    assert second.get("deduplicated") is not True
    assert second["operation_id"] != first["operation_id"]
    assert await _count_ops(memory, bank_id, "graph_maintenance") == 2


@pytest.mark.asyncio
async def test_consolidation_still_ignores_processing(memory, request_context, no_inline_execution):
    """The widened predicate is opt-in and must not reach consolidation.

    A running consolidation holds a watermark from when it started, so content
    added afterwards needs its own run. Deduping against 'processing' here
    would silently drop that work.
    """
    bank_id = await _make_bank(memory, request_context)

    first = await memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
    await _set_status(memory, first["operation_id"], "processing")

    second = await memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)

    assert second.get("deduplicated") is not True
    assert second["operation_id"] != first["operation_id"]
    assert await _count_ops(memory, bank_id, "consolidation") == 2


@pytest.mark.asyncio
async def test_job_hands_off_when_work_lands_during_the_run(memory, request_context, monkeypatch):
    """Deduping against 'processing' opens a gap: a submit made while the job
    runs is suppressed, so whatever it queued must be picked up by the job
    itself or it strands until some unrelated future write.

    Simulated by making Pass 1 a no-op, which leaves the queue row in place
    exactly as if it had been enqueued after the final claim.
    """
    from hindsight_api.engine.graph_maintenance import run_graph_maintenance_job
    from hindsight_api.engine.memories import get_memories

    bank_id = await _make_bank(memory, request_context)
    await _queue_one(memory, bank_id)

    store = get_memories()
    monkeypatch.setattr(store, "relink_pass", lambda **_kwargs: _progress_relink())

    submitted: list[str] = []

    async def _record(bank_id: str, *, request_context, dedupe_excludes_operation_id=None):
        # Record only. Delegating to the real submit would let the sync task
        # backend run the successor inline, and the assertion would then be
        # about the whole chain rather than this run's hand-off decision.
        submitted.append(bank_id)
        return {"operation_id": None, "no_work": True}

    monkeypatch.setattr(memory, "submit_async_graph_maintenance", _record)

    await run_graph_maintenance_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

    assert submitted == [bank_id], "job must hand off the work it could not drain"


@pytest.mark.asyncio
async def test_job_does_not_hand_off_when_queue_is_empty(memory, request_context, monkeypatch):
    """The common case must not submit anything — otherwise every maintenance
    run would enqueue its own successor forever."""
    from hindsight_api.engine.graph_maintenance import run_graph_maintenance_job

    bank_id = await _make_bank(memory, request_context)  # nothing queued

    submitted: list[str] = []

    async def _record(bank_id: str, *, request_context, dedupe_excludes_operation_id=None):
        submitted.append(bank_id)
        return {"operation_id": None, "no_work": True}

    monkeypatch.setattr(memory, "submit_async_graph_maintenance", _record)

    await run_graph_maintenance_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

    assert submitted == [], "no queued work means no successor"


@pytest.mark.asyncio
async def test_job_does_not_chain_forever_when_it_drains_nothing(memory, request_context, monkeypatch):
    """A run that drains nothing and still sees queued work must NOT hand off.

    The successor would hit the same state and hand off again — an endless
    per-bank chain. Caught by an earlier version of this test seeing two
    submissions instead of one.
    """
    from hindsight_api.engine.graph_maintenance import run_graph_maintenance_job
    from hindsight_api.engine.memories import get_memories

    bank_id = await _make_bank(memory, request_context)
    await _queue_one(memory, bank_id)

    async def _no_progress():
        return RelinkPassResult(units_processed=0, links_added=0, queue_exhausted=True)

    monkeypatch.setattr(get_memories(), "relink_pass", lambda **_kwargs: _no_progress())

    submitted: list[str] = []

    async def _record(bank_id: str, *, request_context, dedupe_excludes_operation_id=None):
        submitted.append(bank_id)
        return {"operation_id": None, "no_work": True}

    monkeypatch.setattr(memory, "submit_async_graph_maintenance", _record)

    await run_graph_maintenance_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

    assert submitted == [], "a no-progress run must not chain a successor"


@pytest.mark.asyncio
async def test_handoff_is_not_suppressed_by_the_jobs_own_row(memory, request_context, monkeypatch, no_inline_execution):
    """The hand-off must create a real successor operation, not dedupe against itself.

    Widening dedup to match 'processing' introduced a way for the fix to defeat
    itself: the submitting job is still 'processing' while its body runs (the
    worker only marks it completed afterwards), so its own hand-off matched its
    own row and was silently deduplicated away. The hand-off became dead code
    and the gap it exists to close stayed open.

    The sibling hand-off tests monkeypatch submit_async_graph_maintenance and
    assert the *call* happens, so they cannot see this. This one drives the real
    submit and counts operation rows.
    """
    from hindsight_api.engine.graph_maintenance import run_graph_maintenance_job
    from hindsight_api.engine.memories import get_memories

    bank_id = await _make_bank(memory, request_context)
    await _queue_one(memory, bank_id)

    first = await memory.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)
    await _set_status(memory, first["operation_id"], "processing")

    monkeypatch.setattr(get_memories(), "relink_pass", lambda **_kwargs: _progress_relink())

    before = await _count_ops(memory, bank_id, "graph_maintenance")
    # operation_id is threaded exactly as the worker does it in
    # _handle_graph_maintenance; without it the exclusion cannot apply.
    await run_graph_maintenance_job(
        memory_engine=memory,
        bank_id=bank_id,
        request_context=request_context,
        operation_id=first["operation_id"],
    )
    after = await _count_ops(memory, bank_id, "graph_maintenance")

    assert after == before + 1, (
        f"hand-off produced no successor ({before} -> {after}); the submit matched the job's own 'processing' row"
    )


@pytest.mark.asyncio
async def test_handoff_survives_the_real_worker_path(memory, request_context, monkeypatch):
    """Same guarantee as the test above, but through the worker's own machinery.

    The sibling test calls run_graph_maintenance_job directly and sets
    'processing' with a hand-written UPDATE. That leaves the integration
    unproven: whether the real claim actually marks the row 'processing' before
    the body runs, and whether execute_task threads operation_id down to the
    hand-off. Both are load-bearing — if either is false the self-exclusion
    never applies and the hand-off is silently dead again.

    So this drives the real payload the task backend receives, the real
    mark_operations_processing claim, and the real execute_task router.
    Observed: status=processing, operations 1 -> 2 with the exclusion and
    1 -> 1 without it.
    """
    from hindsight_api.engine.memories import get_memories
    from hindsight_api.engine.memory_engine import acquire_with_retry

    bank_id = f"gm-e2e-{uuid.uuid4().hex[:8]}"
    await memory._ensure_bank_exists(bank_id, request_context)

    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as c:
        await backend.ops.enqueue_graph_maintenance(c, "graph_maintenance_queue", bank_id, [uuid.uuid4()])

    # Capture the REAL task payload the backend would receive, instead of inventing one.
    captured: list[dict] = []

    async def _capture(payload):
        captured.append(dict(payload))

    memory._task_backend.submit_task = _capture

    first = await memory.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)
    assert captured, "no task payload was submitted"
    task_dict = dict(captured[0])

    # Claim it exactly as a worker does — the real status transition.
    async with acquire_with_retry(backend) as c:
        await backend.ops.mark_operations_processing(
            c, "async_operations", "probe-worker", [uuid.UUID(first["operation_id"])]
        )
        st = await c.fetchval("SELECT status FROM async_operations WHERE operation_id=$1::uuid", first["operation_id"])
    assert st == "processing"

    async def _progress():
        return RelinkPassResult(units_processed=3, links_added=0, queue_exhausted=True)

    monkeypatch.setattr(get_memories(), "relink_pass", lambda **_k: _progress())

    async def _count():
        async with acquire_with_retry(backend) as c:
            return await c.fetchval(
                "SELECT count(*) FROM async_operations WHERE bank_id=$1 AND operation_type='graph_maintenance'",
                bank_id,
            )

    before = await _count()
    await memory.execute_task(task_dict)  # <-- the real router, not the job body
    after = await _count()

    assert after == before + 1, f"hand-off produced no successor through the real worker path ({before} -> {after})"
