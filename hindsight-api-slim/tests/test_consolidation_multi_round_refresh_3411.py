"""Regression reproduction for issue #3411.

Mental models with ``trigger.refresh_after_consolidation = true`` are silently
NOT refreshed when a bank's consolidation backlog spans more than one round.

The bug (fixed in this change): ``_run_consolidation_job`` collected the tags of
the memories it consolidated into a per-call ``consolidated_tags`` set (reset on
every round, never carried across the round-limit re-queue) and triggered mental
model refresh *only on the final round*. A tagged, ``all_strict`` model whose
fact was consolidated in an earlier round was therefore never a refresh candidate
— its tag was absent from the final round's ``consolidated_tags`` — and it stayed
stale forever.

The fix accumulates the affected tags across every round of the chain (threaded
through the re-queue payload as ``pending_refresh_tags``) and flushes the refresh
once, on the final round — so every touched model is refreshed exactly once,
deduplicated but not dropped. When the re-queue is deduped into a concurrently
submitted consolidation, the accumulated tags are folded into the surviving op so
they are not lost (`test_requeue_dedupe_merges_pending_refresh_tags`).

This test drives a real multi-round consolidation chain through the worker
executor (``WorkerTaskBackend`` so re-queues don't recurse) with a small round
limit, and asserts EVERY entity-scoped model is refreshed exactly once as the
backlog drains. Before the fix only the final round's model was refreshed and the
"not dropped" assertion failed (7/8 models left stale); it is the regression guard
for #3411.
"""

import json
import uuid
from unittest.mock import patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation import consolidator as consolidator_module
from hindsight_api.engine.consolidation.consolidator import run_consolidation_job
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.task_backend import WorkerTaskBackend


def _make_config(**overrides):
    raw = _get_raw_config()
    return type(raw)(
        **{
            **{f: getattr(raw, f) for f in raw.__dataclass_fields__},
            **overrides,
        }
    )


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


async def _insert_entity_mm(conn, bank_id: str, tag: str) -> str:
    """A pinned, tagged model on the default all_strict scope with
    refresh_after_consolidation=true and a stale last_refreshed_at."""
    mm_id = f"mm-{uuid.uuid4().hex}"
    trigger = {"refresh_after_consolidation": True}  # default tags_match => all_strict
    await conn.execute(
        """
        INSERT INTO mental_models
          (id, bank_id, subtype, name, source_query, content, tags, trigger, last_refreshed_at)
        VALUES ($1, $2, 'pinned', $3, 'what changed', 'body', $4, $5::jsonb,
                now() - INTERVAL '1 day')
        """,
        mm_id,
        bank_id,
        f"model for {tag}",
        [tag],
        json.dumps(trigger),
    )
    return mm_id


async def _unconsolidated_count(memory, bank_id: str, request_context) -> int:
    # consolidation_state='pending' is the read API's name for exactly this predicate:
    # not consolidated, not failed, and a source fact type.
    page = await memory.list_memory_units(
        bank_id=bank_id,
        consolidation_state="pending",
        limit=1,
        request_context=request_context,
    )
    return page["total"]


async def _pending_consolidations(memory, bank_id: str):
    async with memory._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT operation_id, task_payload FROM async_operations
            WHERE bank_id = $1 AND operation_type = 'consolidation'
              AND status = 'pending' AND task_payload IS NOT NULL
            """,
            bank_id,
        )
    return rows


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_multi_round_consolidation_refreshes_all_entity_models(
    memory: MemoryEngine, request_context, monkeypatch
):
    bank_id = f"mm3411-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    n_entities = 8
    tags = [f"entity:{i}" for i in range(n_entities)]

    # 1. Retain one distinctly-tagged document per entity, with consolidation
    #    disabled during retain so the whole backlog is pending at once.
    fake_no_obs = _make_config(enable_observations=False)
    with patch.object(memory._config_resolver, "resolve_full_config", return_value=fake_no_obs):
        for i, tag in enumerate(tags):
            await memory.retain_batch_async(
                bank_id=bank_id,
                contents=[
                    {
                        "content": f"Entity {i} lives in city number {i} and works job number {i}.",
                        "context": "",
                    }
                ],
                document_tags=[tag],
                request_context=request_context,
            )

    # 2. One refresh_after_consolidation model per entity, each stale.
    mm_by_tag: dict[str, str] = {}
    async with memory._pool.acquire() as conn:
        for tag in tags:
            mm_by_tag[tag] = await _insert_entity_mm(conn, bank_id, tag)

    # Precondition: each entity actually produced a pending tagged fact.
    async with memory._pool.acquire() as conn:
        tagged_pending = await conn.fetch(
            """
            SELECT DISTINCT unnest(tags) AS tag FROM memory_units
            WHERE bank_id = $1 AND consolidated_at IS NULL
              AND fact_type IN ('experience', 'world')
            """,
            bank_id,
        )
    pending_tags = {r["tag"] for r in tagged_pending}
    assert set(tags).issubset(pending_tags), (
        f"precondition failed: not every entity has a pending tagged fact. missing={set(tags) - pending_tags}"
    )

    # 3. Record every mental-model refresh the consolidation chain enqueues.
    refreshed: list[str] = []

    async def _record(*, bank_id, mental_model_id, request_context, skip_if_in_flight=False):
        refreshed.append(mental_model_id)
        return {"operation_id": str(uuid.uuid4())}

    monkeypatch.setattr(memory, "submit_async_refresh_mental_model", _record)

    # 4. Drive the full consolidation chain with a SMALL round limit so the
    #    backlog needs several rounds. WorkerTaskBackend makes the in-job
    #    re-queue land as a pending row instead of recursing (worker semantics).
    original_backend = memory._task_backend
    memory._task_backend = WorkerTaskBackend()
    await memory._task_backend.initialize()

    fake_cfg = _make_config(consolidation_max_memories_per_round=3, enable_observations=True)

    consolidation_runs = 0
    try:
        with patch.object(memory._config_resolver, "resolve_full_config", return_value=fake_cfg):
            # Enqueue the first consolidation, then poll+execute like the worker.
            await memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
            for _ in range(50):  # generous cap; real drain is ~3 rounds
                rows = await _pending_consolidations(memory, bank_id)
                if not rows:
                    break
                for row in rows:
                    payload = row["task_payload"]
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    # Claim the row (pending -> processing) exactly as the worker
                    # poller does. Without this the op stays 'pending' during
                    # execution, so the in-job round-limit re-queue's
                    # submit_async_consolidation dedupes against it and never lands
                    # a successor op — collapsing the chain to a single round.
                    async with memory._pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE async_operations SET status = 'processing' WHERE operation_id = $1",
                            row["operation_id"],
                        )
                    await memory.execute_task(payload)
                    consolidation_runs += 1
    finally:
        memory._task_backend = original_backend

    # 5. The backlog must have drained across MORE THAN ONE round — otherwise the
    #    multi-round bug can't manifest and the test would be vacuous.
    assert consolidation_runs >= 2, (
        f"expected a multi-round drain (round limit 3, backlog > 3), but only {consolidation_runs} consolidation(s) ran"
    )
    remaining = await _unconsolidated_count(memory, bank_id, request_context)
    assert remaining == 0, f"backlog did not fully drain: {remaining} unconsolidated memories remain"

    # 6. KEY ASSERTION — every refresh_after_consolidation model must be refreshed
    #    once the chain drains: not dropped (the #3411 bug) and not duplicated.
    distinct_refreshed = set(refreshed)
    expected = set(mm_by_tag.values())
    missing = expected - distinct_refreshed
    missing_tags = sorted(t for t, mm in mm_by_tag.items() if mm in missing)
    assert not missing, (
        f"issue #3411: {len(missing)}/{len(expected)} refresh_after_consolidation "
        f"mental models were NEVER refreshed after a {consolidation_runs}-round "
        f"consolidation drain. Their facts were consolidated in an earlier round, "
        f"whose refresh was skipped. Stale models (tags): {missing_tags}. "
        f"Refreshed: {sorted(distinct_refreshed)}"
    )

    # Exactly once per model: the affected set is accumulated across rounds and flushed
    # a single time on the final round, so no model is refreshed twice during one drain.
    duplicates = sorted(mm for mm in distinct_refreshed if refreshed.count(mm) > 1)
    assert len(refreshed) == len(expected), (
        f"expected exactly one refresh per model ({len(expected)}), got {len(refreshed)} "
        f"submissions; duplicated models: {duplicates}"
    )

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_requeue_dedupe_merges_pending_refresh_tags(memory: MemoryEngine, request_context):
    """A round-limit re-queue that is deduped into an already-pending unscoped
    consolidation must fold its accumulated ``pending_refresh_tags`` into the
    surviving op, not silently drop them — otherwise the models consolidated by the
    earlier rounds go unrefreshed once the survivor drains (#3411 concurrency guard)."""
    bank_id = f"mm3411dedupe-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    # WorkerTaskBackend so submit_async_consolidation only enqueues rows (no execution).
    original_backend = memory._task_backend
    memory._task_backend = WorkerTaskBackend()
    await memory._task_backend.initialize()
    try:
        # A plain consolidation already pending for the bank (e.g. enqueued by a retain
        # mid-drain) — carries no pending_refresh_tags.
        first = await memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
        # A round-limit re-queue arrives carrying its accumulated tags. dedupe_by_bank
        # folds it into `first` instead of inserting a new row.
        second = await memory.submit_async_consolidation(
            bank_id=bank_id,
            request_context=request_context,
            pending_refresh_tags=["entity:1", "entity:3"],
        )
    finally:
        memory._task_backend = original_backend

    assert second.get("deduplicated") is True
    assert second["operation_id"] == first["operation_id"]

    async with memory._pool.acquire() as conn:
        payload = await conn.fetchval(
            "SELECT task_payload FROM async_operations WHERE operation_id = $1::uuid",
            first["operation_id"],
        )
    payload_dict = json.loads(payload) if isinstance(payload, str) else payload
    assert sorted(payload_dict.get("pending_refresh_tags") or []) == ["entity:1", "entity:3"], (
        f"re-queue's pending_refresh_tags were dropped on dedupe: {payload_dict.get('pending_refresh_tags')}"
    )

    await memory.delete_bank(bank_id, request_context=request_context)


async def _op_pending_refresh_tags(memory, operation_id) -> list[str]:
    async with memory._pool.acquire() as conn:
        payload = await conn.fetchval(
            "SELECT task_payload FROM async_operations WHERE operation_id = $1::uuid",
            str(operation_id),
        )
    payload_dict = json.loads(payload) if isinstance(payload, str) else (payload or {})
    return sorted(payload_dict.get("pending_refresh_tags") or [])


@pytest.mark.asyncio
async def test_crash_mid_round_preserves_committed_batch_refresh_tags(
    memory: MemoryEngine, request_context, monkeypatch
):
    """The durability guarantee for Option-3: a batch persists its refresh tags into the
    op's task_payload atomically with its consolidation. If the worker dies after batch 1
    commits and before the round finishes, the op's retry skips batch 1's now-consolidated
    rows — but its tags survived in task_payload, so the final round still refreshes those
    models (#3411). Without the per-batch persistence, batch 1's tags would be gone and its
    model left stale."""
    bank_id = f"mm3411crash-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    tags = [f"entity:{i}" for i in range(4)]

    fake_no_obs = _make_config(enable_observations=False)
    with patch.object(memory._config_resolver, "resolve_full_config", return_value=fake_no_obs):
        for i, tag in enumerate(tags):
            await memory.retain_batch_async(
                bank_id=bank_id,
                contents=[{"content": f"Entity {i} works job number {i}.", "context": ""}],
                document_tags=[tag],
                request_context=request_context,
            )

    mm_by_tag: dict[str, str] = {}
    async with memory._pool.acquire() as conn:
        for tag in tags:
            mm_by_tag[tag] = await _insert_entity_mm(conn, bank_id, tag)

    # A single-round consolidation (no round limit), forced serial one-memory-at-a-time so
    # batches commit one entity at a time. Inject a crash on the 2nd batch: the 1st entity
    # is fully committed + its tag persisted; the 2nd raises before committing.
    cfg = _make_config(
        enable_observations=True,
        consolidation_max_memories_per_round=0,
        consolidation_llm_parallelism=1,
        consolidation_llm_batch_size=1,
        consolidation_batch_size=100,
    )

    op_id = uuid.uuid4()
    payload = {"type": "consolidation", "operation_id": str(op_id), "bank_id": bank_id}
    async with memory._pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload) "
            "VALUES ($1, $2, 'consolidation', 'processing', $3::jsonb)",
            op_id,
            bank_id,
            json.dumps(payload),
        )

    real_pmb = consolidator_module._process_memory_batch

    async def _crashing_pmb(*args, **kwargs):
        # Crash as soon as a prior batch has committed + persisted its tag (task_payload
        # non-empty). Robust to how many _process_memory_batch calls one entity makes,
        # since a batch's tag lands only at its witness commit, after its calls finish.
        if await _op_pending_refresh_tags(memory, op_id):
            raise RuntimeError("injected worker crash after batch 1 committed")
        return await real_pmb(*args, **kwargs)

    monkeypatch.setattr(consolidator_module, "_process_memory_batch", _crashing_pmb)

    with patch.object(memory._config_resolver, "resolve_full_config", return_value=cfg):
        with pytest.raises(RuntimeError, match="injected worker crash"):
            await run_consolidation_job(
                memory_engine=memory,
                bank_id=bank_id,
                request_context=request_context,
                operation_id=str(op_id),
            )

    # THE FIX: the committed batch's tag is durable in the op's task_payload, so a crash
    # cannot lose it. Without per-batch persistence this would be empty.
    persisted = await _op_pending_refresh_tags(memory, op_id)
    assert len(persisted) >= 1, (
        "a committed batch's refresh tag was not persisted before the crash — it would be "
        "lost on retry and its model left stale"
    )

    # Retry: the worker re-reads task_payload and re-runs. Batch 1 is skipped (already
    # consolidated); the surviving entities are processed. The final round unions the
    # durable pre-crash tag with this run's, so every model is refreshed — including the
    # one consolidated pre-crash that the retry never reprocessed.
    monkeypatch.setattr(consolidator_module, "_process_memory_batch", real_pmb)
    refreshed: list[str] = []

    async def _record(*, bank_id, mental_model_id, request_context, skip_if_in_flight=False):
        refreshed.append(mental_model_id)
        return {"operation_id": str(uuid.uuid4())}

    monkeypatch.setattr(memory, "submit_async_refresh_mental_model", _record)

    with patch.object(memory._config_resolver, "resolve_full_config", return_value=cfg):
        await run_consolidation_job(
            memory_engine=memory,
            bank_id=bank_id,
            request_context=request_context,
            operation_id=str(op_id),
            pending_refresh_tags=persisted,  # what the worker would re-read from task_payload
        )

    distinct = set(refreshed)
    missing = set(mm_by_tag.values()) - distinct
    missing_tags = sorted(t for t, mm in mm_by_tag.items() if mm in missing)
    assert not missing, (
        f"models left stale after a mid-round crash + retry: {missing_tags}. The batch "
        f"consolidated before the crash was never reprocessed, and its tags were dropped."
    )

    await memory.delete_bank(bank_id, request_context=request_context)
