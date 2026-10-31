"""Operation runs: one record per operation, with its wall time and where the time went.

What matters: a run reaches the sink once, failed or not; a run opened inside another is its own,
with the outer one as parent, and keeps its own figures; model calls, store calls and phases count
against the run in progress; a store call the store makes to itself counts once; a run inside a
queued task carries the task's id and queue time.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from hindsight_api import operation_runs as ops
from hindsight_api.extensions.model_call import OperationRun


@pytest.fixture
def runs():
    got: list[OperationRun] = []
    ops.set_sink(got.append)
    yield got
    ops.set_sink(None)


def test_a_run_reaches_the_sink_with_its_figures(runs):
    with ops.timed_operation("recall", "bank-1"):
        rid = ops.current_operation_id()
        ops.add_model_call(0.25, 100, 7)
        ops.add_model_call(0.5, 50, 3)
        ops.add_phase("recall.retrieve", 0.01)
        ops.add_phase("recall.retrieve", 0.02)
    [run] = runs
    assert run.id == rid and run.operation == "recall" and run.bank_id == "bank-1"
    assert (run.model_calls, run.model_ms, run.input_tokens, run.output_tokens) == (2, 750, 150, 10)
    assert run.phases_ms == {"recall.retrieve": 30.0}
    assert run.success and run.error is None and run.parent_id is None and run.task_id is None
    assert run.ended_at >= run.started_at and run.duration_ms >= 0
    assert ops.current_operation_id() is None, "unbound once it ends"


def test_a_failed_run_is_recorded_and_the_error_still_raises(runs):
    with pytest.raises(ValueError), ops.timed_operation("reflect", "b"):
        raise ValueError("boom")
    [run] = runs
    assert not run.success and run.error == "ValueError: boom"


def test_a_nested_run_is_its_own_with_the_outer_as_parent(runs):
    with ops.timed_operation("mental_model_refresh", "b"):
        outer = ops.current_operation_id()
        with ops.timed_operation("reflect", "b"):
            ops.add_model_call(1.0, 10, 1)
        ops.add_model_call(0.1, 1, 1)
    inner, top = runs
    assert inner.parent_id == outer and top.id == outer
    assert (inner.model_calls, top.model_calls) == (1, 1), "each counts its own calls only"


async def test_concurrent_work_inside_a_run_counts_against_it(runs):
    async def call():
        await asyncio.sleep(0)
        ops.add_model_call(0.1, 1, 0)

    with ops.timed_operation("retain", "b"):
        await asyncio.gather(*(call() for _ in range(5)))
    assert runs[0].model_calls == 5


class _Store:
    def __init__(self):
        self.n = 0

    async def get_memories(self, x):
        await asyncio.sleep(0.01)
        return await self.count(x)

    async def count(self, x):
        self.n += 1
        return x

    def sync_name(self):
        return "store"


async def test_store_calls_are_timed_once_and_the_store_keeps_its_identity(runs):
    store = _Store()
    assert ops.time_store_calls(store) is store and isinstance(store, _Store)
    assert await store.get_memories(1) == 1, "outside a run: passes through"
    with ops.timed_operation("recall", "b"):
        await store.get_memories(2)
        await store.count(3)
    [run] = runs
    assert run.store_calls == 2, "get_memories calling count counts once"
    assert run.store_ms >= 10
    assert store.sync_name() == "store" and store.n == 3


def test_a_queued_task_gives_its_runs_its_id_and_queue_time(runs):
    queued = datetime(2026, 10, 23, 9, tzinfo=timezone.utc)
    with ops.worker_task("task-1", queued):
        with ops.timed_operation("retain", "b"):
            with ops.timed_operation("consolidation", "b"):
                pass
    inner, top = runs
    assert top.task_id == inner.task_id == "task-1"
    assert top.queued_at == queued and inner.queued_at is None, "queue time belongs to the top run"


def test_without_a_sink_nothing_is_kept(runs):
    ops.set_sink(None)
    with ops.timed_operation("recall", "b"):
        ops.add_model_call(1, 1, 1)
    assert runs == []


def test_a_sink_that_raises_never_fails_the_operation():
    def bad(_run):
        raise RuntimeError("sink down")

    ops.set_sink(bad)
    try:
        with ops.timed_operation("recall", "b"):
            pass
    finally:
        ops.set_sink(None)


def test_engine_phases_feed_the_run_even_with_metrics_off(runs):
    from hindsight_api.metrics import NoOpMetricsCollector

    m = NoOpMetricsCollector()
    with ops.timed_operation("retain", "b"):
        m.record_retain_phase("extract", 0.5)
        m.record_recall_phase("retrieve", 0.1)
        m.record_recall_phase("arm_semantic", 0.1, diagnostic=True)
    assert runs[0].phases_ms == {"retain.extract": 500.0, "recall.retrieve": 100.0}
