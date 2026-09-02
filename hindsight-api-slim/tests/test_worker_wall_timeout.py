"""Wall-clock ceiling for worker tasks (#3002).

A retain that blocks forever — a lock wait with no deadlock cycle for Postgres
to break, an LLM permit that never frees, a producer parked on a queue nobody
drains — used to hold its worker slot until the process restarted. The operation
stayed 'processing', which the API refuses to either retry or cancel, so once
every slot was held the worker stopped claiming retains entirely.

The per-call timeouts that already existed (LLM request, DB statement, DB
acquire) each bound one step; none bounds the task. These tests cover the outer
ceiling that does, and — just as importantly — that it stays out of the way of
everything else.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.worker.poller import ClaimedTask, _wall_timeout_for


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Config is cached process-wide; clear it around each test so an env patch
    here can't leak into another test."""
    clear_config_cache()
    yield
    clear_config_cache()


def _make_poller(executor, on_wall_timeout=None):
    from hindsight_api.worker import WorkerPoller

    poller = WorkerPoller(
        backend=MagicMock(),
        worker_id="w-test",
        executor=executor,
        on_wall_timeout=on_wall_timeout,
    )
    # Stub the terminal-state handlers so the poller never touches the DB.
    poller._mark_completed = AsyncMock()
    poller._mark_failed = AsyncMock()
    poller._defer_operation = AsyncMock()
    poller._schedule_retry = AsyncMock()
    return poller


async def _run(executor, task_type="batch_retain", with_holder=False, on_wall_timeout=None, schema=None):
    """Drive one task through the poller.

    ``with_holder`` mirrors production, where the poller always binds a StageHolder:
    it is what lets ``set_stage`` inside the executor push back a progress-extending
    ceiling. Left off by default so the ceiling tests that predate it keep exercising
    the plain absolute path.
    """
    from hindsight_api.worker.stage import StageHolder

    poller = _make_poller(executor, on_wall_timeout=on_wall_timeout)
    task = ClaimedTask(
        operation_id=str(uuid.uuid4()),
        task_dict={"type": task_type, "operation_type": task_type, "bank_id": "bank-1"},
        schema=schema,
    )
    holder = StageHolder(stage=f"queued.{task_type}") if with_holder else None
    with patch("hindsight_api.worker.poller.get_metrics_collector", return_value=MagicMock()):
        await poller._execute_task_inner(task, holder)
    return poller, task


class TestWallTimeoutResolution:
    def test_retain_variants_are_bounded(self):
        with patch.dict("os.environ", {"HINDSIGHT_API_RETAIN_WALL_TIMEOUT": "1800"}):
            clear_config_cache()
            assert _wall_timeout_for("retain") == 1800.0
            assert _wall_timeout_for("batch_retain") == 1800.0
            assert _wall_timeout_for("file_convert_retain") == 1800.0

    def test_zero_disables(self):
        with patch.dict("os.environ", {"HINDSIGHT_API_RETAIN_WALL_TIMEOUT": "0"}):
            clear_config_cache()
            assert _wall_timeout_for("batch_retain") is None

    def test_consolidation_is_bounded(self):
        with patch.dict("os.environ", {"HINDSIGHT_API_CONSOLIDATION_WALL_TIMEOUT": "7200"}):
            clear_config_cache()
            assert _wall_timeout_for("consolidation") == 7200.0

    def test_consolidation_default_is_two_hours(self):
        """The default has to hold on its own: a deployment that never sets the var
        is exactly the one that hits the wedge."""
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("HINDSIGHT_API_CONSOLIDATION_WALL_TIMEOUT", None)
            clear_config_cache()
            assert _wall_timeout_for("consolidation") == 7200.0

    def test_zero_disables_consolidation(self):
        with patch.dict("os.environ", {"HINDSIGHT_API_CONSOLIDATION_WALL_TIMEOUT": "0"}):
            clear_config_cache()
            assert _wall_timeout_for("consolidation") is None

    def test_unknown_task_types_are_unbounded(self):
        assert _wall_timeout_for("graph_maintenance") is None
        assert _wall_timeout_for("reflect") is None


class TestWallTimeoutEnforcement:
    @pytest.mark.asyncio
    async def test_wedged_retain_is_cancelled_and_marked_failed(self):
        """The whole point: the executor is cancelled (freeing the slot) and the
        operation lands in 'failed', which the API *will* retry — unlike
        'processing', which it refuses to retry or cancel."""
        import asyncio

        cancelled = asyncio.Event()

        async def wedged(_task_dict):
            try:
                await asyncio.Event().wait()  # never completes
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with patch.dict("os.environ", {"HINDSIGHT_API_RETAIN_WALL_TIMEOUT": "1"}):
            clear_config_cache()
            poller, task = await _run(wedged)

        assert cancelled.is_set(), "executor was not cancelled — the worker slot would stay held"
        poller._mark_completed.assert_not_awaited()
        poller._mark_failed.assert_awaited_once()
        message = poller._mark_failed.await_args.args[1]
        assert "wall-clock limit" in message
        assert "HINDSIGHT_API_RETAIN_WALL_TIMEOUT" in message

    @pytest.mark.asyncio
    async def test_failure_message_carries_the_stage(self):
        """The stage at the moment the ceiling fires is the only breadcrumb
        pointing at *where* the task was stuck, so it has to survive into the
        error the operator reads."""
        import asyncio

        from hindsight_api.worker.stage import set_stage

        async def wedged(_task_dict):
            set_stage("llm.bedrock.retain_extract_facts+structured.queued")
            await asyncio.Event().wait()

        with patch.dict("os.environ", {"HINDSIGHT_API_RETAIN_WALL_TIMEOUT": "1"}):
            clear_config_cache()
            poller = _make_poller(wedged)
            task = ClaimedTask(
                operation_id=str(uuid.uuid4()),
                task_dict={"type": "batch_retain", "operation_type": "batch_retain", "bank_id": "bank-1"},
                schema=None,
            )
            from hindsight_api.worker.stage import StageHolder

            holder = StageHolder(stage="queued.batch_retain")
            with patch("hindsight_api.worker.poller.get_metrics_collector", return_value=MagicMock()):
                await poller._execute_task_inner(task, holder)

        message = poller._mark_failed.await_args.args[1]
        assert "llm.bedrock.retain_extract_facts+structured.queued" in message

    @pytest.mark.asyncio
    async def test_fast_task_is_untouched(self):
        with patch.dict("os.environ", {"HINDSIGHT_API_RETAIN_WALL_TIMEOUT": "60"}):
            clear_config_cache()
            poller, task = await _run(AsyncMock())

        poller._mark_completed.assert_awaited_once_with(task.operation_id, task.schema)
        poller._mark_failed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_inner_timeout_is_not_reported_as_a_wedge(self):
        """A TimeoutError raised *by* the task (an asyncpg command timeout, say)
        must not be dressed up as the wall-clock ceiling firing — that would send
        an operator hunting for a wedge that never happened."""

        async def inner_timeout(_task_dict):
            raise TimeoutError("statement timeout")

        with patch.dict("os.environ", {"HINDSIGHT_API_RETAIN_WALL_TIMEOUT": "600"}):
            clear_config_cache()
            poller, task = await _run(inner_timeout)

        poller._mark_failed.assert_awaited_once()
        message = poller._mark_failed.await_args.args[1]
        assert "wall-clock limit" not in message
        assert "statement timeout" in message

    @pytest.mark.asyncio
    async def test_unbounded_type_is_not_cancelled(self):
        """A slow unknown task runs to completion because only mapped types
        have a wall-clock ceiling."""
        import asyncio

        async def slow(_task_dict):
            await asyncio.sleep(0.2)

        with patch.dict("os.environ", {"HINDSIGHT_API_RETAIN_WALL_TIMEOUT": "1"}):
            clear_config_cache()
            poller, task = await _run(slow, task_type="graph_maintenance")

            poller._mark_completed.assert_awaited_once()
            poller._mark_failed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wedged_consolidation_is_cancelled_and_marked_failed(self):
        """A consolidation that never commits a single batch — the #3726 wedge, which
        held its reserved slot (and, via the pending/processing guard in
        banks_needing_consolidation, the bank's whole reconcile sweep) for 22 days."""
        import asyncio

        cancelled = asyncio.Event()

        async def wedged(_task_dict):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with patch.dict("os.environ", {"HINDSIGHT_API_CONSOLIDATION_WALL_TIMEOUT": "1"}):
            clear_config_cache()
            poller, _ = await _run(wedged, task_type="consolidation", with_holder=True)

        assert cancelled.is_set()
        poller._mark_completed.assert_not_awaited()
        poller._mark_failed.assert_awaited_once()
        message = poller._mark_failed.await_args.args[1]
        assert "made no progress" in message
        assert "HINDSIGHT_API_CONSOLIDATION_WALL_TIMEOUT" in message


class TestProgressExtendsTheCeiling:
    """Consolidation's ceiling bounds time *without progress*, not total runtime.

    A bank with a big backlog consolidates in a loop of batches that each commit
    their own memories, and the loop bumps the task's stage as each one lands. An
    absolute ceiling would kill those jobs mid-backlog — which is worse than the
    wedge it was added to fix, because the reconcile sweep would just re-schedule
    the bank into the same doomed run. Only a job that stops committing is a wedge.
    """

    @pytest.mark.asyncio
    async def test_committed_batches_push_the_deadline_out(self):
        """Total runtime well past the ceiling, but never a gap that long."""
        import asyncio

        from hindsight_api.worker.stage import set_stage

        async def steady_progress(_task_dict):
            for batch in range(6):
                await asyncio.sleep(0.3)
                set_stage(f"consolidation.llm_batch.{batch}")

        with patch.dict("os.environ", {"HINDSIGHT_API_CONSOLIDATION_WALL_TIMEOUT": "1"}):
            clear_config_cache()
            poller, _ = await _run(steady_progress, task_type="consolidation", with_holder=True)

        poller._mark_completed.assert_awaited_once()
        poller._mark_failed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_job_that_stops_committing_is_still_cancelled(self):
        """Extension must not become an escape hatch: once progress stops, the
        clock runs out from the *last* batch, not from task start."""
        import asyncio

        from hindsight_api.worker.stage import set_stage

        cancelled = asyncio.Event()

        async def stalls_after_two_batches(_task_dict):
            for batch in range(2):
                await asyncio.sleep(0.3)
                set_stage(f"consolidation.llm_batch.{batch}")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with patch.dict("os.environ", {"HINDSIGHT_API_CONSOLIDATION_WALL_TIMEOUT": "1"}):
            clear_config_cache()
            poller, _ = await _run(stalls_after_two_batches, task_type="consolidation", with_holder=True)

        assert cancelled.is_set()
        poller._mark_failed.assert_awaited_once()
        message = poller._mark_failed.await_args.args[1]
        assert "made no progress" in message
        # The stage of the last batch that landed is the pointer to where it stalled.
        assert "consolidation.llm_batch.1" in message
        assert "HINDSIGHT_API_CONSOLIDATION_WALL_TIMEOUT" in message

    @pytest.mark.asyncio
    async def test_retain_ceiling_stays_absolute(self):
        """Retain is one document, not a backlog: progress must NOT buy it more time,
        or the ceiling stops bounding the wedge it was written for (#3002)."""
        import asyncio

        from hindsight_api.worker.stage import set_stage

        cancelled = asyncio.Event()

        async def chatty_but_endless(_task_dict):
            try:
                while True:
                    await asyncio.sleep(0.2)
                    set_stage("retain.facts.llm")
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with patch.dict("os.environ", {"HINDSIGHT_API_RETAIN_WALL_TIMEOUT": "1"}):
            clear_config_cache()
            poller, _ = await _run(chatty_but_endless, task_type="batch_retain", with_holder=True)

        assert cancelled.is_set()
        poller._mark_failed.assert_awaited_once()
        message = poller._mark_failed.await_args.args[1]
        assert "exceeded the" in message
        assert "made no progress" not in message

    @pytest.mark.asyncio
    async def test_a_late_stage_change_does_not_raise_into_engine_code(self):
        """set_stage runs on the engine's hot path and is called while a cancelled
        task unwinds. Extending an already-fired ceiling has to be a silent no-op."""
        import asyncio

        from hindsight_api.worker.stage import set_stage

        async def stages_while_unwinding(_task_dict):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                set_stage("consolidation.cleanup")
                raise

        with patch.dict("os.environ", {"HINDSIGHT_API_CONSOLIDATION_WALL_TIMEOUT": "1"}):
            clear_config_cache()
            poller, _ = await _run(stages_while_unwinding, task_type="consolidation", with_holder=True)

        poller._mark_failed.assert_awaited_once()


class TestWallTimeoutNotification:
    """The ceiling cancels the executor, so the engine's own except-blocks never run.

    Every other consolidation failure path fires a failure webhook; without this
    notification a timed-out consolidation would be the one outcome subscribers
    never hear about.
    """

    @pytest.mark.asyncio
    async def test_engine_is_told_after_the_operation_is_failed(self):
        import asyncio

        notified = AsyncMock()

        async def wedged(_task_dict):
            await asyncio.Event().wait()

        with patch.dict("os.environ", {"HINDSIGHT_API_CONSOLIDATION_WALL_TIMEOUT": "1"}):
            clear_config_cache()
            poller, task = await _run(wedged, task_type="consolidation", on_wall_timeout=notified, schema="tenant_x")

        poller._mark_failed.assert_awaited_once()
        notified.assert_awaited_once()
        task_dict, schema, message = notified.await_args.args
        assert task_dict["type"] == "consolidation"
        assert schema == "tenant_x"
        assert "wall-clock limit" in message

    @pytest.mark.asyncio
    async def test_notification_is_not_fired_for_ordinary_failures(self):
        """Only the cancelled-by-ceiling path loses the engine's own handling."""
        notified = AsyncMock()

        async def boom(_task_dict):
            raise ValueError("nope")

        with patch.dict("os.environ", {"HINDSIGHT_API_CONSOLIDATION_WALL_TIMEOUT": "600"}):
            clear_config_cache()
            poller, _ = await _run(boom, task_type="consolidation", on_wall_timeout=notified)

        poller._mark_failed.assert_awaited_once()
        notified.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failing_notification_does_not_swallow_the_failure(self):
        """The operation is already marked failed by the time we notify; a broken
        webhook must not turn that into an unhandled error in the poller loop."""
        import asyncio

        notified = AsyncMock(side_effect=RuntimeError("webhook down"))

        async def wedged(_task_dict):
            await asyncio.Event().wait()

        with patch.dict("os.environ", {"HINDSIGHT_API_CONSOLIDATION_WALL_TIMEOUT": "1"}):
            clear_config_cache()
            poller, _ = await _run(wedged, task_type="consolidation", on_wall_timeout=notified)

        poller._mark_failed.assert_awaited_once()
        notified.assert_awaited_once()
