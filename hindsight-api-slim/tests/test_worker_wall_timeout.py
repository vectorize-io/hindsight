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


def _make_poller(executor):
    from hindsight_api.worker import WorkerPoller

    poller = WorkerPoller(backend=MagicMock(), worker_id="w-test", executor=executor)
    # Stub the terminal-state handlers so the poller never touches the DB.
    poller._mark_completed = AsyncMock()
    poller._mark_failed = AsyncMock()
    poller._defer_operation = AsyncMock()
    poller._schedule_retry = AsyncMock()
    return poller


async def _run(executor, task_type="batch_retain"):
    poller = _make_poller(executor)
    task = ClaimedTask(
        operation_id=str(uuid.uuid4()),
        task_dict={"type": task_type, "operation_type": task_type, "bank_id": "bank-1"},
        schema=None,
    )
    with patch("hindsight_api.worker.poller.get_metrics_collector", return_value=MagicMock()):
        await poller._execute_task_inner(task)
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

    def test_other_task_types_are_unbounded(self):
        """Only retain is bounded: reflect self-bounds inside the engine, and the
        rest have no reported wedge. Bounding them here would be a behaviour
        change nobody asked for — consolidation on a large bank is legitimately
        long-running."""
        assert _wall_timeout_for("consolidation") is None
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
        """A slow non-retain task runs to completion even past the retain
        ceiling — the timeout is per operation type, not global."""
        import asyncio

        async def slow(_task_dict):
            await asyncio.sleep(0.2)

        with patch.dict("os.environ", {"HINDSIGHT_API_RETAIN_WALL_TIMEOUT": "1"}):
            clear_config_cache()
            poller, task = await _run(slow, task_type="consolidation")

        poller._mark_completed.assert_awaited_once()
        poller._mark_failed.assert_not_awaited()
