"""Progress-aware stuck-task watchdog coverage.

A worker can process a large consolidation for much longer than the reporting
threshold. That is healthy while its stage keeps advancing; a warning is useful
only when one stage has stopped making progress.
"""

import logging
from unittest.mock import MagicMock

from hindsight_api.worker.poller import ActiveTaskInfo, WorkerPoller
from hindsight_api.worker.stage import StageHolder


def _make_poller() -> WorkerPoller:
    return WorkerPoller(backend=MagicMock(), worker_id="w-test", executor=MagicMock())


def _active_task(holder: StageHolder, started_at: float) -> ActiveTaskInfo:
    return ActiveTaskInfo(
        op_type="consolidation",
        bank_id="bank-1",
        schema=None,
        bg_task=MagicMock(),
        started_at=started_at,
        stage_holder=holder,
        task_type="consolidation",
    )


def test_stuck_watchdog_uses_stage_progress_and_resets_for_new_stage(caplog) -> None:
    """Long tasks remain quiet until their current stage stops advancing."""
    now = 1_000.0
    holder = StageHolder(stage="consolidation.llm_batch.1", updated_at=now - 1)
    info = _active_task(holder, started_at=0)
    poller = _make_poller()

    with caplog.at_level(logging.INFO, logger="hindsight_api.worker.poller"):
        poller._log_per_task_lines({"op-1": info}, now)

    assert "[STUCK?]" not in caplog.text
    assert "[STUCK_STACK]" not in caplog.text

    caplog.clear()
    holder.updated_at = now - 301
    poller._log_per_task_lines({"op-1": info}, now)

    assert "[STUCK?]" in caplog.text
    assert "[STUCK_STACK]" in caplog.text
    assert info.last_stack_dump_threshold == 300

    caplog.clear()
    holder.stage = "consolidation.llm_batch.2"
    holder.updated_at = now - 1
    poller._log_per_task_lines({"op-1": info}, now)

    assert "[STUCK?]" not in caplog.text
    assert "[STUCK_STACK]" not in caplog.text

    caplog.clear()
    holder.updated_at = now - 301
    poller._log_per_task_lines({"op-1": info}, now)

    assert "[STUCK?]" in caplog.text
    assert "[STUCK_STACK]" in caplog.text
