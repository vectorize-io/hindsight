"""execute_task must handle MentalModelRefreshError without a traceback.

refresh_mental_model raises MentalModelRefreshError from _preserve_and_fail
(#3112/#3182): content and watermark stay untouched, retry is intended.
Before this handler the exception fell through execute_task's generic
``except Exception``, which on the overlay called ``traceback.print_exc()``
and on main (#3218) logs ``logger.error(..., exc_info=True)``. Either form
is what the soak watcher treats as unhandled.

These tests isolate execute_task's exception classification by stubbing
_handle_refresh_mental_model. No DB: omit operation_id so the cancel check
is skipped.

Port note vs overlay ca2bd211: main already uses logger.error (not print_exc)
on the generic path. The dedicated MentalModelRefreshError handler logs
logger.error WITHOUT exc_info, then applies the same RetryTaskAt policy.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from hindsight_api.config import get_config
from hindsight_api.engine.memory_engine import MemoryEngine, MentalModelRefreshError
from hindsight_api.worker.exceptions import RetryTaskAt

MM_ID = "mm-test-delta-ops"
BANK_ID = "bank-test-mmrefresh"
REFRESH_ERROR = MentalModelRefreshError(
    f"Refresh failed for mental_model_id={MM_ID}: delta operations did not reach "
    "the document, and the reflect candidate covers only memories newer than the "
    "last refresh, so writing it would drop the rest of the document. Previous "
    "content preserved in DB; reflect_response.refresh_skipped == "
    "'delta_ops_all_skipped' for audit."
)


def _engine() -> MemoryEngine:
    engine = object.__new__(MemoryEngine)
    engine._audit_logger = None
    return engine


def _task(*, retry_count: int) -> dict:
    return {
        "type": "refresh_mental_model",
        "bank_id": BANK_ID,
        "mental_model_id": MM_ID,
        "_retry_count": retry_count,
    }


@pytest.mark.asyncio
async def test_refresh_error_retries_via_logger_error_without_traceback(caplog, capsys):
    """_retry_count=0 -> RetryTaskAt; no Traceback on stderr; ERROR names the skip."""
    engine = _engine()
    with (
        caplog.at_level(logging.ERROR, logger="hindsight_api.engine.memory_engine"),
        patch.object(engine, "_handle_refresh_mental_model", side_effect=REFRESH_ERROR),
        pytest.raises(RetryTaskAt),
    ):
        await engine.execute_task(_task(retry_count=0))

    err = capsys.readouterr().err
    assert "Traceback" not in err
    error_text = "\n".join(r.message for r in caplog.records if r.levelno == logging.ERROR)
    assert MM_ID in error_text
    assert "delta_ops_all_skipped" in error_text
    assert BANK_ID in error_text
    assert "refresh_mental_model" in error_text
    assert "content preserved" in error_text
    # No traceback attached to the designed-skip record.
    skip_records = [r for r in caplog.records if "content preserved" in r.getMessage()]
    assert skip_records
    assert skip_records[0].exc_info is None


@pytest.mark.asyncio
async def test_refresh_error_propagates_at_retry_cap(capsys):
    """At _retry_count == worker_max_retries the MentalModelRefreshError escapes."""
    engine = _engine()
    cap = get_config().worker_max_retries
    with patch.object(engine, "_handle_refresh_mental_model", side_effect=REFRESH_ERROR):
        with pytest.raises(MentalModelRefreshError, match="delta_ops_all_skipped") as excinfo:
            await engine.execute_task(_task(retry_count=cap))
    assert not isinstance(excinfo.value, RetryTaskAt)
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_generic_runtime_error_on_refresh_still_logs_exc_info(caplog):
    """A generic RuntimeError on the same path still uses logger.error(..., exc_info=True).

    Overlay asserted print_exc() on stderr. Main/#3218 replaced that with
    logger.error(..., exc_info=True); silence must not be widened.
    """
    engine = _engine()
    with (
        caplog.at_level(logging.ERROR, logger="hindsight_api.engine.memory_engine"),
        patch.object(
            engine,
            "_handle_refresh_mental_model",
            side_effect=RuntimeError("unexpected refresh boom"),
        ),
        pytest.raises(RetryTaskAt),
    ):
        await engine.execute_task(_task(retry_count=0))

    generic = [r for r in caplog.records if "Task execution failed" in r.getMessage()]
    assert generic
    assert generic[0].exc_info is not None
    assert generic[0].exc_info[0] is RuntimeError
    assert "unexpected refresh boom" in generic[0].getMessage()
