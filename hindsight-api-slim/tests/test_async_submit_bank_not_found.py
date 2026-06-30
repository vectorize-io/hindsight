"""Regression test: submitting an async op for a bank that doesn't exist must
raise a clean validation error, not a raw asyncpg `ForeignKeyViolationError`.

`_submit_async_operation` inserts into `async_operations`, which has an FK to
`banks.bank_id`. If a caller submits for a missing bank (typo, race against a
deletion, integration that derives bank IDs before the bank is created), the
INSERT raises `asyncpg.exceptions.ForeignKeyViolationError`. The FastAPI
endpoint's broad `except Exception` then surfaces it as a 500 — but this is
a client error, not a server error, and should be a 404.

This test exercises the call directly via `MemoryEngine.submit_async_*` so
the failure mode is observable without spinning up the HTTP layer.
"""

import uuid

import pytest

from hindsight_api.extensions.operation_validator import OperationValidationError

pytestmark = pytest.mark.xdist_group("async_submit_bank_not_found_tests")


@pytest.fixture
def no_inline_execution(memory):
    """Prevent SyncTaskBackend from running the submitted op inline so we
    only test the submit-path failure, not downstream execution."""

    async def _noop(_payload):
        return None

    original = memory._task_backend.submit_task
    memory._task_backend.submit_task = _noop
    yield
    memory._task_backend.submit_task = original


@pytest.mark.asyncio
async def test_consolidation_submit_on_missing_bank_raises_validation_error(
    memory, request_context, no_inline_execution
):
    """A `/consolidate` submit against a bank that doesn't exist must raise
    OperationValidationError(404), not a raw asyncpg FK violation that bubbles
    out as a 500 from the API."""
    missing_bank = f"does-not-exist-{uuid.uuid4().hex[:8]}"

    with pytest.raises(OperationValidationError) as exc_info:
        await memory.submit_async_consolidation(
            bank_id=missing_bank,
            request_context=request_context,
        )

    assert exc_info.value.status_code == 404
    assert missing_bank in exc_info.value.reason


@pytest.mark.asyncio
async def test_scoped_consolidation_submit_on_missing_bank_raises_validation_error(
    memory, request_context, no_inline_execution
):
    """Scoped consolidates (with `observation_scopes`) take the
    `dedupe_by_bank=False` branch, which historically skipped the bank lock
    entirely and went straight to the FK-violating INSERT. Same 404 contract."""
    missing_bank = f"does-not-exist-{uuid.uuid4().hex[:8]}"

    with pytest.raises(OperationValidationError) as exc_info:
        await memory.submit_async_consolidation(
            bank_id=missing_bank,
            request_context=request_context,
            observation_scopes=[{"tag": "anything"}],
        )

    assert exc_info.value.status_code == 404
    assert missing_bank in exc_info.value.reason


@pytest.mark.asyncio
async def test_graph_maintenance_on_missing_bank_short_circuits(memory, request_context, no_inline_execution):
    """`submit_async_graph_maintenance` has its own short-circuit that checks
    the per-bank queue before calling `_submit_async_operation`. A missing
    bank means an empty queue, so it returns `no_work=True` without reaching
    the FK-violating INSERT. This test pins that behaviour."""
    missing_bank = f"does-not-exist-{uuid.uuid4().hex[:8]}"

    result = await memory.submit_async_graph_maintenance(
        bank_id=missing_bank,
        request_context=request_context,
    )

    assert result == {"operation_id": None, "no_work": True}
