"""Regression tests for issue #4858.

A failed attempt that the worker retries wrote its error into ``error_message``;
when a later attempt succeeded, the operation became ``completed`` but still
carried that old error, so monitors could not tell it from a failed run. The
error now moves to ``result_metadata.last_retry_error`` and ``error_message`` is
cleared on completion.

The list and single-operation reads also named the same fields differently
(``id``/``operation_id``, ``task_type``/``operation_type``) and only the list
surfaced ``mental_model_id``; both now carry both names.
"""

import json
import uuid

import httpx
import pytest

from hindsight_api.api import create_app
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.worker import WorkerPoller
from hindsight_api.worker.poller import ClaimedTask

# Worker tests share the async_operations table; keep them on one xdist worker.
pytestmark = pytest.mark.xdist_group("worker_tests")


@pytest.mark.asyncio
async def test_completed_after_retry_clears_error_message(memory, request_context):
    bank_id = f"test-retry-clear-{uuid.uuid4().hex[:8]}"
    operation_id = uuid.uuid4()
    backend = await memory._get_backend()
    pool = await memory._get_pool()

    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )
    task_payload = {
        "type": "batch_retain",
        "operation_id": str(operation_id),
        "bank_id": bank_id,
        "contents": [{"content": "Alice moved to Berlin in March 2024."}],
    }
    await pool.execute(
        """
        INSERT INTO async_operations
            (operation_id, bank_id, operation_type, status, task_payload, worker_id, claimed_at)
        VALUES ($1, $2, 'retain', 'processing', $3::jsonb, 'test-worker-1', now())
        """,
        operation_id,
        bank_id,
        json.dumps(task_payload),
    )
    poller = WorkerPoller(backend=backend, worker_id="test-worker-1", executor=memory.execute_task)

    async def run_attempt(retry_count: int) -> None:
        claimed = ClaimedTask(
            operation_id=str(operation_id),
            task_dict={**task_payload, "_retry_count": retry_count},
            schema=None,
        )
        await poller.execute_task(claimed)
        assert await poller.wait_for_active_tasks(timeout=30.0)

    # Attempt 1: extraction fails transiently -> the worker schedules a retry.
    original_call = MockLLM.call

    async def failing_call(self, *args, **kwargs):
        if kwargs.get("scope") == "retain_extract_facts":
            raise RuntimeError("503 Service Unavailable: model overloaded")
        return await original_call(self, *args, **kwargs)

    MockLLM.call = failing_call
    try:
        await run_attempt(0)
    finally:
        MockLLM.call = original_call

    pending = await memory.get_operation_status(bank_id, str(operation_id), request_context=request_context)
    assert pending["status"] == "pending"
    assert "503" in pending["error_message"]

    # Attempt 2: the worker re-claims it and it succeeds.
    await pool.execute(
        "UPDATE async_operations SET status = 'processing', worker_id = 'test-worker-1', claimed_at = now() "
        "WHERE operation_id = $1",
        operation_id,
    )
    await run_attempt(1)

    status = await memory.get_operation_status(bank_id, str(operation_id), request_context=request_context)
    assert status["status"] == "completed"
    assert status["error_message"] is None, f"completed operation kept a stale error: {status['error_message']!r}"
    assert "503" in status["result_metadata"]["last_retry_error"]

    await pool.execute("DELETE FROM async_operations WHERE operation_id = $1", operation_id)


@pytest.mark.asyncio
async def test_list_and_single_read_share_field_names(memory):
    bank_id = f"test-op-shape-{uuid.uuid4().hex[:8]}"
    operation_id = uuid.uuid4()
    pool = await memory._get_pool()
    await pool.execute("INSERT INTO banks (bank_id) VALUES ($1) ON CONFLICT (bank_id) DO NOTHING", bank_id)
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status, result_metadata)
        VALUES ($1, $2, 'refresh_mental_model', 'completed', $3::jsonb)
        """,
        operation_id,
        bank_id,
        json.dumps({"mental_model_id": "mm-1"}),
    )

    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        listed = (await client.get(f"/v1/default/banks/{bank_id}/operations")).json()["operations"][0]
        single = (await client.get(f"/v1/default/banks/{bank_id}/operations/{operation_id}")).json()

    for key in ("id", "operation_id", "task_type", "operation_type", "mental_model_id"):
        assert listed[key] == single[key], f"{key}: list={listed[key]!r} single={single[key]!r}"
    assert single["id"] == str(operation_id)
    assert single["task_type"] == "refresh_mental_model"
    assert single["mental_model_id"] == "mm-1"

    await pool.execute("DELETE FROM async_operations WHERE operation_id = $1", operation_id)
