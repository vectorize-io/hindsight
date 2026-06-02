"""Regression tests for issue #1833.

When fact extraction failed during an async retain, the engine used to silently
drop the document's memory: ``extract_facts_from_contents`` ran per-content
extractions with ``asyncio.gather(..., return_exceptions=True)`` and converted
*every* exception — including the error the inner ``extract_facts_from_text``
deliberately raises to trigger a retry — into an empty ``([], [], TokenUsage())``
result. The streaming producer therefore never saw an error, the worker's
``RetryTaskAt`` machinery never fired, and the operation was marked ``completed``
with 0 facts saved.

The fix never swallows: *any* extraction failure propagates so the worker
retries the task and ultimately fails it loudly if the problem persists, instead
of committing the document with 0 facts. This is provider-agnostic — it does not
depend on recognizing a specific provider's exception types.

These tests drive a ``batch_retain`` task through the *real* ``WorkerPoller`` +
``MemoryEngine.execute_task`` (the production path) with a mock LLM that fails
only on the ``retain_extract_facts`` scope, and assert the operation is retried
(reset to ``pending`` with ``retry_count`` bumped) rather than silently completed.
"""

import json
import uuid

import pytest

from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.worker import WorkerPoller
from hindsight_api.worker.poller import ClaimedTask

# Worker tests share the async_operations table; keep them on one xdist worker.
pytestmark = pytest.mark.xdist_group("worker_tests")


class RateLimitLikeError(Exception):
    """Stand-in for a provider rate-limit / 429 raised mid-extraction."""


class GeminiLikeAPIError(Exception):
    """Stand-in for a non-OpenAI provider error (e.g. Gemini ``APIError``).

    Included to prove the fix is provider-agnostic: it propagates without the
    engine needing to recognize any specific provider's exception types.
    """


async def _count_memory_units(memory, bank_id: str) -> int:
    pool = await memory._get_pool()
    return await pool.fetchval(
        "SELECT COUNT(*) FROM memory_units WHERE bank_id = $1",
        bank_id,
    )


async def _run_retain_through_worker(memory, extraction_error: Exception):
    """Enqueue a one-item retain and run it through the real poller + executor.

    ``extraction_error`` is raised by the mock LLM on the ``retain_extract_facts``
    scope only; every other scope keeps its normal mock behavior. Returns the
    final ``async_operations`` row and the bank_id.
    """
    bank_id = f"test-worker-extract-fail-{uuid.uuid4().hex[:8]}"
    operation_id = uuid.uuid4()

    # Retain resolves its own LLM config (``_retain_llm_config.with_config(...)``),
    # so patching a single engine attribute misses it. Patch the mock provider's
    # ``call`` at the class level and gate on the extraction scope.
    original_call = MockLLM.call

    async def call_raising_on_extract(self, *args, **kwargs):
        if kwargs.get("scope") == "retain_extract_facts":
            raise extraction_error
        return await original_call(self, *args, **kwargs)

    MockLLM.call = call_raising_on_extract

    backend = await memory._get_backend()
    pool = await memory._get_pool()

    # Bank row must exist before async_operations (FK), then enqueue a child
    # retain task exactly as submit_async_retain() would persist it.
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )
    task_payload = {
        "type": "batch_retain",
        "operation_id": str(operation_id),
        "bank_id": bank_id,
        "contents": [{"content": "Alice moved to Berlin in March 2024 and joined Acme as a staff engineer."}],
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

    poller = WorkerPoller(
        backend=backend,
        worker_id="test-worker-1",
        executor=memory.execute_task,
    )
    claimed = ClaimedTask(operation_id=str(operation_id), task_dict=task_payload, schema=None)
    try:
        await poller.execute_task(claimed)
        completed = await poller.wait_for_active_tasks(timeout=30.0)
        assert completed, "Worker task did not settle within timeout"
    finally:
        MockLLM.call = original_call

    final = await pool.fetchrow(
        "SELECT status, retry_count, error_message FROM async_operations WHERE operation_id = $1",
        operation_id,
    )
    return final, bank_id


@pytest.mark.parametrize(
    "extraction_error",
    [
        RateLimitLikeError("429 Too Many Requests: rate limit exceeded"),
        GeminiLikeAPIError("503 Service Unavailable: model overloaded"),
        ValueError("Model does not support the required output token limit."),
    ],
    ids=["rate_limit", "non_openai_provider_5xx", "value_error"],
)
@pytest.mark.asyncio
async def test_extraction_failure_is_retried_never_silently_completed(memory, extraction_error):
    """An extraction-LLM failure must NEVER complete the op with 0 facts.

    Regardless of the failure type or provider, the error propagates into the
    worker's ``RetryTaskAt`` machinery, which resets the task to ``pending`` and
    bumps ``retry_count``. The pre-fix bug marked the op ``completed`` with 0
    memory_units and ``retry_count`` 0, silently losing the document's memory.
    """
    final, bank_id = await _run_retain_through_worker(memory, extraction_error)
    unit_count = await _count_memory_units(memory, bank_id)

    assert final["status"] != "completed", (
        "BUG (#1833): extraction failure was swallowed — operation marked 'completed' with "
        f"{unit_count} memory_units saved. The document's memory was silently dropped and the "
        "RetryTaskAt machinery never fired."
    )
    assert final["status"] == "pending", f"expected task reset to 'pending' for retry, got {final['status']!r}"
    assert final["retry_count"] == 1, f"expected retry_count bumped to 1, got {final['retry_count']}"
