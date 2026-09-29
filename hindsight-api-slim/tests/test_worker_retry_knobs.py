"""
Regression tests for the worker task-retry env knobs.

Prior to this fix, `MemoryEngine.execute_task` hardcoded the retry cap to 3 and
the backoff interval to 60 seconds, ignoring the existing
`HINDSIGHT_API_WORKER_MAX_RETRIES` config knob (declared at config.py but never
plumbed into the retry decision). The 4-minute total window (3 x 60s) is shorter
than typical embeddings-provider outages, so operators saw retain operations
permanently failed when the provider returned transient errors for longer than
~4 minutes.

These tests verify that both knobs are read from config (`worker_max_retries`
and `worker_task_retry_backoff_seconds`) and applied to the resulting
`RetryTaskAt`. Because config is cached for the process lifetime, each test
patches the env var, then clears the config cache so the next `get_config()`
re-reads it; the autouse fixture clears the cache around every test so a
patched value can't leak into other tests.
"""

import json
import os
import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.worker.exceptions import RetryTaskAt


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Clear the cached config before and after each test so env-var patches
    in one test can't bleed into another via the process-wide cache."""
    clear_config_cache()
    yield
    clear_config_cache()


async def _ensure_bank(pool, bank_id: str) -> None:
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


async def _create_pending_operation(pool, bank_id: str, operation_id: uuid.UUID) -> None:
    payload = json.dumps(
        {
            "type": "batch_retain",
            "operation_id": str(operation_id),
            "bank_id": bank_id,
            "contents": [{"content": "test", "document_id": "doc-1"}],
        }
    )
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
        VALUES ($1, $2, 'retain', 'pending', $3::jsonb)
        """,
        operation_id,
        bank_id,
        payload,
    )


async def _cleanup(pool, bank_id: str, operation_id: uuid.UUID) -> None:
    await pool.execute("DELETE FROM async_operations WHERE operation_id = $1", operation_id)
    await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_retry_count_cap_honors_env_var(memory):
    """When HINDSIGHT_API_WORKER_MAX_RETRIES=5, retry_count=4 must still retry (4 < 5)."""
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
    operation_id = uuid.uuid4()

    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    await _create_pending_operation(pool, bank_id, operation_id)

    task_dict = {
        "type": "batch_retain",
        "operation_id": str(operation_id),
        "bank_id": bank_id,
        "contents": [{"content": "test", "document_id": "doc-1"}],
        "_retry_count": 4,
    }

    transient = RuntimeError("transient embeddings outage")
    with patch.dict(os.environ, {"HINDSIGHT_API_WORKER_MAX_RETRIES": "5"}):
        clear_config_cache()  # force get_config() to re-read the patched env
        with patch.object(memory, "_handle_batch_retain", side_effect=transient):
            with pytest.raises(RetryTaskAt):
                await memory.execute_task(task_dict)

    await _cleanup(pool, bank_id, operation_id)


@pytest.mark.asyncio
async def test_retry_stops_at_env_var_cap(memory):
    """When HINDSIGHT_API_WORKER_MAX_RETRIES=2, retry_count=2 must NOT retry (2 < 2 is false)."""
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
    operation_id = uuid.uuid4()

    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    await _create_pending_operation(pool, bank_id, operation_id)

    task_dict = {
        "type": "batch_retain",
        "operation_id": str(operation_id),
        "bank_id": bank_id,
        "contents": [{"content": "test", "document_id": "doc-1"}],
        "_retry_count": 2,
    }

    transient = RuntimeError("transient embeddings outage")
    with patch.dict(os.environ, {"HINDSIGHT_API_WORKER_MAX_RETRIES": "2"}):
        clear_config_cache()  # force get_config() to re-read the patched env
        with patch.object(memory, "_handle_batch_retain", side_effect=transient):
            # Cap reached: re-raised as RuntimeError so the poller's _mark_failed path runs.
            # Must NOT be wrapped in RetryTaskAt.
            with pytest.raises(RuntimeError, match="transient embeddings outage"):
                await memory.execute_task(task_dict)

    await _cleanup(pool, bank_id, operation_id)


@pytest.mark.asyncio
async def test_retry_backoff_honors_env_var(memory):
    """The RetryTaskAt.retry_at delta must equal HINDSIGHT_API_WORKER_TASK_RETRY_BACKOFF_SECONDS."""
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
    operation_id = uuid.uuid4()

    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    await _create_pending_operation(pool, bank_id, operation_id)

    task_dict = {
        "type": "batch_retain",
        "operation_id": str(operation_id),
        "bank_id": bank_id,
        "contents": [{"content": "test", "document_id": "doc-1"}],
    }

    custom_backoff = "300"
    transient = RuntimeError("transient embeddings outage")
    env = {
        "HINDSIGHT_API_WORKER_MAX_RETRIES": "5",
        "HINDSIGHT_API_WORKER_TASK_RETRY_BACKOFF_SECONDS": custom_backoff,
    }
    with patch.dict(os.environ, env):
        clear_config_cache()  # force get_config() to re-read the patched env
        with patch.object(memory, "_handle_batch_retain", side_effect=transient):
            before = datetime.now(UTC)
            with pytest.raises(RetryTaskAt) as exc_info:
                await memory.execute_task(task_dict)
            # retry_at is computed as datetime.now(UTC) + timedelta(seconds=backoff).
            # Allow a generous 30-second envelope around the expected delta to absorb
            # CI scheduling jitter without making the assertion flaky.
            delta_seconds = (exc_info.value.retry_at - before).total_seconds()
            assert int(custom_backoff) - 5 <= delta_seconds <= int(custom_backoff) + 30, (
                f"Expected retry delta ~{custom_backoff}s, got {delta_seconds:.1f}s"
            )

    await _cleanup(pool, bank_id, operation_id)


@pytest.mark.asyncio
async def test_defaults_preserve_existing_behavior(memory):
    """With no env vars set, retry count defaults to 3 and backoff defaults to 60s.

    This guards against silent default drift — operators upgrading to a version
    with the env vars wired in must see the same behavior they had before unless
    they explicitly opt in to a different policy.
    """
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
    operation_id = uuid.uuid4()

    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    await _create_pending_operation(pool, bank_id, operation_id)

    task_dict = {
        "type": "batch_retain",
        "operation_id": str(operation_id),
        "bank_id": bank_id,
        "contents": [{"content": "test", "document_id": "doc-1"}],
        "_retry_count": 2,
    }

    transient = RuntimeError("transient embeddings outage")
    # Remove only the two env vars under test, leaving the rest of the environment
    # intact so the test fixture (DB connection, etc.) keeps working.
    saved = {
        k: os.environ.pop(k, None)
        for k in ("HINDSIGHT_API_WORKER_MAX_RETRIES", "HINDSIGHT_API_WORKER_TASK_RETRY_BACKOFF_SECONDS")
    }
    clear_config_cache()  # force get_config() to re-read without the popped env vars
    try:
        with patch.object(memory, "_handle_batch_retain", side_effect=transient):
            before = datetime.now(UTC)
            with pytest.raises(RetryTaskAt) as exc_info:
                await memory.execute_task(task_dict)
            # Default backoff is 60s.
            delta_seconds = (exc_info.value.retry_at - before).total_seconds()
            assert 55 <= delta_seconds <= 90, f"Default backoff should be ~60s, got {delta_seconds:.1f}s"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    await _cleanup(pool, bank_id, operation_id)
