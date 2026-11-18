"""Tests for the mental_model.refreshed webhook event.

A refresh operation that finishes queues exactly one mental_model.refreshed delivery:
``completed`` with the refresh's outcome, or ``failed`` once the refresh has failed for
good. Deterministic — no LLM, the refresh itself is monkeypatched and the queued task is
run through ``execute_task`` the way the worker runs it.
"""

import json
import uuid

import pytest
import pytest_asyncio

from hindsight_api.engine.llm_interface import ProviderContentPolicyError
from hindsight_api.engine.memory_engine import MemoryEngine, MentalModelRefreshError
from hindsight_api.webhooks.manager import WebhookManager
from hindsight_api.webhooks.models import MentalModelRefreshedEventData, WebhookEventType
from hindsight_api.worker.exceptions import RetryTaskAt


@pytest_asyncio.fixture
async def webhook_manager(memory: MemoryEngine):
    """Install a WebhookManager on the engine for the test, then put the original back."""
    original = memory._webhook_manager
    memory._webhook_manager = WebhookManager(backend=memory._backend, global_webhooks=[])
    try:
        yield memory._webhook_manager
    finally:
        memory._webhook_manager = original


async def _bank_with_model(memory: MemoryEngine, request_context, event_types: list[str]) -> tuple[str, str]:
    """A bank holding one pinned mental model and a webhook subscribed to ``event_types``."""
    bank_id = f"mmwh-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    mm_id = f"mm-{uuid.uuid4().hex}"
    async with memory._pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO mental_models (id, bank_id, subtype, name, source_query, content, tags, trigger)
            VALUES ($1, $2, 'pinned', 'Team preferences', 'what does the team prefer', 'body', '{}',
                    '{"refresh_after_consolidation": false}'::jsonb)
            """,
            mm_id,
            bank_id,
        )
        await conn.execute(
            """
            INSERT INTO webhooks (id, bank_id, url, secret, event_types, enabled, created_at, updated_at)
            VALUES ($1, $2, 'https://example.com/mm-hook', NULL, $3, true, NOW(), NOW())
            """,
            uuid.uuid4(),
            bank_id,
            event_types,
        )
    return bank_id, mm_id


async def _queued_refresh(memory: MemoryEngine, bank_id: str, mm_id: str, request_context, monkeypatch) -> dict:
    """Submit a refresh without running it and return the task the worker would claim."""

    async def _never_runs(task_dict):
        return None

    monkeypatch.setattr(memory._task_backend, "submit_task", _never_runs)
    submitted = await memory.submit_async_refresh_mental_model(bank_id, mm_id, request_context=request_context)
    async with memory._pool.acquire() as conn:
        payload = await conn.fetchval(
            "SELECT task_payload FROM async_operations WHERE operation_id = $1",
            uuid.UUID(submitted["operation_id"]),
        )
    task = json.loads(payload) if isinstance(payload, str) else dict(payload)
    task["operation_id"] = submitted["operation_id"]
    return task


async def _deliveries(memory: MemoryEngine, bank_id: str) -> list[dict]:
    """The mental_model.refreshed events queued for the bank, as receivers get them."""
    async with memory._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT task_payload FROM async_operations
            WHERE operation_type = 'webhook_delivery' AND bank_id = $1
              AND task_payload->>'event_type' = 'mental_model.refreshed'
            ORDER BY created_at
            """,
            bank_id,
        )
    events = []
    for row in rows:
        payload = row["task_payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        events.append(json.loads(payload["payload"]))
    return events


def test_event_type_value():
    assert WebhookEventType.MENTAL_MODEL_REFRESHED == "mental_model.refreshed"
    assert MentalModelRefreshedEventData(mental_model_id="mm").outcome is None


@pytest.mark.asyncio
async def test_a_completed_refresh_sends_its_outcome(
    memory: MemoryEngine, webhook_manager, request_context, monkeypatch
):
    bank_id, mm_id = await _bank_with_model(memory, request_context, ["mental_model.refreshed"])
    task = await _queued_refresh(memory, bank_id, mm_id, request_context, monkeypatch)

    async def _refreshed(**kwargs):
        return {
            "id": mm_id,
            "name": "Team preferences",
            "content": "The team prefers short meetings.",
            "reflect_response": {"outcome": "content_written", "based_on": {}},
        }

    monkeypatch.setattr(memory, "refresh_mental_model", _refreshed)
    await memory.execute_task(task)

    events = await _deliveries(memory, bank_id)
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "mental_model.refreshed"
    assert event["status"] == "completed"
    assert event["operation_id"] == task["operation_id"]
    # Unset fields are left out of the payload, as on every webhook event.
    assert event["data"] == {"mental_model_id": mm_id, "name": "Team preferences", "outcome": "content_written"}


@pytest.mark.asyncio
async def test_a_webhook_not_subscribed_to_the_event_gets_nothing(
    memory: MemoryEngine, webhook_manager, request_context, monkeypatch
):
    bank_id, mm_id = await _bank_with_model(memory, request_context, ["consolidation.completed"])
    task = await _queued_refresh(memory, bank_id, mm_id, request_context, monkeypatch)

    async def _refreshed(**kwargs):
        return {"id": mm_id, "name": "Team preferences", "content": "x", "reflect_response": {}}

    monkeypatch.setattr(memory, "refresh_mental_model", _refreshed)
    await memory.execute_task(task)

    assert await _deliveries(memory, bank_id) == []


@pytest.mark.asyncio
async def test_a_refusal_sends_failed_with_its_outcome_and_reason(
    memory: MemoryEngine, webhook_manager, request_context, monkeypatch
):
    bank_id, mm_id = await _bank_with_model(memory, request_context, ["mental_model.refreshed"])
    task = await _queued_refresh(memory, bank_id, mm_id, request_context, monkeypatch)

    async def _refused(**kwargs):
        raise MentalModelRefreshError(
            "rejected",
            outcome="refresh_failed_candidate_rejected",
            reason="candidate_rejected",
            retryable=False,
        )

    monkeypatch.setattr(memory, "refresh_mental_model", _refused)
    # Not retryable: the operation is failed on this attempt.
    await memory.execute_task(task)

    events = await _deliveries(memory, bank_id)
    assert len(events) == 1
    assert events[0]["status"] == "failed"
    assert events[0]["data"]["mental_model_id"] == mm_id
    assert events[0]["data"]["outcome"] == "refresh_failed_candidate_rejected"
    assert events[0]["data"]["failure_reason"] == "candidate_rejected"
    assert "rejected" in events[0]["data"]["error_message"]


@pytest.mark.asyncio
async def test_a_deterministic_error_sends_failed(memory: MemoryEngine, webhook_manager, request_context, monkeypatch):
    bank_id, mm_id = await _bank_with_model(memory, request_context, ["mental_model.refreshed"])
    task = await _queued_refresh(memory, bank_id, mm_id, request_context, monkeypatch)

    async def _declined(**kwargs):
        raise ProviderContentPolicyError("declined")

    monkeypatch.setattr(memory, "refresh_mental_model", _declined)
    await memory.execute_task(task)

    events = await _deliveries(memory, bank_id)
    assert [e["status"] for e in events] == ["failed"]
    assert events[0]["data"]["outcome"] == "refresh_failed_error"
    assert events[0]["data"]["failure_reason"] == "unexpected_error"


@pytest.mark.asyncio
async def test_a_refresh_that_will_be_retried_sends_nothing_until_its_last_attempt(
    memory: MemoryEngine, webhook_manager, request_context, monkeypatch
):
    from hindsight_api.config import get_config

    bank_id, mm_id = await _bank_with_model(memory, request_context, ["mental_model.refreshed"])
    task = await _queued_refresh(memory, bank_id, mm_id, request_context, monkeypatch)

    async def _flaky(**kwargs):
        raise RuntimeError("store went away")

    monkeypatch.setattr(memory, "refresh_mental_model", _flaky)
    with pytest.raises(RetryTaskAt):
        await memory.execute_task({**task, "_retry_count": 0})
    assert await _deliveries(memory, bank_id) == []

    with pytest.raises(RuntimeError):
        await memory.execute_task({**task, "_retry_count": get_config().worker_max_retries})
    events = await _deliveries(memory, bank_id)
    assert [e["status"] for e in events] == ["failed"]
    assert "store went away" in events[0]["data"]["error_message"]


@pytest.mark.asyncio
async def test_a_refresh_cut_off_by_the_wall_timeout_sends_failed(
    memory: MemoryEngine, webhook_manager, request_context, monkeypatch
):
    bank_id, mm_id = await _bank_with_model(memory, request_context, ["mental_model.refreshed"])
    task = await _queued_refresh(memory, bank_id, mm_id, request_context, monkeypatch)

    await memory.on_task_wall_timeout(task, None, "timed out", "llm.tools.attempt=1/2")

    events = await _deliveries(memory, bank_id)
    assert [e["status"] for e in events] == ["failed"]
    assert events[0]["operation_id"] == task["operation_id"]
    assert events[0]["data"]["error_message"] == "timed out"
