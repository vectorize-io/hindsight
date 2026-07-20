"""Refresh operations expose semantic outcome fields in result_metadata (#2605).

Retain operations have carried machine-readable outcome metadata since 0.8.x
(``unit_ids_count`` etc.). These tests pin the refresh-side parity: a completed
refresh_mental_model operation must let a monitoring layer distinguish
"refreshed with real content" from "refreshed empty" by reading
``result_metadata`` alone, without a follow-up content fetch.
"""

import asyncio
import uuid

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine

# The reflect agent's fallback answer when the LLM returns nothing usable
# (hindsight_api/engine/reflect/agent.py). Non-empty, so it survives the
# empty-content guard in refresh_mental_model and completes wire-successful —
# exactly the case populated_content must expose.
NO_ANSWER_STUB = "No answer provided."


@pytest.fixture
async def bank_with_model(memory: MemoryEngine, request_context):
    """Bank with one mental model, unique per test for xdist safety."""
    bank_id = f"test-refresh-meta-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id, request_context=request_context)
    mm = await memory.create_mental_model(
        bank_id=bank_id,
        name="Outcome Meta Model",
        source_query="What outcome fields does refresh expose?",
        content="Original content",
        request_context=request_context,
    )
    yield memory, bank_id, mm
    await memory.delete_bank(bank_id, request_context=request_context)


def _fake_refreshed(content: str, based_on: dict) -> dict:
    """Shape of refresh_mental_model's return value as consumed by the handler."""
    return {
        "content": content,
        "reflect_response": {"text": content, "based_on": based_on, "mental_models": []},
        "source_query": "What outcome fields does refresh expose?",
    }


async def _submit_with_fake_refresh(memory, monkeypatch, bank_id, mm, request_context, refreshed):
    """Submit an async refresh whose reflect outcome is stubbed to `refreshed`.

    The patch must land before submission: the test task backend executes the
    queued task synchronously on submit, so this exercises the real path
    (execute_task -> _handle_refresh_mental_model -> metadata write).
    """

    async def fake_refresh(bank_id, mental_model_id, *, request_context):
        return refreshed

    monkeypatch.setattr(memory, "refresh_mental_model", fake_refresh)
    result = await memory.submit_async_refresh_mental_model(
        bank_id=bank_id,
        mental_model_id=mm["id"],
        request_context=request_context,
    )
    await asyncio.sleep(0.1)
    return result["operation_id"]


@pytest.mark.asyncio
async def test_completed_refresh_enriches_result_metadata(bank_with_model, request_context, monkeypatch):
    """A completed refresh writes content_len / populated_content / based_on_counts."""
    memory, bank_id, mm = bank_with_model
    content = "x" * 120
    based_on = {
        "world": [{"id": "f1"}, {"id": "f2"}, {"id": "f3"}],
        "mental-models": [{"id": "m1"}],
    }

    operation_id = await _submit_with_fake_refresh(
        memory, monkeypatch, bank_id, mm, request_context, _fake_refreshed(content, based_on)
    )

    status = await memory.get_operation_status(
        bank_id=bank_id, operation_id=operation_id, request_context=request_context
    )
    assert status["status"] == "completed"
    meta = status["result_metadata"]

    # Submit-time keys are merged with, not replaced by, the outcome fields:
    # existing consumers join on mental_model_id/name.
    assert meta["mental_model_id"] == mm["id"]
    assert meta["name"] == "Outcome Meta Model"

    assert meta["content_len"] == 120
    assert meta["populated_content"] is True
    assert meta["based_on_counts"] == {"world": 3, "mental-models": 1}


@pytest.mark.asyncio
async def test_no_answer_stub_reads_as_unpopulated(bank_with_model, request_context, monkeypatch):
    """The historical 19-char stub completes wire-successful but must not read as populated."""
    memory, bank_id, mm = bank_with_model

    operation_id = await _submit_with_fake_refresh(
        memory, monkeypatch, bank_id, mm, request_context, _fake_refreshed(NO_ANSWER_STUB, {})
    )

    status = await memory.get_operation_status(
        bank_id=bank_id, operation_id=operation_id, request_context=request_context
    )
    assert status["status"] == "completed"
    meta = status["result_metadata"]

    assert meta["content_len"] == len(NO_ANSWER_STUB)
    assert meta["populated_content"] is False
    assert meta["based_on_counts"] == {}
