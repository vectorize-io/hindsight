"""
Test the client-level retain suspension seam.

``Hindsight.suspend_retains`` lets a caller run a read-only session against a
real bank: recall and reflect keep working, while every retain entry point
becomes a no-op that sends no request. These tests pin that both halves hold
and that leaving the scope restores normal retains.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_client import Hindsight

BANK = "bank-1"


def _make_client():
    client = Hindsight(base_url="http://localhost:8888")
    client._memory_api = MagicMock()
    client._memory_api.retain_memories = AsyncMock(return_value=MagicMock())
    client._memory_api.recall_memories = AsyncMock(return_value=MagicMock())
    client._files_api = MagicMock()
    client._files_api.file_retain = AsyncMock(return_value=MagicMock())
    return client


def test_retain_is_suppressed_while_suspended():
    client = _make_client()

    with client.suspend_retains():
        response = client.retain(BANK, "should not be stored")

    assert client._memory_api.retain_memories.await_count == 0
    assert response.items_count == 0
    assert response.success is True
    assert response.bank_id == BANK


def test_retain_batch_is_suppressed_while_suspended():
    client = _make_client()

    with client.suspend_retains():
        response = client.retain_batch(BANK, [{"content": "a"}, {"content": "b"}])

    assert client._memory_api.retain_memories.await_count == 0
    assert response.items_count == 0


def test_retain_files_is_suppressed_while_suspended(tmp_path):
    client = _make_client()
    sample = tmp_path / "note.txt"
    sample.write_text("hello")

    with client.suspend_retains():
        response = client.retain_files(BANK, [sample])

    assert client._files_api.file_retain.await_count == 0
    assert response.operation_ids == []


def test_recall_still_reaches_the_api_while_suspended():
    client = _make_client()

    with client.suspend_retains():
        client.recall(BANK, "a question")

    assert client._memory_api.recall_memories.await_count == 1


def test_retain_resumes_after_the_block():
    client = _make_client()

    with client.suspend_retains():
        client.retain(BANK, "dropped")
    client.retain(BANK, "stored")

    assert client._memory_api.retain_memories.await_count == 1


def test_retains_are_enabled_outside_a_suspension_scope():
    client = _make_client()

    client.retain(BANK, "stored")
    assert client._memory_api.retain_memories.await_count == 1


async def test_retain_suspension_does_not_suppress_another_task():
    client = _make_client()
    task_a_suspended = asyncio.Event()
    task_b_retained = asyncio.Event()

    async def read_only_task():
        with client.suspend_retains():
            task_a_suspended.set()
            await task_b_retained.wait()
            return await client.aretain(BANK, "task A should not be stored")

    async def normal_task():
        await task_a_suspended.wait()
        response = await client.aretain(BANK, "task B should be stored")
        task_b_retained.set()
        return response

    task_a_response, task_b_response = await asyncio.gather(read_only_task(), normal_task())

    assert task_a_response.items_count == 0
    assert task_b_response is client._memory_api.retain_memories.return_value
    assert client._memory_api.retain_memories.await_count == 1


def test_retain_suspension_scopes_are_nested():
    client = _make_client()

    with client.suspend_retains():
        client.retain(BANK, "outer")
        with client.suspend_retains():
            client.retain(BANK, "inner")
        client.retain(BANK, "outer again")
    client.retain(BANK, "stored")

    assert client._memory_api.retain_memories.await_count == 1


def test_retain_suspension_is_reset_after_an_exception():
    client = _make_client()

    with pytest.raises(RuntimeError, match="stop"):
        with client.suspend_retains():
            raise RuntimeError("stop")
    client.retain(BANK, "stored")

    assert client._memory_api.retain_memories.await_count == 1
