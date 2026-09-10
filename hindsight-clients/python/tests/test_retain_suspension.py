"""
Test the client-level retain suspension seam.

``Hindsight.retain_suspended`` lets a caller run a read-only session against a
real bank: recall and reflect keep working, while every retain entry point
becomes a no-op that sends no request. These tests pin that both halves hold
and that clearing the flag restores normal retains.
"""

from unittest.mock import AsyncMock, MagicMock

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

    client.retain_suspended = True
    response = client.retain(BANK, "should not be stored")

    assert client._memory_api.retain_memories.await_count == 0
    assert response.items_count == 0
    assert response.success is True
    assert response.bank_id == BANK


def test_retain_batch_is_suppressed_while_suspended():
    client = _make_client()

    client.retain_suspended = True
    response = client.retain_batch(BANK, [{"content": "a"}, {"content": "b"}])

    assert client._memory_api.retain_memories.await_count == 0
    assert response.items_count == 0


def test_retain_files_is_suppressed_while_suspended(tmp_path):
    client = _make_client()
    sample = tmp_path / "note.txt"
    sample.write_text("hello")

    client.retain_suspended = True
    response = client.retain_files(BANK, [sample])

    assert client._files_api.file_retain.await_count == 0
    assert response.operation_ids == []


def test_recall_still_reaches_the_api_while_suspended():
    client = _make_client()

    client.retain_suspended = True
    client.recall(BANK, "a question")

    assert client._memory_api.recall_memories.await_count == 1


def test_retain_resumes_after_the_block():
    client = _make_client()

    client.retain_suspended = True
    client.retain(BANK, "dropped")
    client.retain_suspended = False
    client.retain(BANK, "stored")

    assert client._memory_api.retain_memories.await_count == 1
    assert client.retain_suspended is False


def test_retains_are_enabled_by_default():
    client = _make_client()

    assert client.retain_suspended is False
    client.retain(BANK, "stored")
    assert client._memory_api.retain_memories.await_count == 1
