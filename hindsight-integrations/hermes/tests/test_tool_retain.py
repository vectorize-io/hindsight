"""Manual retains honor the configured mode and share the automatic retain writer."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import FakeClient


def test_async_tool_returns_before_acknowledgement_and_tracks_operation(provider):
    entered = threading.Event()
    release = threading.Event()

    class BlockingClient(FakeClient):
        async def aretain_batch(self, **kwargs: Any) -> SimpleNamespace:
            await super().aretain_batch(**kwargs)
            entered.set()
            if not await asyncio.to_thread(release.wait, 5):
                raise TimeoutError("test did not release the retain acknowledgement")
            return SimpleNamespace(operation_id="tool-op", operation_ids=[])

    instance, fake = provider({"bank_id": "team"}, client=BlockingClient())
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            call = executor.submit(instance.handle_tool_call, "hindsight_retain", {"content": "Ada likes tea"})
            try:
                assert entered.wait(2), "retain never reached the client"
                result = json.loads(call.result(timeout=2))
                assert result == {"result": "Memory queued for storage."}
                assert not release.is_set()
                assert not instance._pending_retain_ops
            finally:
                release.set()
    finally:
        instance.shutdown()

    assert fake.retains[0]["retain_async"] is True
    assert "retain_async" not in fake.retains[0]["items"][0]
    assert instance._pending_retain_ops == {"tool-op"}
    assert instance._retain_ops_bank_id == "team"


def test_sync_tool_waits_for_storage_and_passes_retain_mode(provider):
    instance, fake = provider({"retain_async": False, "bank_id": "team"})
    try:
        result = json.loads(instance.handle_tool_call("hindsight_retain", {"content": "Ada likes tea"}))
        assert result == {"result": "Memory stored successfully."}
        assert len(fake.retains) == 1
        assert fake.retains[0]["bank_id"] == "team"
        assert fake.retains[0]["retain_async"] is False
        assert "retain_async" not in fake.retains[0]["items"][0]
        assert instance._retain_queue.empty()
    finally:
        instance.shutdown()


def test_queued_tool_preserves_payload_and_order_with_turn_retains(provider, monkeypatch):
    instance, fake = provider({"bank_id": "original", "retain_tags": ["base"], "observation_scopes": "per_tag"})
    # Hold the writer until both calls are queued, then change the live settings.
    monkeypatch.setattr(instance, "_ensure_writer", lambda: None)
    monkeypatch.setattr(instance, "_register_atexit", lambda: None)
    try:
        instance.sync_turn("hello", "hi")
        result = json.loads(
            instance.handle_tool_call(
                "hindsight_retain",
                {
                    "content": "Ada likes tea",
                    "context": "preferences",
                    "tags": ["drink"],
                    "occurred_at": "2026-09-01T12:00:00Z",
                },
            )
        )
        assert result == {"result": "Memory queued for storage."}
        assert fake.retains == []
        assert instance._retain_queue.qsize() == 2
        instance._bank_id = "later"
        instance._retain_tags = ["later"]
        instance._observation_scopes = "combined"
        instance._retain_async = False

        for _ in range(2):
            job = instance._retain_queue.get_nowait()
            try:
                job()
            finally:
                instance._retain_queue.task_done()

        assert [call["bank_id"] for call in fake.retains] == ["original", "original"]
        assert fake.retains[0]["document_id"] == "session-1"
        call = fake.retains[1]
        assert call["retain_async"] is True
        item = call["items"][0]
        assert item["content"] == "Ada likes tea"
        assert item["context"] == "preferences"
        assert item["tags"] == ["base", "drink"]
        assert item["observation_scopes"] == "per_tag"
        assert item["timestamp"] == "2026-09-01T12:00:00Z"
    finally:
        instance.shutdown()


@pytest.mark.parametrize("retain_async", [False, True])
def test_tool_retain_failures_follow_configured_mode(provider, caplog, retain_async):
    class FailingClient(FakeClient):
        async def aretain_batch(self, **kwargs: Any) -> SimpleNamespace:
            raise RuntimeError("backend unavailable")

    instance, _ = provider({"retain_async": retain_async}, client=FailingClient())
    try:
        result = instance.handle_tool_call("hindsight_retain", {"content": "Ada likes tea"})
    finally:
        instance.shutdown()

    if retain_async:
        assert json.loads(result) == {"result": "Memory queued for storage."}
        assert "Hindsight retain failed: backend unavailable" in caplog.text
    else:
        assert result == "ERROR: Failed to store memory: backend unavailable"
