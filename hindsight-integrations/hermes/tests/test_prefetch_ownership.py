"""Event-ordered races: a bounded join is not worker cancellation."""

import threading

import pytest

from conftest import FakeClient


@pytest.mark.parametrize("method", ["recall", "reflect"])
def test_waiting_prefetch_keeps_scheduled_bank(provider, monkeypatch, method):
    instance, client = provider({"bank_id_template": "bank-{session}", "recall_prefetch_method": method})
    entered, release = threading.Event(), threading.Event()

    def wait_for_retain(timeout):
        entered.set()
        assert release.wait(5)
        return True

    monkeypatch.setattr(instance, "_wait_for_retains_drained", wait_for_retain)
    # Simulate the bounded join expiring without spending three seconds asleep.
    monkeypatch.setattr(instance, "_join_prefetch", lambda *a, **kw: None)
    old_bank = instance._bank_id
    instance.queue_prefetch("old query")
    worker = instance._prefetch_thread
    try:
        assert entered.wait(5)
        instance.on_session_switch("session-2")
        assert instance._bank_id != old_bank
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    calls = client.recalls if method == "recall" else client.reflects
    assert [call["bank_id"] for call in calls] == [old_bank]


@pytest.mark.parametrize("method", ["recall", "reflect"])
@pytest.mark.parametrize("outcome", ["success", "empty", "error"])
@pytest.mark.parametrize("new_completed", [False, True])
def test_stale_completion_cannot_change_new_prefetch(provider, monkeypatch, method, outcome, new_completed):
    entered, release = threading.Event(), threading.Event()
    new_entered, new_release = threading.Event(), threading.Event()

    class BlockedClient(FakeClient):
        async def arecall(self, **kwargs):
            return await self.respond(super().arecall, kwargs)

        async def areflect(self, **kwargs):
            return await self.respond(super().areflect, kwargs)

        async def respond(self, operation, kwargs):
            # The client's loop must stay free for the other worker.
            import asyncio

            if kwargs["query"] == "new query":
                new_entered.set()
                assert await asyncio.to_thread(new_release.wait, 5)
            if kwargs["query"] == "old query":
                entered.set()
                assert await asyncio.to_thread(release.wait, 5)
                if outcome == "error":
                    raise RuntimeError("old request failed")
                if outcome == "empty":
                    return type("Empty", (), {"results": [], "text": ""})()
            return await operation(**kwargs)

    instance, _ = provider(
        {"bank_id_template": "bank-{session}", "recall_prefetch_method": method, "prefetch_waits_for_retain": False},
        client=BlockedClient(recall_texts=["owned result"], reflect_text="owned result"),
    )
    monkeypatch.setattr(instance, "_join_prefetch", lambda *a, **kw: None)
    instance.queue_prefetch("old query")
    old_worker = instance._prefetch_thread
    try:
        assert entered.wait(5)
        instance.on_session_switch("session-2")
        instance.queue_prefetch("new query")
        new_worker = instance._prefetch_thread
        assert new_entered.wait(5)
        if new_completed:
            new_release.set()
            new_worker.join(5)
            assert not new_worker.is_alive()
            assert "owned result" in instance.prefetch("consume new result")
        expected_count = 1 if new_completed and method == "recall" else 0
        release.set()
        old_worker.join(5)
        assert not old_worker.is_alive()
        assert instance._prefetch_thread is new_worker
        assert new_worker.is_alive() is not new_completed
        assert instance._last_recall_count == expected_count
        assert instance._prefetch_count == 0
        assert instance.prefetch("no stale result") == ""
    finally:
        release.set()
        new_release.set()
        old_worker.join(5)
        if instance._prefetch_thread is not old_worker:
            instance._prefetch_thread.join(5)
    assert not instance._prefetch_thread.is_alive()


@pytest.mark.parametrize("method", ["recall", "reflect"])
@pytest.mark.parametrize("stage", ["retain_wait", "result"])
def test_shutdown_does_not_publish_late_prefetch(provider, monkeypatch, method, stage):
    entered, release = threading.Event(), threading.Event()
    instance, _ = provider({"recall_prefetch_method": method}, client=FakeClient(["old"], "old"))

    def wait_for_retain(timeout):
        entered.set()
        assert release.wait(5)
        return True

    if stage == "retain_wait":
        monkeypatch.setattr(instance, "_wait_for_retains_drained", wait_for_retain)
    else:
        do_recall = instance._do_recall

        def wait_after_recall(*args, **kwargs):
            result = do_recall(*args, **kwargs)
            entered.set()
            assert release.wait(5)
            return result

        monkeypatch.setattr(instance, "_do_recall", wait_after_recall)
    monkeypatch.setattr(instance, "_join_prefetch", lambda *a, **kw: None)
    instance.queue_prefetch("old query")
    worker = instance._prefetch_thread
    try:
        assert entered.wait(5)
        instance.shutdown()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert instance._client is None
    assert instance.prefetch("after shutdown") == ""
