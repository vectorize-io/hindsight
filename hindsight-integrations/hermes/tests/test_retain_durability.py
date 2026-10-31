"""Automatic retains are not lost when a write fails: the plugin resends a failed write
(#5218), asks the server to retry an operation it failed (#5591), and stops appending to a
bank that keeps no document text (#5599)."""

import json
import types

from hindsight_client_api.exceptions import ApiException

from conftest import FakeClient


class FlakyClient(FakeClient):
    """Fails the next ``failures`` retains with ``error``; records every attempt."""

    def __init__(self, failures: int = 0, error: Exception | None = None, op_status: str = "completed"):
        super().__init__()
        self.failures, self.error = failures, error or ConnectionError("server unreachable")
        self.attempts: list[dict] = []
        self.op_status = op_status
        self.retried: list[str] = []
        self.operations = types.SimpleNamespace(
            get_operation_status=self._get_operation_status, retry_operation=self._retry_operation
        )

    async def aretain_batch(self, **kwargs):
        self.attempts.append(kwargs)
        if self.failures:
            self.failures -= 1
            raise self.error
        self.retains.append(kwargs)
        return types.SimpleNamespace(operation_id=kwargs.get("operation_id"), operation_ids=None)

    async def _get_operation_status(self, bank_id, operation_id):
        return types.SimpleNamespace(status=self.op_status)

    async def _retry_operation(self, bank_id, operation_id):
        self.retried.append(operation_id)
        self.op_status = "pending"
        return types.SimpleNamespace(success=True)


def _contents(calls: list[dict]) -> list[list[str]]:
    return [[m["content"] for m in json.loads(call["items"][0]["content"])] for call in calls]


def test_a_failed_retain_is_resent_with_the_next_one(provider):
    """#5218: the buffer had already moved on, so the failed turn was gone for good."""
    client = FlakyClient(failures=1)
    instance, _ = provider({"prefetch_waits_for_retain": False}, client=client)
    instance.sync_turn("first", "one")
    instance._retain_queue.join()
    instance.sync_turn("second", "two")
    instance.shutdown()

    assert _contents(client.retains) == [["User: first", "Assistant: one"], ["User: second", "Assistant: two"]]
    # The resend reuses the failed attempt's operation_id: if the server did get it, it is a replay.
    assert client.attempts[0]["operation_id"] == client.retains[0]["operation_id"]
    assert client.retains[0]["operation_id"] != client.retains[1]["operation_id"]


def test_a_failed_retain_is_resent_at_shutdown(provider):
    client = FlakyClient(failures=1)
    instance, _ = provider({}, client=client)
    instance.sync_turn("only turn", "reply")
    instance._retain_queue.join()
    instance.shutdown()

    assert _contents(client.retains) == [["User: only turn", "Assistant: reply"]]


def test_a_retain_that_can_never_succeed_is_not_resent(provider):
    client = FlakyClient(failures=1, error=ApiException(status=422, reason="Unprocessable"))
    instance, _ = provider({}, client=client)
    instance.sync_turn("bad", "turn")
    instance._retain_queue.join()
    instance.sync_turn("good", "turn")
    instance.shutdown()

    assert _contents(client.retains) == [["User: good", "Assistant: turn"]]


def test_a_retain_is_dropped_after_too_many_attempts(provider, monkeypatch):
    import hindsight_hermes as plugin

    monkeypatch.setattr(plugin, "_RETAIN_MAX_ATTEMPTS", 2)
    client = FlakyClient(failures=2)
    instance, _ = provider({"prefetch_waits_for_retain": False}, client=client)
    instance.sync_turn("lost", "turn")
    instance._retain_queue.join()  # attempt 1
    instance.sync_turn("second", "turn")  # attempt 2 of the first fails: dropped; the second waits
    instance._retain_queue.join()
    instance.shutdown()

    assert _contents(client.retains) == [["User: second", "Assistant: turn"]]


def test_an_operation_the_server_failed_is_retried_there(provider):
    """#5591: a server-side "failed" counted as done, and the turn was lost."""
    client = FlakyClient(op_status="failed")
    instance, _ = provider({"prefetch_waits_for_retain": False}, client=client)
    instance.sync_turn("first", "one")
    instance._retain_queue.join()
    instance.sync_turn("second", "two")  # the writer checks the first op before sending this one
    instance.shutdown()

    assert client.retried == [client.retains[0]["operation_id"]]


def test_a_completed_operation_is_not_retried(provider):
    client = FlakyClient(op_status="completed")
    instance, _ = provider({"prefetch_waits_for_retain": False}, client=client)
    instance.sync_turn("first", "one")
    instance._retain_queue.join()
    instance.sync_turn("second", "two")
    instance.shutdown()

    assert client.retried == []
    assert len(instance._unsettled_retain_ops) <= 1  # the first op is settled and forgotten


def test_server_retries_stop_at_the_limit(provider, monkeypatch):
    import hindsight_hermes as plugin

    monkeypatch.setattr(plugin, "_SERVER_RETRY_LIMIT", 1)
    client = FlakyClient(op_status="failed")
    client._retry_operation_orig = client._retry_operation

    async def _retry_and_fail_again(bank_id, operation_id):
        await client._retry_operation_orig(bank_id, operation_id)
        client.op_status = "failed"

    client.operations.retry_operation = _retry_and_fail_again
    instance, _ = provider({"prefetch_waits_for_retain": False}, client=client)
    for i in range(4):
        instance.sync_turn(f"turn {i}", "reply")
        instance._retain_queue.join()
    instance.shutdown()

    first_op = client.retains[0]["operation_id"]
    assert client.retried.count(first_op) == 1


def test_a_bank_without_document_text_stops_getting_appends(provider):
    """#5599: every append after the session's first was refused and its turns lost."""
    refused = ApiException(status=400, reason="Bad Request")
    refused.body = json.dumps(
        {"detail": "update_mode='append' is not supported when document text storage (store_document_text) is disabled"}
    )
    client = FlakyClient(failures=1, error=refused)
    instance, _ = provider({"prefetch_waits_for_retain": False}, client=client)
    instance.sync_turn("refused", "append")
    instance._retain_queue.join()
    instance.sync_turn("later", "turn")
    instance.shutdown()

    rescued, later = client.retains
    assert _contents([rescued]) == [["User: refused", "Assistant: append"]]
    assert "update_mode" not in rescued["items"][0]
    assert rescued["document_id"] != client.attempts[0]["document_id"]  # its own document
    # Later turns use the non-append path: a per-process document, rewritten whole each time.
    assert "update_mode" not in later["items"][0]
    assert later["document_id"] == instance._document_id
