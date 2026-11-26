"""Automatic retains are not lost when a write fails: the plugin resends a failed write
(#5218), asks the server to retry an operation it failed (#5591), and stops appending to a
bank that keeps no document text (#5599)."""

import json
import types
from pathlib import Path

import pytest
from hindsight_client_api.exceptions import ApiException

from conftest import FakeClient, plugin


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


def test_ops_are_not_kept_for_a_wait_that_is_turned_off(provider):
    """#6548: with the prefetch wait off nothing ever cleared these, one per turn for the session."""
    client = FlakyClient()
    instance, _ = provider({"prefetch_waits_for_retain": False}, client=client)
    instance.sync_turn("first", "one")
    instance._retain_queue.join()

    assert instance._pending_retain_ops == set()
    assert instance._unsettled_retain_ops  # the retry watch (#5591) still holds the op
    instance.shutdown()


def test_the_status_line_says_when_saves_fail(provider, monkeypatch):
    """#6819: the line said "saving to memory…" on every turn while every write failed."""
    import hindsight_hermes as plugin

    monkeypatch.setattr(plugin, "_RETAIN_MAX_ATTEMPTS", 2)
    lines: list[str] = []
    client = FlakyClient(failures=2)
    instance, _ = provider({"prefetch_waits_for_retain": False}, client=client, status_callback=lines.append)
    instance.sync_turn("first", "one")  # fails once: kept for a resend
    instance._retain_queue.join()
    instance.sync_turn("second", "two")  # the resend fails too: dropped
    instance._retain_queue.join()
    instance.sync_turn("third", "three")
    instance._retain_queue.join()
    instance.sync_turn("fourth", "four")
    instance.shutdown()

    assert [line.split(" — ", 1)[1] for line in lines] == [
        "saving to memory…",
        "couldn't save to memory yet, will retry (1 waiting)",
        "1 earlier save(s) to memory failed and were dropped (see the Hermes log)",
        "saving to memory…",
    ]


def test_a_required_pre_compress_checkpoint_is_stored_synchronously_before_compaction(provider):
    """#6449: Hermes' strict compression gate needs a v2 provider that saves before it compacts."""
    instance, client = provider()
    messages = [{"role": "user", "content": "my cat is Miso ✓"}, {"role": "assistant", "content": "noted"}]

    assert plugin.HindsightMemoryProvider.pre_compress_checkpoint_api_version == 2
    assert instance.on_pre_compress(messages) == ""  # best effort: turns are already retained
    assert client.retains == []

    assert instance.on_pre_compress(messages, require_checkpoint=True) == ""
    instance.on_pre_compress(messages, require_checkpoint=True)  # a retried compaction
    first, again = client.retains
    assert first["retain_async"] is False
    assert first["document_id"] == again["document_id"]  # same messages, same document
    assert first["document_id"].startswith("session-1-checkpoint-")
    assert json.loads(first["items"][0]["content"]) == messages
    assert "checkpoint:pre-compress" in first["items"][0]["tags"]


def test_a_failed_required_checkpoint_raises_so_hermes_keeps_the_transcript(provider):
    instance, _ = provider(client=FlakyClient(failures=1, error=ApiException(status=400)))
    with pytest.raises(ApiException):
        instance.on_pre_compress([{"role": "user", "content": "x"}], require_checkpoint=True)


def test_a_message_that_is_not_plain_json_still_checkpoints(provider):
    """A raise here would block every compression of the conversation, retries included."""
    instance, client = provider()
    instance.on_pre_compress([{"role": "user", "content": [object()]}], require_checkpoint=True)
    assert len(client.retains) == 1


def test_a_run_that_cannot_write_refuses_a_required_checkpoint(provider):
    instance, client = provider(agent_context="kanban")
    with pytest.raises(RuntimeError):
        instance.on_pre_compress([{"role": "user", "content": "x"}], require_checkpoint=True)
    assert client.retains == []


class HangingClient(FlakyClient):
    """Each retain takes longer than shutdown waits, like a slow LLM extraction (#6915)."""

    def __init__(self, hang: float):
        super().__init__()
        self.hang = hang

    async def aretain_batch(self, **kwargs):
        import asyncio

        await asyncio.sleep(self.hang)
        return await super().aretain_batch(**kwargs)


def test_retains_unsent_at_exit_are_sent_by_the_next_session(provider, monkeypatch):
    """#6915: shutdown waited 10s, then dropped whatever was still queued."""
    import hindsight_hermes as plugin

    warnings: list[str] = []
    slow = HangingClient(hang=0.5)
    instance, _ = provider({"prefetch_waits_for_retain": False}, client=slow, warning_callback=warnings.append)
    for n in range(3):
        instance.sync_turn(f"turn {n}", "reply")
    with monkeypatch.context() as patch:
        patch.setattr(plugin.threading.Thread, "join", lambda self, timeout=None: None)  # shutdown gives up at once
        instance.shutdown()

    assert warnings and "will send them next time" in warnings[0]
    nxt = FlakyClient()
    second, _ = provider({"prefetch_waits_for_retain": False}, client=nxt)
    second.shutdown()
    # A set: the retain in flight at exit may land in both sessions, a replay by design.
    sent = {c[0] for c in _contents(slow.retains + nxt.retains)}
    assert sent == {"User: turn 0", "User: turn 1", "User: turn 2"}


def test_a_backlog_left_by_a_down_server_survives_exit(provider):
    """The comment on #6915: an hour of turns failed while the server was down, then Hermes quit."""
    down = FlakyClient(failures=10_000)
    instance, _ = provider({"prefetch_waits_for_retain": False}, client=down)
    for n in range(3):
        instance.sync_turn(f"turn {n}", "reply")
        instance._retain_queue.join()
    instance.shutdown()
    assert down.retains == []

    up = FlakyClient()
    second, _ = provider({"prefetch_waits_for_retain": False}, client=up)
    second.shutdown()
    assert [c[0] for c in _contents(up.retains)] == ["User: turn 0", "User: turn 1", "User: turn 2"]
    # The resend keeps each retain's operation_id, so one the server did get is a replay.
    assert down.attempts[0]["operation_id"] == up.retains[0]["operation_id"]


def _spool_one(provider, hermes_env) -> list[Path]:
    """Leave one unsent retain on disk, as a session that quit with the server down would."""
    instance, _ = provider({"prefetch_waits_for_retain": False}, client=FlakyClient(failures=10_000))
    instance.sync_turn("left over", "reply")
    instance._retain_queue.join()
    instance.shutdown()
    return list((hermes_env / "hindsight" / "unsent-retains").glob("*.json"))


@pytest.mark.parametrize("garbage", ["not json", "5", '[{"item": 1}]'])
def test_an_unreadable_spool_file_is_skipped(provider, hermes_env, garbage):
    (spooled,) = _spool_one(provider, hermes_env)
    spooled.write_text(garbage)
    up = FlakyClient()
    second, _ = provider({}, client=up)
    second.shutdown()

    assert up.retains == []  # skipped, and the session still started


def test_a_spool_file_is_sent_by_one_session_only(provider, hermes_env):
    """The gateway starts one provider per chat: only the first to claim a file sends it."""
    _spool_one(provider, hermes_env)
    first, second = FlakyClient(), FlakyClient()
    a, _ = provider({}, client=first)
    b, _ = provider({}, client=second)
    a.shutdown()
    b.shutdown()

    assert [c[0] for c in _contents(first.retains + second.retains)] == ["User: left over"]
    assert not list((hermes_env / "hindsight" / "unsent-retains").iterdir())
