"""Empty turns must not write, consume a batch slot, or change session state."""

import json

import hindsight_hermes as plugin
import pytest


@pytest.mark.parametrize("user,assistant", [("", ""), (" \t\n", ""), ("", "\r\n"), ("\u2003", "\u00a0")])
def test_empty_turn_does_not_enqueue_or_change_state(provider, monkeypatch, user, assistant):
    instance, fake = provider({"retain_indicator": True})
    queued, made, indicators = [], [], []
    monkeypatch.setattr(instance, "_enqueue_retain", queued.append)
    monkeypatch.setattr(instance, "_make_turn_retain_job", lambda *a, **k: made.append(k))
    instance._status_callback = indicators.append
    before = (instance._turn_counter, instance._turn_index, instance._last_retained_turn_count, instance._session_id)
    try:
        instance.sync_turn(user, assistant, session_id="empty-session")
        assert queued == made == indicators == []
        assert instance._session_turns == []
        assert (
            instance._turn_counter,
            instance._turn_index,
            instance._last_retained_turn_count,
            instance._session_id,
        ) == before
    finally:
        instance.shutdown()
    assert fake.retains == []


@pytest.mark.parametrize(
    "user,assistant,roles,contents",
    [
        ("  user fact\n", "", ["user"], ["User:   user fact\n"]),
        ("\t", "  assistant fact\n", ["assistant"], ["Assistant:   assistant fact\n"]),
        (
            "  user fact\n",
            " assistant fact\t",
            ["user", "assistant"],
            ["User:   user fact\n", "Assistant:  assistant fact\t"],
        ),
        ("0", "", ["user"], ["User: 0"]),
        ("", "\u200b", ["assistant"], ["Assistant: \u200b"]),
    ],
)
def test_nonempty_roles_and_content_are_preserved(provider, user, assistant, roles, contents):
    instance, fake = provider({})
    try:
        instance.sync_turn(user, assistant)
    finally:
        instance.shutdown()
    assert len(fake.retains) == 1
    item = fake.retains[0]["items"][0]
    messages = json.loads(item["content"])[0]
    assert [m["role"] for m in messages] == roles
    assert [m["content"] for m in messages] == contents
    assert item["metadata"]["message_count"] == str(len(roles))
    assert item["metadata"]["turn_index"] == "1"
    assert len({m["timestamp"] for m in messages}) == 1


@pytest.mark.parametrize("append", [True, False])
@pytest.mark.parametrize("finish", ["batch", "shutdown", "switch"])
def test_empty_turn_preserves_buffered_content_and_batch_cadence(provider, monkeypatch, append, finish):
    instance, fake = provider({"retain_every_n_turns": 2})
    original_document_id = instance._document_id
    monkeypatch.setattr(plugin, "_check_api_supports_update_mode_append", lambda *a, **k: append)
    try:
        instance.sync_turn("first fact", "")
        buffered = list(instance._session_turns)
        instance.sync_turn(" \n", "\t", session_id="must-not-replace-session")
        assert instance._session_turns == buffered
        assert instance._turn_counter == instance._turn_index == 1
        assert instance._session_id == "session-1"
        assert fake.retains == []
        if finish == "batch":
            instance.sync_turn("", "second fact")
        elif finish == "switch":
            instance.on_session_switch("session-2", reset=True)
    finally:
        instance.shutdown()
    if finish == "shutdown":
        # Shutdown drains queued work; it does not flush a partial in-memory batch.
        assert fake.retains == []
        assert instance._session_turns == buffered
        return
    assert len(fake.retains) == 1
    call = fake.retains[0]
    assert call["document_id"] == ("session-1" if append else original_document_id)
    item = call["items"][0]
    turns = json.loads(item["content"])
    expected = [["User: first fact"]]
    if finish == "batch":
        expected.append(["Assistant: second fact"])
    assert [[m["content"] for m in turn] for turn in turns] == expected
    assert item["metadata"]["message_count"] == str(len(expected))


def test_custom_prefixes_do_not_make_blank_content_retainable(provider):
    instance, fake = provider({"retain_user_prefix": "Human", "retain_assistant_prefix": "Agent"})
    try:
        instance.sync_turn("", "")
        instance.sync_turn("kept", "")
    finally:
        instance.shutdown()
    assert len(fake.retains) == 1
    messages = json.loads(fake.retains[0]["items"][0]["content"])[0]
    assert [m["content"] for m in messages] == ["Human: kept"]
