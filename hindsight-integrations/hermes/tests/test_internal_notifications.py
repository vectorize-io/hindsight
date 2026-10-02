"""Typed notification intake through the provider, with the existing recording client."""

import json
from typing import Any, Callable

import pytest


@pytest.mark.parametrize("kind", ["async_delegation_complete", "process_complete"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_internal_receipt_is_not_user_input_but_assistant_findings_survive(
    provider: Callable[..., Any], kind: str, batch_size: int
) -> None:
    instance, fake = provider({"retain_every_n_turns": batch_size})
    instance.sync_turn(
        "machine receipt",
        "The root cause is a missing bounds check.",
        messages=[{"role": "user", "content": "machine receipt", "display_kind": kind}],
    )
    instance.on_session_switch("session-2", reset=True)
    instance.shutdown()
    assert len(fake.retains) == 1
    item = fake.retains[0]["items"][0]
    turns = json.loads(item["content"])
    assert [[message["role"] for message in turn] for turn in turns] == [["assistant"]]
    assert turns[0][0]["content"] == "Assistant: The root cause is a missing bounds check."
    assert item["metadata"]["message_count"] == "1"


@pytest.mark.parametrize("kind", [None, "steer", "future_kind"])
def test_current_human_turn_is_kept_after_old_notification(provider: Callable[..., Any], kind: str | None) -> None:
    instance, fake = provider({})
    instance.sync_turn(
        "my correction",
        "noted",
        messages=[
            {"role": "user", "content": "old receipt", "display_kind": "async_delegation_complete"},
            {"role": "user", "content": "my correction", "display_kind": kind},
        ],
    )
    instance.shutdown()
    assert [m["role"] for m in json.loads(fake.retains[0]["items"][0]["content"])[0]] == ["user", "assistant"]


def test_uncertain_message_identity_fails_open(provider: Callable[..., Any]) -> None:
    instance, fake = provider({})
    instance.sync_turn(
        "a different genuine input",
        "answer",
        messages=[{"role": "user", "content": "old receipt", "display_kind": "async_delegation_complete"}],
    )
    instance.shutdown()
    assert json.loads(fake.retains[0]["items"][0]["content"])[0][0]["content"] == "User: a different genuine input"


def test_notification_without_assistant_content_never_buffers(provider: Callable[..., Any]) -> None:
    instance, fake = provider({"retain_every_n_turns": 3})
    instance.sync_turn(
        "receipt",
        "",
        messages=[{"role": "user", "content": "receipt", "display_kind": "process_complete"}],
    )
    instance.on_session_switch("session-2", reset=True)
    instance.shutdown()
    assert fake.retains == []
