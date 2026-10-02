"""Structured user content must still match the flattened input the host sends.

Hermes flattens the current user message to text before calling ``sync_turn``
(``run_agent._summarize_user_message_for_log``) while the ``messages`` rows keep
their original structured content, so an exact ``==`` guard never fires for a
typed receipt that carries a content-part list.
"""

import json
from typing import Any, Callable

import pytest


def _roles(fake: Any) -> list[str]:
    return [message["role"] for message in json.loads(fake.retains[0]["items"][0]["content"])[0]]


@pytest.mark.parametrize("kind", ["async_delegation_complete", "process_complete"])
def test_structured_receipt_still_excluded_from_user_messages(provider: Callable[..., Any], kind: str) -> None:
    instance, fake = provider({})
    instance.sync_turn(
        "machine receipt",
        "The root cause is a missing bounds check.",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "machine receipt"}],
                "display_kind": kind,
            }
        ],
    )
    instance.shutdown()
    assert _roles(fake) == ["assistant"]


def test_structured_genuine_input_is_not_dropped(provider: Callable[..., Any]) -> None:
    """Same shape, different text: identity is uncertain, so fail open."""
    instance, fake = provider({})
    instance.sync_turn(
        "a different genuine input",
        "answer",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "old receipt"}],
                "display_kind": "async_delegation_complete",
            }
        ],
    )
    instance.shutdown()
    assert _roles(fake) == ["user", "assistant"]


def test_multimodal_receipt_fails_open_to_user_message(provider: Callable[..., Any]) -> None:
    """Image-bearing structured content projects differently than the host text.

    The projections cannot be proven equal, so the user row is kept (fail open).
    """
    instance, fake = provider({})
    instance.sync_turn(
        "[1 image] machine receipt",
        "answer",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "machine receipt"},
                ],
                "display_kind": "process_complete",
            }
        ],
    )
    instance.shutdown()
    assert _roles(fake) == ["user", "assistant"]
