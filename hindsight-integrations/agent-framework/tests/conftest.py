"""Shared fixtures for the Hindsight Agent Framework tests."""

import types
from unittest.mock import AsyncMock

from agent_framework import Message, SessionContext


def msg(role: str, text: str) -> Message:
    # NB: contents must be a *list* — a bare string is treated as a sequence of
    # characters by Message, so wrap it.
    return Message(role, [text])


def recall_response(*texts: str):
    """A stand-in for hindsight_client RecallResponse with .results[].text."""
    return types.SimpleNamespace(results=[types.SimpleNamespace(text=t) for t in texts])


def fake_client(*recall_texts: str):
    """An async-mocked Hindsight client returning the given recall texts."""
    client = AsyncMock()
    client.arecall = AsyncMock(return_value=recall_response(*recall_texts))
    client.aretain = AsyncMock(return_value=types.SimpleNamespace(success=True))
    client.acreate_bank = AsyncMock()
    return client


def session_context(*, input_texts=("hello there",), response_text=None) -> SessionContext:
    """Build a SessionContext like the framework would, optionally with a response."""
    ctx = SessionContext(input_messages=[msg("user", t) for t in input_texts])
    if response_text is not None:
        # response is framework-set + read-only; populate the backing field for tests.
        ctx._response = types.SimpleNamespace(messages=[msg("assistant", response_text)])
    return ctx
