"""Shared fixtures/helpers for the Aider wrapper tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


def make_recall_response(texts):
    return SimpleNamespace(results=[SimpleNamespace(text=t) for t in texts])


def make_client(texts=None) -> MagicMock:
    client = MagicMock()
    client.recall = MagicMock(return_value=make_recall_response(texts or []))
    client.retain = MagicMock()
    return client
